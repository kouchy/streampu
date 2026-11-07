#ifdef SPU_JULIA

// When enabled, unsafe Julia codelets (`set_jl_safety(false)`) are compiled into a typed `@cfunction` native C
// entry point and invoked directly via a C function pointer (`invoke_trampoline`), bypassing `jl_call`'s dynamic
// multiple-dispatch lookup, GC frame push/pop, and `Int32` return boxing/unboxing (~20-25 ns saved per call).
// Comment out this macro to fall back to `jl_call` in unsafe mode.
#define SPU_JULIA_TRAMPOLINE

#include <atomic>
#include <cstdint>
#include <functional>
#include <julia.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <typeindex>
#include <vector>

#include "Module/Module_Julia.hpp"
#include "Runtime/Socket/Socket.hpp"
#include "Tools/Exception/exception.hpp"
#include "Tools/Julia/Julia.hpp"
#include "Tools/Julia/Proxy.hpp"
#include "Tools/Julia/Ref_keeper.hpp"

using namespace spu;
using namespace spu::module;

jl_value_t*
Module_Julia::get_jl_datatype(const runtime::Socket* sck)
{
    // Resolve the C++ socket's runtime `std::type_index` into the corresponding built-in Julia primitive `DataType`.
    std::type_index type = sck->get_datatype();
    if (type == typeid(int8_t))
        return (jl_value_t*)jl_int8_type;
    else if (type == typeid(int16_t))
        return (jl_value_t*)jl_int16_type;
    else if (type == typeid(int32_t))
        return (jl_value_t*)jl_int32_type;
    else if (type == typeid(int64_t))
        return (jl_value_t*)jl_int64_type;
    else if (type == typeid(uint8_t))
        return (jl_value_t*)jl_uint8_type;
    else if (type == typeid(uint16_t))
        return (jl_value_t*)jl_uint16_type;
    else if (type == typeid(uint32_t))
        return (jl_value_t*)jl_uint32_type;
    else if (type == typeid(uint64_t))
        return (jl_value_t*)jl_uint64_type;
    else if (type == typeid(float))
        return (jl_value_t*)jl_float32_type;
    else if (type == typeid(double))
        return (jl_value_t*)jl_float64_type;
    else
    {
        std::stringstream message;
        message << "Unsupported socket datatype ('sck->get_datatype_string()' = " << sck->get_datatype_string() << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }
}

tools::Proxy
Module_Julia::jl_new_array_from_data(const runtime::Socket* sck, size_t n_elmts)
{
    void* p = (void*)sck->get_dataptr();
    jl_value_t* datatype = Module_Julia::get_jl_datatype(sck);
    tools::Proxy r = tools::Proxy((jl_value_t*)jl_ptr_to_array_1d(jl_apply_array_type(datatype, 1), p, n_elmts, 0));
    return r;
}

jl_value_t*
Module_Julia::get_or_create_socket_array(const size_t tid, const size_t s, void* p, const size_t n_elmts)
{
    auto& cache = this->jl_sockets_cache[tid][s];
    const size_t n_entries = cache.entries.size();
    if (n_entries > 0)
    {
        // 1. O(1) cyclic fast path: in multi-wave execution (`w = 0, 1, ..., W-1`) or Pipeline adaptor ring buffers,
        //    consecutive calls step sequentially through `cache.entries`. Checking `(last_hit_idx + 1) % n_entries`
        //    first resolves the lookup in a single comparison without scanning.
        const size_t next_idx = (cache.last_hit_idx + 1) % n_entries;
        const auto& next_entry = cache.entries[next_idx];
        if (next_entry.dataptr == p && next_entry.n_elmts == n_elmts)
        {
            cache.last_dataptr = p;
            cache.last_hit_idx = next_idx;
            return next_entry.jl_array;
        }

        // 2. Linear scan fallback across all cached entries for this socket.
        for (size_t i = 0; i < n_entries; i++)
        {
            const auto& entry = cache.entries[i];
            if (entry.dataptr == p && entry.n_elmts == n_elmts)
            {
                cache.last_dataptr = p;
                cache.last_hit_idx = i;
                return entry.jl_array;
            }
        }
    }

    // 3. Stale-length reuse: if `n_elmts` changed after `eval()` (e.g., `Sequence` or `Pipeline` reconfigured
    //    `set_n_frames()`), any entry created with the previous `n_elmts` is obsolete. Reuse its `Ref_keeper`
    //    slot in place via `update_value()` instead of consuming a new GC root slot.
    //    Note: `jl_gc_enable(0)` is held briefly around `jl_ptr_to_array_1d()` + `Ref_keeper` registration so
    //    a concurrent Stop-The-World GC sweep cannot collect `jl_arr` before it is rooted in `marks`.
    for (size_t i = 0; i < n_entries; i++)
    {
        auto& entry = cache.entries[i];
        if (entry.n_elmts != n_elmts)
        {
            auto gc_state = jl_gc_enable(0);
            jl_value_t* jl_arr = (jl_value_t*)jl_ptr_to_array_1d(cache.array_type, p, n_elmts, 0);
            tools::Ref_keeper::get()->update_value(entry.jl_id, jl_arr);
            jl_gc_enable(gc_state);

            entry.dataptr = p;
            entry.n_elmts = n_elmts;
            entry.jl_array = jl_arr;
            cache.last_dataptr = p;
            cache.last_hit_idx = i;
            return jl_arr;
        }
    }

    // 4. Cache miss: allocate a new 1D Julia `Array` wrapper (`own_buffer = 0`, so C++ retains ownership of `p`).
    //    Scale `max_entries` with `get_n_waves()` so multi-wave tasks coupled to Pipeline ring buffers never thrash,
    //    and use round-robin (FIFO) eviction via `update_value()` if `max_entries` is ever reached.
    const size_t max_entries = std::max<size_t>(64, this->self_module->get_n_waves() * 16);
    auto gc_state = jl_gc_enable(0);
    jl_value_t* jl_arr = (jl_value_t*)jl_ptr_to_array_1d(cache.array_type, p, n_elmts, 0);
    if (n_entries < max_entries)
    {
        size_t jl_id = tools::Julia::preserve(jl_arr);
        this->jl_constants_id[tid].push_back(jl_id);
        cache.entries.push_back({ p, n_elmts, jl_arr, jl_id });
        cache.last_hit_idx = n_entries;
    }
    else
    {
        const size_t evict_idx = cache.next_evict_idx % max_entries;
        cache.next_evict_idx = (evict_idx + 1) % max_entries;
        auto& victim = cache.entries[evict_idx];
        tools::Ref_keeper::get()->update_value(victim.jl_id, jl_arr);
        victim.dataptr = p;
        victim.n_elmts = n_elmts;
        victim.jl_array = jl_arr;
        cache.last_hit_idx = evict_idx;
    }
    jl_gc_enable(gc_state);

    cache.last_dataptr = p;
    return jl_arr;
}

size_t
Module_Julia::get_task_id(runtime::Task& task)
{
    auto& module = task.get_module();
    for (size_t tid = 0; tid < module.tasks.size(); tid++)
        if (&task == module.tasks[tid].get()) return tid;

    std::stringstream message;
    message << "Task is not in the current module, this should never happen.";
    throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
}

Module_Julia::Module_Julia(Module* self_module, Module_Julia& (*to_module_julia)(Module&), const bool jl_safe)
  : self_module(self_module)
  , to_module_julia(to_module_julia)
  , jl_states_ptr(std::vector<jl_value_t*>())
  , jl_states_id(std::vector<size_t>())
  , jl_constants_ptr(std::vector<std::vector<jl_value_t*>>())
  , jl_constants_id(std::vector<std::vector<size_t>>())
  , jl_sockets_cache(std::vector<std::vector<Socket_jl_cache>>())
  , jl_func_args(std::vector<std::vector<void*>>())
  , evaluated(bool(false))
  , jl_eval_thread_keeper(nullptr)
  , n_clones(new size_t(0))
  , jl_evaluate_states(new std::vector<std::function<void()>>())
  , jl_create_states(new std::vector<std::function<void(Module_Julia& m)>>())
  , jl_create_constants(new std::vector<std::vector<std::function<void(Module_Julia& m)>>>())
  , jl_evaluate(new std::vector<std::function<void(Module_Julia& m)>>())
  , jl_create_codelet(new std::vector<std::function<void(Module_Julia& m)>>())
  , jl_safe(jl_safe)
{
}

Module_Julia::~Module_Julia()
{
    if (this->jl_eval_thread_keeper != nullptr && this->jl_eval_thread_keeper == (void*)tools::Ref_keeper::get())
    {
        for (size_t jl_id : this->jl_states_id)
            tools::Julia::release(jl_id);
        for (size_t tid = 0; tid < this->jl_constants_id.size(); tid++)
            for (size_t jl_id : this->jl_constants_id[tid])
                tools::Julia::release(jl_id);
    }
    (*this->n_clones)--;
}

void
Module_Julia::deep_copy(const Module_Julia& /*m*/, Module* self_module)
{
    this->self_module = self_module;
    (*this->n_clones)++;

    this->jl_states_ptr.clear();
    this->jl_states_id.clear();
    for (auto& ptrs_list : this->jl_constants_ptr)
        ptrs_list.clear();
    for (auto& ids_list : this->jl_constants_id)
        ids_list.clear();
    for (auto& scks_cache_list : this->jl_sockets_cache)
        scks_cache_list.clear();
    for (auto& funcs_list : this->jl_func_args)
        funcs_list.clear();

    this->evaluated = false;
    this->jl_eval_thread_keeper = nullptr;
}

bool
Module_Julia::is_eval() const
{
    return this->evaluated;
}

void
Module_Julia::set_jl_safety(const bool jl_safe)
{
    this->jl_safe = jl_safe;
}

bool
Module_Julia::is_jl_safe()
{
    return this->jl_safe;
}

const std::vector<jl_value_t*>&
Module_Julia::_get_states() const
{
    return this->jl_states_ptr;
}

void
Module_Julia::_create_state(const std::string& julia_constructor, const bool shared, const std::string& julia_code)
{
    if ((*this->n_clones) > 0)
    {
        std::stringstream message;
        message << "It is not possible to create a new state when clones exist ('n_clones' = " << (*this->n_clones)
                << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    if (!julia_code.empty())
    {
        auto code_evaluated = std::make_shared<std::atomic<bool>>(false);
        this->jl_evaluate_states->push_back(
          [julia_code, code_evaluated]()
          {
              if (!code_evaluated->exchange(true)) tools::Julia::eval_string(julia_code);
          });
    }

    this->_register_state(julia_constructor, shared);
}

void
Module_Julia::_create_state_file(const std::string& julia_constructor,
                                 const bool shared,
                                 const std::string& julia_filepath)
{
    if ((*this->n_clones) > 0)
    {
        std::stringstream message;
        message << "It is not possible to create a new state when clones exist ('n_clones' = " << (*this->n_clones)
                << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    if (!julia_filepath.empty())
    {
        auto file_evaluated = std::make_shared<std::atomic<bool>>(false);
        this->jl_evaluate_states->push_back(
          [julia_filepath, file_evaluated]()
          {
              if (!file_evaluated->exchange(true)) tools::Julia::eval_file(julia_filepath);
          });
    }

    this->_register_state(julia_constructor, shared);
}

void
Module_Julia::_register_state(const std::string& julia_constructor, const bool shared)
{
    if (shared)
    {
        auto shared_ptr = std::make_shared<jl_value_t*>(nullptr);
        this->jl_create_states->push_back(
          [julia_constructor, shared_ptr](Module_Julia& m)
          {
              if (*shared_ptr == nullptr)
              {
                  jl_value_t* jl_state_ptr = tools::Julia::eval_string(julia_constructor);
                  if (jl_state_ptr == nullptr)
                  {
                      std::stringstream message;
                      message << "Julia state constructor ('" << julia_constructor << "') returned nullptr.";
                      throw tools::julia_error(__FILE__, __LINE__, __func__, message.str());
                  }
                  size_t jl_state_id = tools::Julia::preserve(jl_state_ptr);
                  m.jl_states_id.push_back(jl_state_id);
                  *shared_ptr = jl_state_ptr;
              }
              m.jl_states_ptr.push_back(*shared_ptr);
          });
    }
    else
    {
        this->jl_create_states->push_back(
          [julia_constructor](Module_Julia& m)
          {
              jl_value_t* jl_state_ptr = tools::Julia::eval_string(julia_constructor);
              if (jl_state_ptr == nullptr)
              {
                  std::stringstream message;
                  message << "Julia state constructor ('" << julia_constructor << "') returned nullptr.";
                  throw tools::julia_error(__FILE__, __LINE__, __func__, message.str());
              }
              size_t jl_state_id = tools::Julia::preserve(jl_state_ptr);
              m.jl_states_ptr.push_back(jl_state_ptr);
              m.jl_states_id.push_back(jl_state_id);
          });
    }
}

void
Module_Julia::eval()
{
    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    if (this->is_eval())
    {
        std::stringstream message;
        message << "This module has already been evaluated.";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    if ((*this->jl_evaluate).size() == 0)
    {
        std::stringstream message;
        message << "There is no Julia code to evaluate, please call the 'create_codelet()' method first.";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    // 1. Evaluate task files/strings and state type definitions in Main
    for (size_t tid = 0; tid < this->self_module->tasks.size(); tid++)
        (*this->jl_evaluate)[tid](*this);
    for (auto& eval_state_def : *this->jl_evaluate_states)
        eval_state_def();

    // 2. Instantiate and GC-preserve module states for this instance/clone
    for (auto& create_state : *this->jl_create_states)
        create_state(*this);

    // 3. Create task constants and bind jl_func_args (including module states at indices 1..K)
    for (size_t tid = 0; tid < this->self_module->tasks.size(); tid++)
    {
        for (auto& create_constant : (*this->jl_create_constants)[tid])
            create_constant(*this);
        (*this->jl_create_codelet)[tid](*this);
    }

    this->evaluated = true;
    this->jl_eval_thread_keeper = (void*)tools::Ref_keeper::get();
}

runtime::Task&
Module_Julia::_create_task(const std::string& name)
{
    if ((*this->n_clones) > 0)
    {
        std::stringstream message;
        message << "It is not possible to create a new task when clones exist ('n_clones' = " << (*this->n_clones)
                << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    this->jl_create_constants->push_back(std::vector<std::function<void(Module_Julia & m)>>());
    this->jl_constants_ptr.push_back(std::vector<jl_value_t*>());
    this->jl_constants_id.push_back(std::vector<size_t>());
    this->jl_sockets_cache.push_back(std::vector<Socket_jl_cache>());
    this->jl_func_args.push_back(std::vector<void*>());

    return this->self_module->create_task(name);
}

void
Module_Julia::create_codelet(runtime::Task& task, const std::string& julia_code, const std::string& julia_function_name)
{
    if ((*this->n_clones) > 0)
    {
        std::stringstream message;
        message << "It is not possible to create a new codelet when clones exist ('n_clones' = " << (*this->n_clones)
                << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    this->jl_evaluate->push_back([julia_code](Module_Julia&) { jl_eval_string(julia_code.c_str()); });
    this->_create_codelet(task, julia_function_name);
}

void
Module_Julia::create_cdl(runtime::Task& task, const std::string& julia_code, const std::string& julia_function_name)
{
    this->create_codelet(task, julia_code, julia_function_name);
}

void
Module_Julia::create_codelet_file(runtime::Task& task,
                                  const std::string& julia_filepath,
                                  const std::string& julia_function_name)
{
    if ((*this->n_clones) > 0)
    {
        std::stringstream message;
        message << "It is not possible to create a new codelet when clones exist ('n_clones' = " << (*this->n_clones)
                << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    this->jl_evaluate->push_back([julia_filepath](Module_Julia&) { tools::Julia::eval_file(julia_filepath); });
    this->_create_codelet(task, julia_function_name);
}

void
Module_Julia::create_cdl_file(runtime::Task& task,
                              const std::string& julia_filepath,
                              const std::string& julia_function_name)
{
    this->create_codelet_file(task, julia_filepath, julia_function_name);
}

void
Module_Julia::create_codelet_function(runtime::Task& task, jl_function_t* f)
{
    if ((*this->n_clones) > 0)
    {
        std::stringstream message;
        message << "It is not possible to create a new codelet when clones exist ('n_clones' = " << (*this->n_clones)
                << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    size_t tid = Module_Julia::get_task_id(task);

    this->jl_evaluate->push_back(
      [f, tid](Module_Julia& m)
      {
          size_t f_id = tools::Julia::preserve(f);
          m.jl_constants_ptr[tid].push_back(f);
          m.jl_constants_id[tid].push_back(f_id);
      });
    this->_create_codelet(task);
}

void
Module_Julia::create_cdl_function(runtime::Task& task, jl_function_t* f)
{
    this->create_codelet_function(task, f);
}

#ifdef SPU_JULIA_TRAMPOLINE
// Maximum number of Julia object arguments (states + sockets + constants) unpacked directly into C ABI arguments
// (`a[0], a[1], ..., a[K], frame_id, n_frames_per_wave`) via C++11 variadic templates.
// Above 16 object arguments, `_spu_make_trampoline` automatically emits a two-stage trampoline (`_spu_trampoline_outer`
// -> `ccall` -> `_spu_trampoline_inner`) accepting `func_args.data()` (`void* const*`) as a single pointer argument.
static constexpr size_t SPU_MAX_DIRECT_TRAMPOLINE_ARGS = 16;

// C++11 compile-time integer sequence (`std::index_sequence` equivalent, which was only introduced in C++14)
// used to unpack `a[0], a[1], ..., a[K]` into a typed C function pointer call.
template<size_t... Is>
struct Index_seq
{
};

template<size_t N, size_t... Is>
struct Make_index_seq : Make_index_seq<N - 1, N - 1, Is...>
{
};

template<size_t... Is>
struct Make_index_seq<0, Is...> : Index_seq<Is...>
{
};

template<size_t>
using Void_ptr_t = void*;

// Unpacks `sizeof...(Is)` pointers (`a[0]` = Julia function singleton, `a[1..K]` = states/sockets/constants)
// followed by the two unboxed runtime `uint32_t` scalars (`frame_id` and `n_frames_per_wave`) into a direct C call.
template<size_t... Is>
static inline int32_t
call_trampoline_n(void* cfunc, void* const* a, const uint32_t fid, const uint32_t nfw, Index_seq<Is...>)
{
    using fn_t = int32_t (*)(Void_ptr_t<Is>..., uint32_t, uint32_t);
    return ((fn_t)cfunc)(a[Is]..., fid, nfw);
}

// Base case (`n_obj_args > SPU_MAX_DIRECT_TRAMPOLINE_ARGS`): invokes the outer `@cfunction` trampoline by passing
// the base address of `func_args` (`void* const* a`) alongside `fid` and `nfw`.
template<size_t I>
static inline typename std::enable_if<(I > SPU_MAX_DIRECT_TRAMPOLINE_ARGS), int32_t>::type
invoke_trampoline(void* cfunc, void* const* a, const uint32_t fid, const uint32_t nfw, const size_t /*n_obj_args*/)
{
    return ((int32_t(*)(void* const*, uint32_t, uint32_t))cfunc)(a, fid, nfw);
}

// Recursive compile-time dispatch for `0 <= n_obj_args <= SPU_MAX_DIRECT_TRAMPOLINE_ARGS`.
// Compiles down to a dense jump table where each branch invokes `cfunc` with the exact arity `I + 1`
// (`a[0]` for `f`, plus `a[1..I]` for the `I` object arguments).
template<size_t I = 0>
static inline typename std::enable_if<(I <= SPU_MAX_DIRECT_TRAMPOLINE_ARGS), int32_t>::type
invoke_trampoline(void* cfunc, void* const* a, const uint32_t fid, const uint32_t nfw, const size_t n_obj_args)
{
    if (n_obj_args == I) return call_trampoline_n(cfunc, a, fid, nfw, Make_index_seq<I + 1>{});
    return invoke_trampoline<I + 1>(cfunc, a, fid, nfw, n_obj_args);
}

// Invokes the safe `@cfunction` trampoline (`_spu_trampoline_safe_N`), which wraps the inner typed trampoline call
// inside a Julia `try ... catch` block and returns a packed `Int64`:
// - Lower 32 bits: the zero-extended `Int32` status returned by the Julia codelet.
// - Upper 32 bits: `0` on success, `1` if a Julia exception was caught (in which case `_spu_take_exception()` retrieves
//   the formatted exception + backtrace string for the current task and rethrows it as a C++ `tools::julia_error`).
static inline int32_t
invoke_safe_trampoline(void* safe_cfunc, void* const* a, const uint32_t fid, const uint32_t nfw)
{
    const int64_t ret = ((int64_t(*)(void* const*, uint32_t, uint32_t))safe_cfunc)(a, fid, nfw);
    if (ret >> 32)
    {
        jl_function_t* take_exc_fn = jl_get_function(jl_main_module, "_spu_take_exception");
        jl_value_t* msg_val = take_exc_fn ? jl_call0(take_exc_fn) : nullptr;
        const char* msg = (msg_val && jl_is_string(msg_val)) ? jl_string_ptr(msg_val) : "Julia exception occurred.";
        throw tools::julia_error(__FILE__, __LINE__, __func__, msg);
    }
    return (int32_t)ret;
}

// Compiles (or retrieves from `_SPU_TRAMPOLINE_CACHE`) both the unsafe and safe native C entry points (`@cfunction`)
// for the concrete runtime signature `(typeof(f), typeof(arg_1), ..., typeof(arg_K), UInt32, UInt32)`.
//
// Key design details:
// 1. `_spu_trampoline_inner_N` declares every object argument with its exact concrete Julia `DataType`
//    (`f::FType, a1::T1, ..., aK::TK, frame_id::UInt32, n_frames_per_wave::UInt32`) and is compiled with
//    `@cfunction(..., Int32, (Ref{FType}, Ref{T1}, ..., Ref{TK}, UInt32, UInt32))`.
//    In Julia's `@cfunction` ABI, `Ref{T}` accepts a raw `jl_value_t*` (`void*`) pointer from C++ for any `T`
//    (both mutable heap objects like `Vector{T}` and boxed `isbits` constants like `UInt32`/`Float64`) and
//    passes `a_i::T_i` as genuine Julia function parameters.
// 2. Because `a_i::T_i` are function parameters already rooted by C++ (`Ref_keeper`), Julia emits ZERO GC frame
//    push/pop (`%gcframe`) inside `_spu_trampoline_inner_N`, and LLVM attaches `nonnull align 8 dereferenceable(24)`
//    and `!tbaa_arraybuf` metadata to `Vector{T}` arguments, preserving full SIMD vectorization when `@inline f(...)`
//    is inlined into the trampoline.
// 3. When `n_obj_args > 16`, `_spu_make_trampoline` wraps `_spu_trampoline_inner_N` inside a tiny outer `@cfunction`
//    (`_spu_trampoline_outer_N(args::Ptr{Ptr{Cvoid}}, ...)`) that loads the pointers from `args` and invokes
//    `_spu_trampoline_inner_N` via `ccall`, preserving 100% of the inner function's SIMD vectorization and zero GC
//    frame.
// 4. For safe mode (`set_jl_safety(true)`), `_spu_make_trampoline` also compiles `_spu_trampoline_safe_N`, which wraps
//    a `ccall` to `_spu_trampoline_inner_N` inside a `try ... catch` block. Calling `_spu_trampoline_inner_N` via
//    `ccall` inside `try` prevents `sigsetjmp` (`returns_twice`) from spilling `Vector{T}` pointers to `volatile` stack
//    slots, thereby preserving 100% of SIMD vectorization and eliminating `%gcframe` and tuple allocations in safe
//    mode.
static void
get_or_compile_trampoline(const std::vector<void*>& func_args, void*& out_unsafe_cfunc, void*& out_safe_cfunc)
{
    jl_function_t* make_trampoline_fn = jl_get_function(jl_main_module, "_spu_make_trampoline");
    if (make_trampoline_fn == nullptr)
    {
        static const char* trampoline_gen_code = R"""(
const _SPU_TRAMPOLINE_CACHE = Dict{Any, Tuple{Ptr{Cvoid}, Ptr{Cvoid}}}()
const _SPU_TRAMPOLINE_COUNTER = Ref{Int}(0)
const _SPU_ERR_LOCK = Threads.SpinLock()
const _SPU_ERR_MSGS = Dict{UInt, String}()

@noinline function _spu_record_exception(@nospecialize(e), bt)
    msg = sprint(Base.showerror, e, bt)
    key = UInt(pointer_from_objref(current_task()))
    lock(_SPU_ERR_LOCK)
    try
        _SPU_ERR_MSGS[key] = msg
    finally
        unlock(_SPU_ERR_LOCK)
    end
    return nothing
end

function _spu_take_exception()::String
    key = UInt(pointer_from_objref(current_task()))
    lock(_SPU_ERR_LOCK)
    try
        return pop!(_SPU_ERR_MSGS, key, "Unknown Julia exception")
    finally
        unlock(_SPU_ERR_LOCK)
    end
end

function _spu_make_trampoline(raw_args_ptr::Ptr{Cvoid}, n_types::Int, raw_out_ptrs::Ptr{Cvoid})::Nothing
    args_ptr = Ptr{Ptr{Cvoid}}(raw_args_ptr)
    out_ptrs = Ptr{Ptr{Cvoid}}(raw_out_ptrs)
    types = DataType[typeof(unsafe_pointer_to_objref(unsafe_load(args_ptr, i))) for i in 1:n_types]
    key = Tuple(types)
    cached = get(_SPU_TRAMPOLINE_CACHE, key, (C_NULL, C_NULL))
    if cached[1] != C_NULL
        unsafe_store!(out_ptrs, cached[1], 1)
        unsafe_store!(out_ptrs, cached[2], 2)
        return nothing
    end
    _SPU_TRAMPOLINE_COUNTER[] += 1
    id = _SPU_TRAMPOLINE_COUNTER[]
    inner_name = Symbol("_spu_trampoline_inner_", id)
    n_obj_args = n_types - 1
    FType = types[1]
    arg_syms = [Symbol("a", i) for i in 1:n_obj_args]
    typed_params = [:($(arg_syms[i])::$(types[i + 1])) for i in 1:n_obj_args]
    ref_types = [:(Ref{$(types[i])}) for i in 1:n_types]
    inner_def = quote
        function $inner_name(f::$FType, $(typed_params...), frame_id::UInt32, n_frames_per_wave::UInt32)::Int32
            return (@inline f($(arg_syms...), frame_id, n_frames_per_wave)) % Int32
        end
    end
    Core.eval(Main, inner_def)
    inner_cfunc_expr = :(@cfunction($inner_name, Int32, ($(ref_types...), UInt32, UInt32)))
    inner_ptr = Core.eval(Main, inner_cfunc_expr)::Ptr{Cvoid}
    unsafe_ptr = inner_ptr

    ptr_types = [:(Ptr{Cvoid}) for _ in 1:n_types]
    load_exprs = [:(unsafe_load(args, $i)) for i in 1:n_types]

    if n_obj_args > 16
        outer_name = Symbol("_spu_trampoline_outer_", id)
        outer_def = quote
            function $outer_name(args::Ptr{Ptr{Cvoid}}, frame_id::UInt32, n_frames_per_wave::UInt32)::Int32
                return ccall($inner_ptr, Int32, ($(ptr_types...), UInt32, UInt32), $(load_exprs...), frame_id, n_frames_per_wave)
            end
        end
        Core.eval(Main, outer_def)
        outer_cfunc_expr = :(@cfunction($outer_name, Int32, (Ptr{Ptr{Cvoid}}, UInt32, UInt32)))
        unsafe_ptr = Core.eval(Main, outer_cfunc_expr)::Ptr{Cvoid}
    end

    safe_name = Symbol("_spu_trampoline_safe_", id)
    safe_def = quote
        function $safe_name(args::Ptr{Ptr{Cvoid}}, frame_id::UInt32, n_frames_per_wave::UInt32)::Int64
            try
                status = ccall($inner_ptr, Int32, ($(ptr_types...), UInt32, UInt32), $(load_exprs...), frame_id, n_frames_per_wave)
                return Int64(reinterpret(UInt32, status))
            catch e
                _spu_record_exception(e, catch_backtrace())
                return Int64(1) << 32
            end
        end
    end
    Core.eval(Main, safe_def)
    safe_cfunc_expr = :(@cfunction($safe_name, Int64, (Ptr{Ptr{Cvoid}}, UInt32, UInt32)))
    safe_ptr = Core.eval(Main, safe_cfunc_expr)::Ptr{Cvoid}

    _SPU_TRAMPOLINE_CACHE[key] = (unsafe_ptr, safe_ptr)
    unsafe_store!(out_ptrs, unsafe_ptr, 1)
    unsafe_store!(out_ptrs, safe_ptr, 2)
    return nothing
end
)""";
        (void)tools::Julia::eval_string(trampoline_gen_code);
        make_trampoline_fn = jl_get_function(jl_main_module, "_spu_make_trampoline");
    }

    const size_t n_types = func_args.size() - 2;
    void* out_ptrs[2] = { nullptr, nullptr };
    (void)jl_call3(make_trampoline_fn,
                   jl_box_voidpointer((void*)func_args.data()),
                   jl_box_int64((int64_t)n_types),
                   jl_box_voidpointer((void*)out_ptrs));

    if (jl_exception_occurred() || out_ptrs[0] == nullptr || out_ptrs[1] == nullptr)
    {
        jl_exception_clear();
        std::stringstream message;
        message << "Failed to compile '@cfunction' trampoline for Julia codelet.";
        throw tools::julia_error(__FILE__, __LINE__, __func__, message.str());
    }
    out_unsafe_cfunc = out_ptrs[0];
    out_safe_cfunc = out_ptrs[1];
}
#endif

void
Module_Julia::_create_codelet(runtime::Task& task, const std::string& julia_function_name)
{
    size_t tid = Module_Julia::get_task_id(task);
    auto to_jl = this->to_module_julia;

    // Shared compilation flag across the original task and all its thread clones
    auto task_compiled = std::make_shared<std::atomic<bool>>(false);
#ifdef SPU_JULIA_TRAMPOLINE
    // Native C function pointers (unsafe and safe) produced by `@cfunction` once on the first execution and shared
    // across all clones
    auto task_cfunc = std::make_shared<std::atomic<void*>>(nullptr);
    auto task_safe_cfunc = std::make_shared<std::atomic<void*>>(nullptr);
#endif

    this->self_module->create_codelet(
      task,
#ifdef SPU_JULIA_TRAMPOLINE
      [tid, to_jl, task_compiled, task_cfunc, task_safe_cfunc](
        module::Module& m, runtime::Task& t, const size_t frame_id) -> int
#else
      [tid, to_jl, task_compiled](module::Module& m, runtime::Task& t, const size_t frame_id) -> int
#endif
      {
          auto& sjl = to_jl(m);

          if (!sjl.is_eval())
          {
              std::stringstream message;
              message << "The task cannot be executed because the module has not been evaluated.";
              throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
              return 0;
          }

          // Update `func_args` socket entries only when a socket's underlying C++ data pointer or byte size changes
          // since the last execution (e.g., multi-wave execution, Pipeline adaptor double-buffering, or dynamic
          // rebinding). For single-wave static Sequence chains, `cache.last_dataptr == p` hits on every iteration
          // with zero Julia C API calls and zero GC root mutations.
          const size_t n_states = sjl.jl_states_ptr.size();
          const size_t n_frames_per_wave = m.get_n_frames_per_wave();
          const size_t n_scks = t.sockets.size() - 1;
          auto& scks_cache = sjl.jl_sockets_cache[tid];
          auto& func_args = sjl.jl_func_args[tid];
          for (size_t s = 0; s < n_scks; s++)
          {
              const runtime::Socket* sck = t.sockets[s].get();
              void* p = (void*)sck->get_dataptr();
              const size_t databytes = sck->get_databytes();
              auto& cache = scks_cache[s];
              if (cache.last_dataptr != p || cache.last_databytes != databytes ||
                  cache.last_n_frames_per_wave != n_frames_per_wave)
              {
                  const size_t n_elmts = sck->get_n_elmts() * n_frames_per_wave / m.get_n_frames();
                  cache.last_databytes = databytes;
                  cache.last_n_frames_per_wave = n_frames_per_wave;
                  func_args[1 + n_states + s] = (void*)sjl.get_or_create_socket_array(tid, s, p, n_elmts);
              }
          }

#ifdef SPU_JULIA_TRAMPOLINE
          // If the Julia function's `@cfunction` trampolines (unsafe and safe) have not been JIT compiled yet:
          if (!task_compiled->load(std::memory_order_acquire))
          {
              static std::mutex compile_mtx;
              int8_t prev_gc_state = 0;
              const bool can_safe = (jl_get_pgcstack() != nullptr);
              // Enter GC_SAFE state while waiting on the compilation lock
              // so that if another thread triggers a GC during JIT compilation,
              // this waiting thread will NOT block the Stop-The-World GC.
              if (can_safe) prev_gc_state = jl_gc_safe_enter(jl_current_task->ptls);

              std::unique_lock<std::mutex> lock(compile_mtx);

              if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, prev_gc_state);

              if (!task_compiled->load(std::memory_order_relaxed))
              {
                  void* cfunc = nullptr;
                  void* safe_cfunc = nullptr;
                  get_or_compile_trampoline(func_args, cfunc, safe_cfunc);
                  task_cfunc->store(cfunc, std::memory_order_release);
                  task_safe_cfunc->store(safe_cfunc, std::memory_order_release);
                  int32_t status = sjl.is_jl_safe()
                                     ? invoke_safe_trampoline(
                                         safe_cfunc, func_args.data(), (uint32_t)frame_id, (uint32_t)n_frames_per_wave)
                                     : invoke_trampoline(cfunc,
                                                         func_args.data(),
                                                         (uint32_t)frame_id,
                                                         (uint32_t)n_frames_per_wave,
                                                         func_args.size() - 3);
                  task_compiled->store(true, std::memory_order_release);
                  return status;
              }
          }

          if (sjl.is_jl_safe())
          {
              void* safe_cfunc = task_safe_cfunc->load(std::memory_order_relaxed);
              return invoke_safe_trampoline(
                safe_cfunc, func_args.data(), (uint32_t)frame_id, (uint32_t)n_frames_per_wave);
          }
          void* cfunc = task_cfunc->load(std::memory_order_relaxed);
          return invoke_trampoline(
            cfunc, func_args.data(), (uint32_t)frame_id, (uint32_t)n_frames_per_wave, func_args.size() - 3);
#else
          // Mutate the pre-allocated heap `UInt32` boxes in place before calling `jl_call` or `tools::Julia::call`.
          *(uint32_t*)func_args[func_args.size() - 2] = (uint32_t)frame_id;
          *(uint32_t*)func_args[func_args.size() - 1] = (uint32_t)n_frames_per_wave;

          // If the Julia function has not been JIT compiled yet:
          if (!task_compiled->load(std::memory_order_acquire))
          {
              static std::mutex compile_mtx;
              int8_t prev_gc_state = 0;
              const bool can_safe = (jl_get_pgcstack() != nullptr);
              // Enter GC_SAFE state while waiting on the compilation lock
              // so that if another thread triggers a GC during JIT compilation,
              // this waiting thread will NOT block the Stop-The-World GC.
              if (can_safe) prev_gc_state = jl_gc_safe_enter(jl_current_task->ptls);

              std::unique_lock<std::mutex> lock(compile_mtx);

              if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, prev_gc_state);

              if (!task_compiled->load(std::memory_order_relaxed))
              {
                  // In unsafe mode (`!sjl.is_jl_safe()`), call `jl_call` directly on the pre-rooted `func_args`
                  // vector without going through `Ref_keeper::call` (`JL_GC_PUSHARGS` + `StreamPU.call` `Ref{Any}`
                  // assignment + `StreamPU.unref`). Because the return value of a StreamPU codelet is an `Int32`
                  // (`runtime::status_t`), `jl_box_int32` returns a pre-allocated singleton for `-512..511` or a
                  // thread-local pool box read immediately by `jl_unbox_int32(res)` before any GC safepoint.
                  jl_value_t* res = sjl.is_jl_safe() ? tools::Julia::call(func_args, true)
                                                     : jl_call((jl_function_t*)func_args[0],
                                                               (jl_value_t**)&func_args[1],
                                                               (uint32_t)(func_args.size() - 1));
                  task_compiled->store(true, std::memory_order_release);
                  return jl_unbox_int32(res);
              }
          }

          jl_value_t* res =
            sjl.is_jl_safe()
              ? tools::Julia::call(func_args, true)
              : jl_call((jl_function_t*)func_args[0], (jl_value_t**)&func_args[1], (uint32_t)(func_args.size() - 1));
          return jl_unbox_int32(res);
#endif
      });

    this->jl_create_codelet->push_back(
      [tid, julia_function_name](Module_Julia& m)
      {
          runtime::Task& task = *(m.self_module->tasks[tid].get());
          if (m.jl_func_args[tid].size() > 0)
          {
              std::stringstream message;
              message << "jl_func_args[tid] has to be empty ('jl_func_args[tid].size()' = "
                      << m.jl_func_args[tid].size() << ", 'tid' = " << tid << ").";
              throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
          }

          const size_t n_states = m.jl_states_ptr.size();
          const size_t n_csts = m.jl_constants_ptr[tid].size();
          const size_t n_scks = task.sockets.size() > 0 ? task.sockets.size() - 1 : 0;
          m.jl_func_args[tid].resize(1 + n_states + task.sockets.size() + n_csts + 1);
          m.jl_sockets_cache[tid].resize(n_scks);

          if (m.jl_func_args[tid][0] == nullptr)
          {
              const std::string& fn_name = julia_function_name.empty() ? task.get_name() : julia_function_name;
              jl_function_t* jl_codelet = jl_get_function(jl_main_module, fn_name.c_str());
              if (jl_codelet == nullptr)
              {
                  std::stringstream message;
                  message << "Julia function '" << fn_name << "' not found in Main module.";
                  throw tools::julia_error(__FILE__, __LINE__, __func__, message.str());
              }
              m.jl_func_args[tid][0] = (void*)jl_codelet;
          }

          for (size_t i = 0; i < n_states; i++)
              m.jl_func_args[tid][1 + i] = (void*)m.jl_states_ptr[i];

          const size_t n_frames = m.self_module->get_n_frames();
          const size_t n_frames_per_wave = m.self_module->get_n_frames_per_wave();
          const size_t n_waves = m.self_module->get_n_waves();
          const size_t n_frames_per_wave_rest = m.self_module->get_n_frames_per_wave_rest();
          const size_t n_full_waves = (n_frames_per_wave_rest == 0) ? n_waves : (n_waves - 1);

          for (size_t s = 0; s < n_scks; s++)
          {
              const runtime::Socket* sck = task.sockets[s].get();
              jl_value_t* datatype = Module_Julia::get_jl_datatype(sck);
              auto& cache = m.jl_sockets_cache[tid][s];
              // Resolve and cache the 1D Julia `Vector{T}` (`Array{T, 1}`) type once at `eval()` time
              // instead of calling `jl_apply_array_type()` on every socket buffer creation.
              cache.array_type = jl_apply_array_type(datatype, 1);

              // If socket buffers are already bound when `eval()` runs (e.g., after socket binding in `Sequence`),
              // pre-allocate and GC-root the `Array` views for all full waves (`w = 0 .. n_full_waves - 1`) up front
              // so the very first codelet execution already hits the hot cache.
              int8_t* base_ptr = (int8_t*)sck->get_dataptr();
              const size_t n_elmts_per_wave = sck->get_n_elmts() * n_frames_per_wave / n_frames;
              const size_t bytes_per_wave = n_elmts_per_wave * sck->get_datatype_size();
              if (base_ptr != nullptr && n_elmts_per_wave > 0 && n_full_waves > 0)
              {
                  for (size_t w = 0; w < n_full_waves; w++)
                  {
                      void* wave_ptr = (void*)(base_ptr + w * bytes_per_wave);
                      m.get_or_create_socket_array(tid, s, wave_ptr, n_elmts_per_wave);
                  }
                  m.jl_func_args[tid][1 + n_states + s] = (void*)cache.entries[0].jl_array;
                  cache.last_dataptr = cache.entries[0].dataptr;
                  cache.last_databytes = sck->get_databytes();
                  cache.last_n_frames_per_wave = n_frames_per_wave;
                  cache.last_hit_idx = 0;
              }
          }

          for (size_t i = 0; i < n_csts; i++)
              m.jl_func_args[tid][m.jl_func_args[tid].size() - (n_csts + 2) + i] = (void*)m.jl_constants_ptr[tid][i];

          // Allocate dedicated heap-backed UInt32 boxes (passing a value >= 1024 bypasses Julia's
          // global box_uint32_cache[0..1023] singletons, which must never be mutated in-place).
          jl_value_t* jl_frame_id = jl_box_uint32(std::numeric_limits<uint32_t>::max());
          *(uint32_t*)jl_frame_id = 0;
          size_t jl_frame_id_id = tools::Julia::preserve(jl_frame_id);
          m.jl_constants_ptr[tid].push_back(jl_frame_id);
          m.jl_constants_id[tid].push_back(jl_frame_id_id);
          m.jl_func_args[tid][m.jl_func_args[tid].size() - 2] = (void*)jl_frame_id;

          jl_value_t* jl_n_frames_per_wave = jl_box_uint32(std::numeric_limits<uint32_t>::max());
          *(uint32_t*)jl_n_frames_per_wave = (uint32_t)m.self_module->get_n_frames_per_wave();
          size_t jl_n_frames_per_wave_id = tools::Julia::preserve(jl_n_frames_per_wave);
          m.jl_constants_ptr[tid].push_back(jl_n_frames_per_wave);
          m.jl_constants_id[tid].push_back(jl_n_frames_per_wave_id);
          m.jl_func_args[tid][m.jl_func_args[tid].size() - 1] = (void*)jl_n_frames_per_wave;
      });
}

#endif /* SPU_JULIA */
