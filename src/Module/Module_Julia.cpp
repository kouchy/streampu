#ifdef SPU_JULIA

#include <atomic>
#include <cstdint>
#include <functional>
#include <julia.h>
#include <mutex>
#include <sstream>
#include <typeindex>
#include <vector>

#include "Module/Module_Julia.hpp"
#include "Runtime/Socket/Socket.hpp"
#include "Tools/Exception/exception.hpp"
#include "Tools/Julia/Julia.hpp"
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

// A `Ref_keeper` is `thread_local` (one per thread, not thread-safe) and a slot index is only valid in the keeper
// that allocated it. A socket array is rooted by the thread that executes the task, which is not necessarily the one
// that evaluated the module (e.g. a module evaluated on the main thread and executed by a `Pipeline` stage thread),
// so each cache entry records its keeper. Update the slot in place only when it belongs to the current thread;
// otherwise root the new array in the current thread's keeper (the old slot is left to its keeper, it keeps a
// non-owning array view alive, which is harmless).
static inline void
reroot_socket_array(size_t& jl_id, void*& keeper, jl_value_t* jl_arr)
{
    tools::Ref_keeper* current = tools::Ref_keeper::get();
    if (keeper == (void*)current)
        current->update_value(jl_id, jl_arr);
    else
    {
        jl_id = current->mark(jl_arr);
        keeper = (void*)current;
    }
}

jl_value_t*
Module_Julia::get_or_create_socket_array(const size_t tid, const size_t s, void* p, const size_t n_elmts)
{
    auto& cache = this->get_task_cache(tid).sockets[s];
    const size_t h = Socket_jl_cache::hash_dataptr(p);
    if (cache.ht[h].dataptr == p && cache.n_elmts_per_wave == n_elmts)
    {
        cache.last_dataptr = p;
        return cache.ht[h].jl_array;
    }

    const size_t n_entries = cache.entries.size();
    if (n_entries > 0)
    {
        // 1. O(1) cyclic fast path: in multi-wave execution (`w = 0, 1, ..., W-1`) or Pipeline adaptor ring buffers,
        //    consecutive calls step sequentially through `cache.entries`. Checking `last_hit_idx + 1` with a
        //    branchless wrap-around (instead of `% n_entries` integer division) resolves cyclic lookups in O(1).
        const size_t next_idx = (cache.last_hit_idx + 1 < n_entries) ? (cache.last_hit_idx + 1) : 0;
        const auto& next_entry = cache.entries[next_idx];
        if (next_entry.dataptr == p && next_entry.n_elmts == n_elmts)
        {
            cache.last_dataptr = p;
            cache.last_hit_idx = next_idx;
            cache.ht[h] = { p, next_entry.jl_array };
            return next_entry.jl_array;
        }

        // 2. Linear scan fallback across all cached entries for this socket (e.g. if two buffer addresses collided
        //    in the direct-mapped `ht` table or arrived out of cyclic order).
        for (size_t i = 0; i < n_entries; i++)
        {
            const auto& entry = cache.entries[i];
            if (entry.dataptr == p && entry.n_elmts == n_elmts)
            {
                cache.last_dataptr = p;
                cache.last_hit_idx = i;
                cache.ht[h] = { p, entry.jl_array };
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
            const size_t old_h = Socket_jl_cache::hash_dataptr(entry.dataptr);
            if (cache.ht[old_h].dataptr == entry.dataptr) cache.ht[old_h] = { nullptr, nullptr };

            auto gc_state = jl_gc_enable(0);
            jl_value_t* jl_arr = (jl_value_t*)jl_ptr_to_array_1d(cache.array_type, p, n_elmts, 0);
            reroot_socket_array(entry.jl_id, entry.keeper, jl_arr);
            jl_gc_enable(gc_state);

            entry.dataptr = p;
            entry.n_elmts = n_elmts;
            entry.jl_array = jl_arr;
            cache.last_dataptr = p;
            cache.last_hit_idx = i;
            cache.ht[h] = { p, jl_arr };
            return jl_arr;
        }
    }

    // 4. Cache miss: allocate a new 1D Julia `Array` wrapper (`own_buffer = 0`, so C++ retains ownership of `p`).
    //    Use a base capacity of 256 entries (scaling with `get_n_waves() * 64`) so even large Pipeline adaptor ring
    //    buffers (e.g., `buffer_size = 64`, which circulates `64 + 2 = 66` distinct buffers) never thrash the FIFO
    //    cache.
    const size_t max_entries = std::max<size_t>(256, this->self_module->get_n_waves() * 64);
    auto gc_state = jl_gc_enable(0);
    jl_value_t* jl_arr = (jl_value_t*)jl_ptr_to_array_1d(cache.array_type, p, n_elmts, 0);
    if (n_entries < max_entries)
    {
        // Released by `~Module_Julia()` from `cache.entries` (not `jl_constants_id`, which belongs to the keeper
        // of the evaluating thread)
        size_t jl_id = tools::Julia::preserve(jl_arr);
        cache.entries.push_back({ p, n_elmts, jl_arr, jl_id, (void*)tools::Ref_keeper::get() });
        cache.last_hit_idx = n_entries;
    }
    else
    {
        const size_t evict_idx = (cache.next_evict_idx < max_entries) ? cache.next_evict_idx : 0;
        cache.next_evict_idx = (evict_idx + 1 < max_entries) ? (evict_idx + 1) : 0;
        auto& victim = cache.entries[evict_idx];
        const size_t old_h = Socket_jl_cache::hash_dataptr(victim.dataptr);
        if (cache.ht[old_h].dataptr == victim.dataptr) cache.ht[old_h] = { nullptr, nullptr };
        reroot_socket_array(victim.jl_id, victim.keeper, jl_arr);
        victim.dataptr = p;
        victim.n_elmts = n_elmts;
        victim.jl_array = jl_arr;
        cache.last_hit_idx = evict_idx;
    }
    jl_gc_enable(gc_state);

    cache.last_dataptr = p;
    cache.ht[h] = { p, jl_arr };
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
  , extra_tasks_cache(std::vector<Task_jl_cache>())
  , jl_states_ptr(std::vector<jl_value_t*>())
  , jl_states_id(std::vector<size_t>())
  , jl_constants_ptr(std::vector<std::vector<jl_value_t*>>())
  , jl_constants_id(std::vector<std::vector<size_t>>())
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
    for (size_t i = 0; i < SPU_MAX_INLINE_TASKS; i++)
    {
        this->inline_tasks_cache[i].owner = this;
        this->inline_tasks_cache[i].tid = i;
        this->inline_tasks_cache[i].jl_safe = jl_safe;
    }
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

    // Socket arrays: release only the slots owned by the current thread's keeper. The others belong to the keeper
    // of another thread (e.g. a `Pipeline` stage thread): releasing them here would corrupt the freelist of this
    // thread's keeper and unroot unrelated Julia objects. They are left to their keeper, as the states and the
    // constants of a module evaluated by another thread.
    tools::Ref_keeper* keeper = tools::Ref_keeper::get();
    auto release_socket_arrays = [keeper](Task_jl_cache& tcache)
    {
        for (auto& cache : tcache.sockets)
            for (auto& entry : cache.entries)
                if (entry.keeper == (void*)keeper) keeper->unmark(entry.jl_id);
    };
    for (size_t i = 0; i < SPU_MAX_INLINE_TASKS; i++)
        release_socket_arrays(this->inline_tasks_cache[i]);
    for (auto& tcache : this->extra_tasks_cache)
        release_socket_arrays(tcache);

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

    for (size_t i = 0; i < SPU_MAX_INLINE_TASKS; i++)
    {
        this->inline_tasks_cache[i].owner = this;
        this->inline_tasks_cache[i].tid = i;
        this->inline_tasks_cache[i].jl_safe = this->jl_safe;
    }
    for (size_t i = 0; i < this->extra_tasks_cache.size(); i++)
    {
        this->extra_tasks_cache[i].owner = this;
        this->extra_tasks_cache[i].tid = SPU_MAX_INLINE_TASKS + i;
        this->extra_tasks_cache[i].jl_safe = this->jl_safe;
    }

    for (size_t tid = 0; tid < this->self_module->tasks.size(); tid++)
    {
        auto& tcache = this->get_task_cache(tid);
        tcache.unsafe_cfunc = nullptr;
        tcache.safe_cfunc = nullptr;
        tcache.func_args_ptr = nullptr;
        tcache.last_n_frames = 0;
        tcache.last_n_frames_per_wave = 0;
        tcache.sockets.clear();
        tcache.func_args.clear();

        runtime::Task& t = *this->self_module->tasks[tid];
        const size_t n_scks = t.sockets.size() > 0 ? t.sockets.size() - 1 : 0;
        for (size_t s = 0; s < SPU_MAX_FAST_SOCKETS; s++)
        {
            tcache.fast_scks[s].sck = (s < n_scks) ? t.sockets[s].get() : nullptr;
            tcache.fast_scks[s].last_dataptr = nullptr;
        }
    }

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
    for (size_t i = 0; i < SPU_MAX_INLINE_TASKS; i++)
        this->inline_tasks_cache[i].jl_safe = jl_safe;
    for (auto& tc : this->extra_tasks_cache)
        tc.jl_safe = jl_safe;
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
    int8_t prev_gc_state = 0;
    const bool can_safe = (jl_get_pgcstack() != nullptr);
    if (can_safe) prev_gc_state = jl_gc_safe_enter(jl_current_task->ptls);
    std::unique_lock<std::mutex> lock(mutex);
    if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, prev_gc_state);

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

    // 3. Create task constants and bind func_args (including module states at indices 1..K)
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

    const size_t tid = this->self_module->tasks.size();
    this->jl_create_constants->push_back(std::vector<std::function<void(Module_Julia & m)>>());
    this->jl_constants_ptr.push_back(std::vector<jl_value_t*>());
    this->jl_constants_id.push_back(std::vector<size_t>());
    if (tid >= SPU_MAX_INLINE_TASKS)
    {
        Task_jl_cache tc;
        tc.owner = this;
        tc.tid = tid;
        tc.jl_safe = this->jl_safe;
        this->extra_tasks_cache.push_back(std::move(tc));
    }
    else
    {
        this->inline_tasks_cache[tid].owner = this;
        this->inline_tasks_cache[tid].tid = tid;
        this->inline_tasks_cache[tid].jl_safe = this->jl_safe;
    }

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

#ifdef _MSC_VER
#define SPU_NOINLINE __declspec(noinline)
#define SPU_UNLIKELY(x) (x)
#else
#define SPU_NOINLINE __attribute__((noinline))
#define SPU_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

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
static inline typename std::enable_if<(I > Module_Julia::SPU_MAX_DIRECT_TRAMPOLINE_ARGS), int32_t>::type
invoke_trampoline(void* cfunc, void* const* a, const uint32_t fid, const uint32_t nfw, const size_t /*n_obj_args*/)
{
    return ((int32_t (*)(void* const*, uint32_t, uint32_t))cfunc)(a, fid, nfw);
}

// Recursive compile-time dispatch for `0 <= n_obj_args <= SPU_MAX_DIRECT_TRAMPOLINE_ARGS`.
// Compiles down to a dense jump table where each branch invokes `cfunc` with the exact arity `I + 1`
// (`a[0]` for `f`, plus `a[1..I]` for the `I` object arguments).
template<size_t I = 0>
static inline typename std::enable_if<(I <= Module_Julia::SPU_MAX_DIRECT_TRAMPOLINE_ARGS), int32_t>::type
invoke_trampoline(void* cfunc, void* const* a, const uint32_t fid, const uint32_t nfw, const size_t n_obj_args)
{
    if (n_obj_args == I) return call_trampoline_n(cfunc, a, fid, nfw, Make_index_seq<I + 1>{});
    return invoke_trampoline<I + 1>(cfunc, a, fid, nfw, n_obj_args);
}

static SPU_NOINLINE void
rethrow_julia_exception()
{
    jl_function_t* take_exc_fn = jl_get_function(jl_main_module, "_spu_take_exception");
    jl_value_t* msg_val = take_exc_fn ? jl_call0(take_exc_fn) : nullptr;
    const char* msg = (msg_val && jl_is_string(msg_val)) ? jl_string_ptr(msg_val) : "Julia exception occurred.";
    throw tools::julia_error(__FILE__, __LINE__, __func__, msg);
}

// Invokes the safe `@cfunction` trampoline (`_spu_trampoline_safe_N`), which wraps the inner typed trampoline call
// inside a Julia `try ... catch` block and returns a packed `Int64`:
// - Lower 32 bits: the zero-extended `Int32` status returned by the Julia codelet.
// - Upper 32 bits: `0` on success, `1` if a Julia exception was caught (in which case `_spu_take_exception()` retrieves
//   the formatted exception + backtrace string for the current task and rethrows it as a C++ `tools::julia_error`).
static SPU_NOINLINE int32_t
invoke_safe_trampoline(void* safe_cfunc, void* const* a, const uint32_t fid, const uint32_t nfw)
{
    const int64_t ret = ((int64_t (*)(void* const*, uint32_t, uint32_t))safe_cfunc)(a, fid, nfw);
    if (SPU_UNLIKELY((ret >> 32) != 0)) rethrow_julia_exception();
    return (int32_t)ret;
}

template<size_t N_SCKS>
static inline uintptr_t
diff_fast_sockets(const Module_Julia::Task_jl_cache* tcache)
{
    const void* p = tcache->fast_scks[N_SCKS - 1].sck->get_dataptr<const uint8_t>();
    return diff_fast_sockets<N_SCKS - 1>(tcache) |
           ((uintptr_t)tcache->fast_scks[N_SCKS - 1].last_dataptr ^ (uintptr_t)p);
}

template<>
inline uintptr_t
diff_fast_sockets<0>(const Module_Julia::Task_jl_cache* /*tcache*/)
{
    return 0;
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

    const size_t n_types = func_args.size();
    void* out_ptrs[2] = { nullptr, nullptr };
    // Root the boxed arguments: each `jl_box_*` allocates, and a GC (triggered by any thread, e.g. another `Pipeline`
    // stage) between two allocations would collect the boxes created before (`jl_call3` only roots its arguments
    // once it is entered).
    jl_value_t* jl_args_ptr = nullptr;
    jl_value_t* jl_n_types = nullptr;
    jl_value_t* jl_out_ptrs = nullptr;
    JL_GC_PUSH3(&jl_args_ptr, &jl_n_types, &jl_out_ptrs);
    jl_args_ptr = jl_box_voidpointer((void*)func_args.data());
    jl_n_types = jl_box_int64((int64_t)n_types);
    jl_out_ptrs = jl_box_voidpointer((void*)out_ptrs);
    (void)jl_call3(make_trampoline_fn, jl_args_ptr, jl_n_types, jl_out_ptrs);
    JL_GC_POP();

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

SPU_NOINLINE int
Module_Julia::slow_exec_codelet(Task_jl_cache* tcache, const size_t frame_id)
{
    Module_Julia& sjl = *tcache->owner;
    if (!sjl.is_eval())
    {
        std::stringstream message;
        message << "The task cannot be executed because the module has not been evaluated.";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    const size_t tid = tcache->tid;
    Module& m = *sjl.self_module;
    const size_t n_frames = m.Module::get_n_frames();
    const size_t n_frames_per_wave = m.get_n_frames_per_wave();
    const size_t n_scks = tcache->n_scks;
    const size_t first_sck_arg = tcache->first_sck_arg;
    void** func_args = tcache->func_args_ptr;

    if (tcache->last_n_frames != n_frames || tcache->last_n_frames_per_wave != n_frames_per_wave)
    {
        for (size_t s = 0; s < n_scks; s++)
        {
            auto& cache = tcache->sockets[s];
            const size_t new_n_elmts = cache.sck->get_n_elmts() * n_frames_per_wave / n_frames;
            if (cache.n_elmts_per_wave != new_n_elmts)
            {
                cache.n_elmts_per_wave = new_n_elmts;
                cache.last_dataptr = nullptr;
                for (size_t h = 0; h < Socket_jl_cache::HT_SIZE; h++)
                    cache.ht[h] = { nullptr, nullptr };
                if (s < SPU_MAX_FAST_SOCKETS) tcache->fast_scks[s].last_dataptr = nullptr;
            }
        }
    }

    for (size_t s = 0; s < n_scks; s++)
    {
        auto& cache = tcache->sockets[s];
        void* p = (void*)cache.sck->get_dataptr<const uint8_t>();
        if (cache.last_dataptr != p)
        {
            func_args[first_sck_arg + s] = (void*)sjl.get_or_create_socket_array(tid, s, p, cache.n_elmts_per_wave);
            if (s < SPU_MAX_FAST_SOCKETS) tcache->fast_scks[s].last_dataptr = p;
        }
    }

    if (tcache->unsafe_cfunc == nullptr)
    {
        auto& shared = *tcache->shared_trampoline;
        if (!shared.compiled.load(std::memory_order_acquire))
        {
            static std::mutex compile_mtx;
            int8_t prev_gc_state = 0;
            const bool can_safe = (jl_get_pgcstack() != nullptr);
            if (can_safe) prev_gc_state = jl_gc_safe_enter(jl_current_task->ptls);

            std::unique_lock<std::mutex> lock(compile_mtx);

            if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, prev_gc_state);

            if (!shared.compiled.load(std::memory_order_relaxed))
            {
                void* unsafe_fn = nullptr;
                void* safe_fn = nullptr;
                get_or_compile_trampoline(tcache->func_args, unsafe_fn, safe_fn);
                shared.unsafe_cfunc.store(unsafe_fn, std::memory_order_relaxed);
                shared.safe_cfunc.store(safe_fn, std::memory_order_relaxed);
                shared.compiled.store(true, std::memory_order_release);

                tcache->unsafe_cfunc = unsafe_fn;
                tcache->safe_cfunc = safe_fn;
                if (tcache->fixed_arity_valid)
                {
                    tcache->last_n_frames = n_frames;
                    tcache->last_n_frames_per_wave = n_frames_per_wave;
                }

                return sjl.jl_safe
                         ? invoke_safe_trampoline(safe_fn, func_args, (uint32_t)frame_id, (uint32_t)n_frames_per_wave)
                         : invoke_trampoline(
                             unsafe_fn, func_args, (uint32_t)frame_id, (uint32_t)n_frames_per_wave, tcache->n_obj_args);
            }
        }
        tcache->unsafe_cfunc = shared.unsafe_cfunc.load(std::memory_order_relaxed);
        tcache->safe_cfunc = shared.safe_cfunc.load(std::memory_order_relaxed);
    }

    if (tcache->fixed_arity_valid)
    {
        tcache->last_n_frames = n_frames;
        tcache->last_n_frames_per_wave = n_frames_per_wave;
    }

    if (sjl.jl_safe)
        return invoke_safe_trampoline(tcache->safe_cfunc, func_args, (uint32_t)frame_id, (uint32_t)n_frames_per_wave);
    return invoke_trampoline(
      tcache->unsafe_cfunc, func_args, (uint32_t)frame_id, (uint32_t)n_frames_per_wave, tcache->n_obj_args);
}

// Inline zero-copy / multi-wave socket rebinding helpers for `exec_codelet_fixed<N_SCKS, N_OBJ_ARGS>`:
// In a `Pipeline` with `no_copy=true` (where `Adaptor_m_to_n` swaps socket `dataptr` pointers on every frame) or in
// multi-wave execution, `diff_fast_sockets<N_SCKS>(tcache)` is non-zero on every frame/wave. Instead of bailing out to
// the out-of-line `slow_exec_codelet()` (which loses the compile-time arity `N_OBJ_ARGS`), `rebind_fast_sockets`
// resolves the `jl_array_t*` wrapper for each changed socket in O(1) via the direct-mapped L1 hash table `cache.ht`
// and proceeds directly to `call_trampoline_n`.
template<size_t I>
static inline void
rebind_fast_socket_at(Module_Julia::Task_jl_cache* tcache)
{
    void* p = const_cast<void*>(static_cast<const void*>(tcache->fast_scks[I].sck->get_dataptr<const uint8_t>()));
    if (tcache->fast_scks[I].last_dataptr != p)
    {
        tcache->fast_scks[I].last_dataptr = p;
        auto& cache = tcache->sockets[I];
        const size_t h = Module_Julia::Socket_jl_cache::hash_dataptr(p);
        jl_value_t* jl_arr = (cache.ht[h].dataptr == p)
                               ? cache.ht[h].jl_array
                               : tcache->owner->get_or_create_socket_array(tcache->tid, I, p, cache.n_elmts_per_wave);
        cache.last_dataptr = p;
        tcache->func_args_ptr[tcache->first_sck_arg + I] = (void*)jl_arr;
    }
}

template<size_t N_SCKS>
static inline void
rebind_fast_sockets(Module_Julia::Task_jl_cache* tcache)
{
    rebind_fast_sockets<N_SCKS - 1>(tcache);
    rebind_fast_socket_at<N_SCKS - 1>(tcache);
}

template<>
inline void
rebind_fast_sockets<0>(Module_Julia::Task_jl_cache* /*tcache*/)
{
}

// Fast-path codelet execution template specialized at `create_codelet()` time for the exact socket count `N_SCKS`
// (`0..8`) and exact object argument count `N_OBJ_ARGS` (`N_SCKS..16`).
// All three exit paths (`slow_exec_codelet`, `invoke_safe_trampoline`, and `call_trampoline_n`) are tail calls
// with zero local stack objects or destructors, allowing the compiler to omit the stack frame prologue/epilogue
// (`stp`/`ldp`) and tail-jump (`br`) directly into the Julia `@cfunction`.
template<size_t N_SCKS, size_t N_OBJ_ARGS>
static inline int
exec_codelet_fixed(Module_Julia::Task_jl_cache* tcache, module::Module& m, const size_t frame_id)
{
    const size_t n_frames = m.Module::get_n_frames();
    const size_t n_frames_per_wave = m.get_n_frames_per_wave();
    if (SPU_UNLIKELY(((tcache->last_n_frames ^ n_frames) | (tcache->last_n_frames_per_wave ^ n_frames_per_wave)) != 0))
    {
        return Module_Julia::slow_exec_codelet(tcache, frame_id);
    }
    if (diff_fast_sockets<N_SCKS>(tcache) != 0)
    {
        rebind_fast_sockets<N_SCKS>(tcache);
    }
    if (SPU_UNLIKELY(tcache->jl_safe))
    {
        return invoke_safe_trampoline(
          tcache->safe_cfunc, tcache->func_args_ptr, (uint32_t)frame_id, (uint32_t)n_frames_per_wave);
    }
    return call_trampoline_n(tcache->unsafe_cfunc,
                             tcache->func_args_ptr,
                             (uint32_t)frame_id,
                             (uint32_t)n_frames_per_wave,
                             Make_index_seq<N_OBJ_ARGS + 1>{});
}

// C++11 tag-dispatch recursion (std::true_type for I <= SPU_MAX_DIRECT_TRAMPOLINE_ARGS, std::false_type for the base
// case I > SPU_MAX_DIRECT_TRAMPOLINE_ARGS) that passes the specialized lambda directly to Module::create_codelet(task,
// lambda) without intermediate std::function type erasure, allowing Task::fast_codelet_fn to bind directly to
// Task::call_fast_codelet<Lambda>.
template<size_t N_SCKS, size_t I>
void
Module_Julia::bind_fast_codelet_arity(runtime::Task& task,
                                      const ptrdiff_t tcache_offset,
                                      const size_t /*expected_n_obj_args*/,
                                      std::false_type)
{
    this->self_module->create_codelet(
      task,
      [tcache_offset](module::Module& m, runtime::Task& /*t*/, const size_t frame_id) -> int
      {
          auto* tcache = reinterpret_cast<Module_Julia::Task_jl_cache*>(reinterpret_cast<char*>(&m) + tcache_offset);
          return Module_Julia::slow_exec_codelet(tcache, frame_id);
      });
}

template<size_t N_SCKS, size_t I>
void
Module_Julia::bind_fast_codelet_arity(runtime::Task& task,
                                      const ptrdiff_t tcache_offset,
                                      const size_t expected_n_obj_args,
                                      std::true_type)
{
    if (expected_n_obj_args == I)
    {
        this->self_module->create_codelet(
          task,
          [tcache_offset](module::Module& m, runtime::Task& /*t*/, const size_t frame_id) -> int
          {
              auto* tcache =
                reinterpret_cast<Module_Julia::Task_jl_cache*>(reinterpret_cast<char*>(&m) + tcache_offset);
              return exec_codelet_fixed<N_SCKS, I>(tcache, m, frame_id);
          });
        return;
    }
    constexpr bool next_in_range = (I + 1 <= Module_Julia::SPU_MAX_DIRECT_TRAMPOLINE_ARGS);
    this->bind_fast_codelet_arity<N_SCKS, I + 1>(
      task, tcache_offset, expected_n_obj_args, std::integral_constant<bool, next_in_range>());
}

void
Module_Julia::_create_codelet(runtime::Task& task, const std::string& julia_function_name)
{
    const size_t tid = Module_Julia::get_task_id(task);
    const size_t n_scks = task.sockets.size();
    const size_t n_states_at_create = this->jl_create_states->size();
    const size_t n_csts_at_create = (*this->jl_create_constants)[tid].size();
    const size_t expected_n_obj_args = n_states_at_create + n_scks + n_csts_at_create;

    auto shared_trampoline = std::make_shared<Shared_trampoline>();
    auto& init_tcache = this->get_task_cache(tid);
    init_tcache.owner = this;
    init_tcache.tid = tid;
    init_tcache.n_scks = n_scks;
    init_tcache.expected_n_obj_args = expected_n_obj_args;
    init_tcache.shared_trampoline = shared_trampoline;
    init_tcache.jl_safe = this->jl_safe;
    for (size_t s = 0; s < SPU_MAX_FAST_SOCKETS; s++)
    {
        init_tcache.fast_scks[s].sck = (s < n_scks) ? task.sockets[s].get() : nullptr;
        init_tcache.fast_scks[s].last_dataptr = nullptr;
    }

    if (tid < SPU_MAX_INLINE_TASKS && n_scks <= SPU_MAX_FAST_SOCKETS &&
        expected_n_obj_args <= SPU_MAX_DIRECT_TRAMPOLINE_ARGS)
    {
        const ptrdiff_t tcache_offset = reinterpret_cast<const char*>(&this->inline_tasks_cache[tid]) -
                                        reinterpret_cast<const char*>(this->self_module);
        switch (n_scks)
        {
            case 0:
                this->bind_fast_codelet_arity<0>(task, tcache_offset, expected_n_obj_args);
                break;
            case 1:
                this->bind_fast_codelet_arity<1>(task, tcache_offset, expected_n_obj_args);
                break;
            case 2:
                this->bind_fast_codelet_arity<2>(task, tcache_offset, expected_n_obj_args);
                break;
            case 3:
                this->bind_fast_codelet_arity<3>(task, tcache_offset, expected_n_obj_args);
                break;
            case 4:
                this->bind_fast_codelet_arity<4>(task, tcache_offset, expected_n_obj_args);
                break;
            case 5:
                this->bind_fast_codelet_arity<5>(task, tcache_offset, expected_n_obj_args);
                break;
            case 6:
                this->bind_fast_codelet_arity<6>(task, tcache_offset, expected_n_obj_args);
                break;
            case 7:
                this->bind_fast_codelet_arity<7>(task, tcache_offset, expected_n_obj_args);
                break;
            case 8:
                this->bind_fast_codelet_arity<8>(task, tcache_offset, expected_n_obj_args);
                break;
            default:
                break;
        }
    }
    else
    {
        const ptrdiff_t jl_offset =
          reinterpret_cast<const char*>(this) - reinterpret_cast<const char*>(this->self_module);
        this->self_module->create_codelet(
          task,
          [tid, jl_offset](module::Module& m, runtime::Task& /*t*/, const size_t frame_id) -> int
          {
              auto& sjl = *reinterpret_cast<Module_Julia*>(reinterpret_cast<char*>(&m) + jl_offset);
              return Module_Julia::slow_exec_codelet(&sjl.get_task_cache(tid), frame_id);
          });
    }

    this->jl_create_codelet->push_back(
      [tid, julia_function_name, expected_n_obj_args, shared_trampoline](Module_Julia& m)
      {
          runtime::Task& task = *(m.self_module->tasks[tid].get());
          auto& tcache = m.get_task_cache(tid);
          if (tcache.func_args.size() > 0)
          {
              std::stringstream message;
              message << "tcache.func_args has to be empty ('tcache.func_args.size()' = " << tcache.func_args.size()
                      << ", 'tid' = " << tid << ").";
              throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
          }

          const size_t n_states = m.jl_states_ptr.size();
          const size_t n_csts = m.jl_constants_ptr[tid].size();
          const size_t n_scks = task.sockets.size() > 0 ? task.sockets.size() - 1 : 0;
          const size_t n_obj_args = n_states + n_scks + n_csts;

          tcache.owner = &m;
          tcache.tid = tid;
          tcache.n_scks = n_scks;
          tcache.first_sck_arg = 1 + n_states;
          tcache.n_obj_args = n_obj_args;
          tcache.expected_n_obj_args = expected_n_obj_args;
          tcache.fixed_arity_valid =
            (tid < SPU_MAX_INLINE_TASKS && n_scks <= SPU_MAX_FAST_SOCKETS &&
             n_obj_args <= SPU_MAX_DIRECT_TRAMPOLINE_ARGS && n_obj_args == expected_n_obj_args);
          tcache.shared_trampoline = shared_trampoline;
          tcache.jl_safe = m.jl_safe;
          tcache.last_n_frames = 0;
          tcache.last_n_frames_per_wave = 0;

          tcache.func_args.resize(1 + n_obj_args);
          tcache.func_args_ptr = tcache.func_args.data();
          tcache.sockets.resize(n_scks);

          for (size_t s = 0; s < SPU_MAX_FAST_SOCKETS; s++)
          {
              tcache.fast_scks[s].sck = (s < n_scks) ? task.sockets[s].get() : nullptr;
              tcache.fast_scks[s].last_dataptr = nullptr;
          }

          if (tcache.func_args[0] == nullptr)
          {
              const std::string& fn_name = julia_function_name.empty() ? task.get_name() : julia_function_name;
              jl_function_t* jl_codelet = jl_get_function(jl_main_module, fn_name.c_str());
              if (jl_codelet == nullptr)
              {
                  std::stringstream message;
                  message << "Julia function '" << fn_name << "' not found in Main module.";
                  throw tools::julia_error(__FILE__, __LINE__, __func__, message.str());
              }
              tcache.func_args[0] = (void*)jl_codelet;
          }

          for (size_t i = 0; i < n_states; i++)
              tcache.func_args[1 + i] = (void*)m.jl_states_ptr[i];

          const size_t n_frames = m.self_module->get_n_frames();
          const size_t n_frames_per_wave = m.self_module->get_n_frames_per_wave();
          const size_t n_waves = m.self_module->get_n_waves();
          const size_t n_frames_per_wave_rest = m.self_module->get_n_frames_per_wave_rest();
          const size_t n_full_waves = (n_frames_per_wave_rest == 0) ? n_waves : (n_waves - 1);

          for (size_t s = 0; s < n_scks; s++)
          {
              const runtime::Socket* sck = task.sockets[s].get();
              jl_value_t* datatype = Module_Julia::get_jl_datatype(sck);
              auto& cache = tcache.sockets[s];
              cache.sck = sck;
              // Resolve and cache the 1D Julia `Vector{T}` (`Array{T, 1}`) type once at `eval()` time
              // instead of calling `jl_apply_array_type()` on every socket buffer creation.
              cache.array_type = jl_apply_array_type(datatype, 1);

              // If socket buffers are already bound when `eval()` runs (e.g., after socket binding in `Sequence`),
              // pre-allocate and GC-root the `Array` views for all full waves (`w = 0 .. n_full_waves - 1`) up front
              // so the very first codelet execution already hits the hot cache.
              int8_t* base_ptr = (int8_t*)sck->get_dataptr<const uint8_t>();
              const size_t n_elmts_per_wave = sck->get_n_elmts() * n_frames_per_wave / n_frames;
              const size_t bytes_per_wave = n_elmts_per_wave * sck->get_datatype_size();
              cache.n_elmts_per_wave = n_elmts_per_wave;
              if (base_ptr != nullptr && n_elmts_per_wave > 0 && n_full_waves > 0)
              {
                  for (size_t w = 0; w < n_full_waves; w++)
                  {
                      void* wave_ptr = (void*)(base_ptr + w * bytes_per_wave);
                      m.get_or_create_socket_array(tid, s, wave_ptr, n_elmts_per_wave);
                  }
                  tcache.func_args[1 + n_states + s] = (void*)cache.entries[0].jl_array;
                  cache.last_dataptr = cache.entries[0].dataptr;
                  cache.last_hit_idx = 0;
                  if (s < SPU_MAX_FAST_SOCKETS) tcache.fast_scks[s].last_dataptr = cache.entries[0].dataptr;
              }
          }

          for (size_t i = 0; i < n_csts; i++)
              tcache.func_args[1 + n_states + n_scks + i] = (void*)m.jl_constants_ptr[tid][i];
      });
}

#endif /* SPU_JULIA */
