#include <algorithm>
#include <cassert>
#include <sstream>

#include "Runtime/Task/Task.hpp"

namespace spu
{
namespace runtime
{

bool
Task::is_stats() const
{
    return this->stats;
}

bool
Task::is_fast() const
{
#ifndef SPU_FAST
    return this->fast;
#else
    return true;
#endif
}

bool
Task::is_debug() const
{
    return this->debug;
}

bool
Task::is_debug_hex() const
{
    return this->debug_hex;
}

bool
Task::is_last_input_socket(const Socket& s_in) const
{
    return last_input_socket == &s_in;
}

bool
Task::is_outbuffers_allocated() const
{
    return this->outbuffers_allocated;
}

module::Module&
Task::get_module() const
{
    return *this->module;
}

std::string
Task::get_name() const
{
    return this->name;
}

uint32_t
Task::get_n_calls() const
{
    return this->n_calls;
}

Socket&
Task::operator[](const size_t id)
{
    assert((size_t)id < this->sockets.size());
    assert(this->sockets[id] != nullptr);

    return *this->sockets[id];
}

void
Task::update_timer(const size_t id, const std::chrono::nanoseconds& duration)
{
    if (this->is_stats())
    {
        this->timers_n_calls[id]++;
        this->timers_total[id] += duration;
        if (this->n_calls)
        {
            this->timers_max[id] = std::max(this->timers_max[id], duration);
            this->timers_min[id] = std::min(this->timers_min[id], duration);
        }
        else
        {
            this->timers_max[id] = duration;
            this->timers_min[id] = duration;
        }
    }
}

const std::vector<int>&
Task::get_status() const
{
    return this->status;
}

#if defined(_MSC_VER)
#define SPU_TASK_LIKELY(x) (x)
#define SPU_TASK_UNLIKELY(x) (x)
#else
#define SPU_TASK_LIKELY(x) __builtin_expect(!!(x), 1)
#define SPU_TASK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// Inlined hot path for Task::exec().
// Because Sequence::exec_subsequence* always calls task->exec() with default arguments (frame_id = -1,
// managed_memory = true), inlining this fast path in Task.hxx allows the compiler to:
// 1. Constant-fold (frame_id == -1 || frame_id == 0) to true, leaving only a single byte test on `this->is_fast_exec`;
// 2. Eliminate the function call prologue/epilogue (stp/ldp of 4 registers + bl/ret);
// 3. Evaluate `*this->status_dataptr_ptr` AFTER `fast_codelet_fn` returns so no extra callee-saved register is
//    spilled across the codelet call;
// 4. Dead-code eliminate the unused `return this->status` address computation in `exec_subsequence_without_statuses`.
inline const std::vector<int>&
Task::exec(const int frame_id, const bool managed_memory)
{
#ifndef SPU_FAST
    if (SPU_TASK_LIKELY(this->is_fast_exec && (frame_id == -1 || frame_id == 0)))
    {
        const int exec_status = this->fast_codelet_fn(this->fast_codelet_data, *this->module, *this, 0);
        *(int*)(*this->status_dataptr_ptr) = exec_status;
        this->n_calls++;
        return this->status;
    }
    return this->_exec_general(frame_id, managed_memory);
#else
    if (SPU_TASK_LIKELY(this->is_single_wave_exec && (frame_id == -1 || frame_id == 0)))
    {
        const int exec_status = this->fast_codelet_fn(this->fast_codelet_data, *this->module, *this, 0);
        *(int*)(*this->status_dataptr_ptr) = exec_status;
        this->n_calls++;
        return this->status;
    }
    this->_exec_slow(frame_id, managed_memory);
    this->n_calls++;
    return this->status;
#endif /* !SPU_FAST */
}

// Static trampoline instantiated for each trivially copyable codelet lambda type DF.
// Reinterprets the inline buffer fast_codelet_data as const DF* and invokes its operator() directly with frame_id
// passed by value in a register.
template<typename DF>
int
Task::call_fast_codelet(const void* data, module::Module& m, Task& t, size_t frame_id)
{
    return (*reinterpret_cast<const DF*>(data))(m, t, frame_id);
}

// C++11 tag-dispatch overloads (std::true_type / std::false_type) for registering a codelet functor DF.
// Using tag dispatch from a single create_codelet(F&&) template avoids out-of-line SFINAE overload matching
// ambiguities in GCC while remaining strictly C++11 compliant across GCC, Clang, MSVC, and icpc.
template<typename DF>
inline void
Task::register_fast_codelet(const DF& codelet, std::true_type)
{
    new (this->fast_codelet_data) DF(codelet);
    this->fast_codelet_fn = &Task::template call_fast_codelet<DF>;
}

template<typename DF>
inline void
Task::register_fast_codelet(const DF& /*codelet*/, std::false_type)
{
}

template<typename F,
         typename std::enable_if<
           !std::is_same<typename std::decay<F>::type, std::function<int(module::Module&, Task&, const size_t)>>::value,
           int>::type>
inline void
Task::create_codelet(F&& codelet)
{
    using DF = typename std::decay<F>::type;
    // Keep this->codelet populated for full backward compatibility, then override fast_codelet_fn with the direct
    // static trampoline call_fast_codelet<DF> when DF fits inline in fast_codelet_data and is trivially copyable
    // (so Task::clone()'s default copy constructor safely duplicates fast_codelet_data byte-for-byte).
    std::function<int(module::Module & m, Task & t, const size_t frame_id)> fn(codelet);
    this->create_codelet(fn);
    constexpr bool is_fast_callable = std::is_trivially_copyable<DF>::value &&
                                      (sizeof(DF) <= Task::fast_codelet_capacity) && (alignof(DF) <= alignof(double));
    this->register_fast_codelet<DF>(codelet, std::integral_constant<bool, is_fast_callable>());
}
}
}
