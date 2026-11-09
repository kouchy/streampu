#ifdef SPU_JULIA
#include <julia.h>
#endif

#include "Module/Stateful/Adaptor/Adaptor_m_to_n.hpp"
#include "Tools/Display/Statistics/Statistics.hpp"
#include "Tools/Math/utils.h"

using namespace spu;
using namespace spu::module;

Adaptor_m_to_n::~Adaptor_m_to_n()
{
    for (auto b : this->buffer_to_free)
        delete[] b;
    if (this->cloned) (*this->n_clones)--;
}

Adaptor_m_to_n*
Adaptor_m_to_n::clone() const
{
    auto m = new Adaptor_m_to_n(*this);
    m->deep_copy(*this);
    return m;
}

void
Adaptor_m_to_n::deep_copy(const Adaptor_m_to_n& m)
{
    Stateful::deep_copy(m);

    if (*this->buffers_allocated)
    {
        std::stringstream message;
        message << "Shared buffers have already been allocated, cloning is no more possible." << std::endl;
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    this->cloned = true;
    (*this->n_clones)++;
    this->channels_ptr = this->channels->data();
    this->tid_push = -1;
    this->tid_pull = -1;
    this->cur_push_id = -1;
    this->cur_pull_id = -1;
    this->push_birth_ts_ptr = nullptr;
    this->push_stage_start_ts_ptr = nullptr;
    this->pull_birth_ts_ptr = nullptr;
    this->pull_stage_start_ts_ptr = nullptr;
}

void
Adaptor_m_to_n::alloc_buffers()
{
    if (*this->buffers_allocated)
    {
        std::stringstream message;
        message << "Synchronization buffers have already been allocated.";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    if (this->cloned)
    {
        std::stringstream message;
        message << "'alloc_buffers()' cannot be called on a cloned module ('tid_push' = " << this->tid_push
                << ", 'tid_pull' = " << this->tid_pull << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    size_t ppcm = tools::find_smallest_common_multiple(*this->n_pushers, *this->n_pullers);

    if (ppcm > 1000)
    {
        std::stringstream message;
        message << "'ppcm' cannot exceed 1000 ('ppcm' = " << ppcm << ").";
        throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }

    const size_t stride = this->buffer_size * this->n_sockets;
    this->buffer->resize(ppcm * stride, nullptr);
    this->ts_buffer->resize(ppcm, std::vector<uint64_t>(this->buffer_size, 0));

    if (this->buffer_to_free.size())
    {
        std::stringstream message;
        message << "This should never happen.";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    for (size_t d = 0; d < ppcm; d++)
    {
        for (size_t b = 0; b < this->buffer_size; b++)
            for (size_t s = 0; s < this->n_sockets; s++)
            {
                int8_t* ptr = new int8_t[this->n_frames * this->n_bytes[s]];
                (*this->buffer)[(d * this->buffer_size + b) * this->n_sockets + s] = ptr;
                this->buffer_to_free.push_back(ptr);
            }
        auto& ch = this->channels_ptr[d];
        ch.reset();
        ch.push_slots_base = &(*this->buffer)[d * stride];
        ch.pull_slots_base = &(*this->buffer)[d * stride];
        ch.push_ts_base = (*this->ts_buffer)[d].data();
        ch.pull_ts_base = (*this->ts_buffer)[d].data();
        ch.push_n_channels = static_cast<uint32_t>(ppcm);
        ch.pull_n_channels = static_cast<uint32_t>(ppcm);
    }

    *this->buffers_allocated = true;
}

void
Adaptor_m_to_n::add_pusher()
{
    if (*this->n_pushers - 1 + *this->n_pullers - 1 >= *this->n_clones)
    {
        std::stringstream message;
        message << "Pusher cannot be added because a clone is missing ('n_pushers' = " << *this->n_pushers
                << ", 'n_pullers' = " << *this->n_pullers << ", 'n_clones' = " << *this->n_clones << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    this->tid_push = (*this->n_pushers);
    (*this->n_pushers)++;
    this->cur_push_id = this->tid_push;
}

void
Adaptor_m_to_n::add_puller()
{
    if (*this->n_pushers - 1 + *this->n_pullers - 1 >= *this->n_clones)
    {
        std::stringstream message;
        message << "Puller cannot be added because a clone is missing ('n_pushers' = " << *this->n_pushers
                << ", 'n_pullers' = " << *this->n_pullers << ", 'n_clones' = " << *this->n_clones << ").";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    this->tid_pull = (*this->n_pullers);
    (*this->n_pullers)++;
    this->cur_pull_id = this->tid_pull;
}

void
Adaptor_m_to_n::send_cancel_signal()
{
    this->waiting_canceled->store(true, std::memory_order_relaxed);
}

void
Adaptor_m_to_n::reset()
{
    if (!this->cloned)
    {
        this->waiting_canceled->store(false, std::memory_order_relaxed);
        const size_t n_ch = this->channels_ptr[0].push_n_channels;
        for (size_t d = 0; d < n_ch; d++)
        {
            this->channels_ptr[d].reset();
        }
    }
    this->cur_push_id = (size_t)this->tid_push;
    this->cur_pull_id = (size_t)this->tid_pull;
    this->reset_buffer();
}

void
Adaptor_m_to_n::set_no_copy_pull(const bool no_copy_pull)
{
    this->no_copy_pull = no_copy_pull;
}

void
Adaptor_m_to_n::set_no_copy_push(const bool no_copy_push)
{
    this->no_copy_push = no_copy_push;
}

void
Adaptor_m_to_n::reset_buffer()
{
    if (!this->cloned && *this->buffers_allocated)
    {
        for (size_t i = 0; i < this->buffer_to_free.size(); i++)
            (*this->buffer)[i] = this->buffer_to_free[i];
    }
}

void
Adaptor_m_to_n::set_n_frames(const size_t n_frames)
{
    const auto old_n_frames = this->get_n_frames();
    if (old_n_frames != n_frames)
    {
        Module::set_n_frames(n_frames);
        if (!this->cloned && *this->buffers_allocated)
        {
            const size_t n_ch = this->channels_ptr[0].push_n_channels;
            for (size_t d = 0; d < n_ch; d++)
                for (size_t b = 0; b < this->buffer_size; b++)
                    for (size_t s = 0; s < this->n_sockets; s++)
                    {
                        const size_t idx = (d * this->buffer_size + b) * this->n_sockets + s;
                        auto old_ptr = (*this->buffer)[idx];
                        int8_t* new_ptr = new int8_t[this->n_bytes[s] * n_frames];
                        (*this->buffer)[idx] = new_ptr;

                        bool found = false;
                        for (size_t bf = 0; bf < this->buffer_to_free.size(); bf++)
                            if (this->buffer_to_free[bf] == old_ptr)
                            {
                                delete[] this->buffer_to_free[bf];
                                this->buffer_to_free[bf] = new_ptr;
                                found = true;
                                break;
                            }

                        if (found == false)
                        {
                            std::stringstream message;
                            message << "This should never happen.";
                            throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
                        }
                    }
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------

void
Adaptor_m_to_n::push(const runtime::Task& t, const size_t base_sck, const size_t /*frame_id*/)
{
    this->wait_push();

    const size_t n_frm = this->Module::get_n_frames();
    for (size_t s = 0; s < this->n_sockets; s++)
    {
        const int8_t* in = static_cast<const int8_t*>(t.sockets[base_sck + s]->_get_dataptr());
        int8_t* out = static_cast<int8_t*>(this->get_empty_buffer(s));
        std::copy(in, in + n_frm * this->n_bytes[s], out);
    }

    this->wake_up_puller();
}

void
Adaptor_m_to_n::pull(runtime::Task& t, const size_t base_sck, const size_t /*frame_id*/)
{
    this->wait_pull();

    const size_t n_frm = this->Module::get_n_frames();
    for (size_t s = 0; s < this->n_sockets; s++)
    {
        const int8_t* in = static_cast<const int8_t*>(this->get_filled_buffer(s));
        int8_t* out = static_cast<int8_t*>(t.sockets[base_sck + s]->_get_dataptr());
        std::copy(in, in + n_frm * this->n_bytes[s], out);
    }

    this->wake_up_pusher();
}

void
Adaptor_m_to_n::wait_push()
{
    if (this->tid_push < 0)
    {
        std::stringstream message;
        message << "This instance is not a pusher.";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    // When the inter-stage buffer is full, a thread registered with the Julia runtime ('jl_get_pgcstack() != nullptr')
    // must enter GC_SAFE before blocking (active spin-wait or passive condition_variable wait) so that another pipeline
    // stage running Julia tasks can execute a Stop-The-World GC cycle and drain the buffer without deadlocking.
    // When Real-Time pipeline statistics are enabled (`push_stage_start_ts_ptr != nullptr`), the time spent blocked
    // waiting for a free downstream slot is added to `*push_stage_start_ts_ptr` so it is subtracted from this stage's
    // pure processing latency (`t_end - cur_stage_start_ts`).
    if (this->active_waiting)
    {
        if (this->is_full(this->cur_push_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
        {
            const uint64_t t_wait_start = (this->push_stage_start_ts_ptr != nullptr) ? tools::read_tsc() : 0;
#ifdef SPU_JULIA
            // Phase 1 (Fast-Spin): spin for up to `SPU_FAST_SPIN_ITERS` iterations (~150-200 ns) using `cpu_relax()`
            // before paying the atomic TLS cost of `jl_gc_safe_enter` / `jl_gc_safe_leave`. In steady-state streaming,
            // short inter-stage stalls resolve within a few tens of nanoseconds, eliminating >99% of GC-state atomic
            // transitions while still entering `GC_SAFE` if the downstream stage is genuinely blocked or in a GC cycle.
            static constexpr size_t SPU_FAST_SPIN_ITERS = 64;
            size_t spin_iters = 0;
            while (spin_iters < SPU_FAST_SPIN_ITERS && this->is_full(this->cur_push_id) &&
                   !this->waiting_canceled->load(std::memory_order_relaxed))
            {
                Adaptor_m_to_n::cpu_relax();
                spin_iters++;
            }
            if (this->is_full(this->cur_push_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
            {
                int8_t gc_state = 0;
                const bool can_safe = (jl_get_pgcstack() != nullptr);
                if (can_safe) gc_state = jl_gc_safe_enter(jl_current_task->ptls);
                while (this->is_full(this->cur_push_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
                    Adaptor_m_to_n::cpu_relax();
                if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, gc_state);
            }
#else
            while (this->is_full(this->cur_push_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
                Adaptor_m_to_n::cpu_relax();
#endif
            if (this->push_stage_start_ts_ptr != nullptr)
                *this->push_stage_start_ts_ptr += (tools::read_tsc() - t_wait_start);
        }
    }
    else // passive waiting
    {
        if (this->is_full(this->cur_push_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
        {
            const uint64_t t_wait_start = (this->push_stage_start_ts_ptr != nullptr) ? tools::read_tsc() : 0;
#ifdef SPU_JULIA
            int8_t gc_state = 0;
            const bool can_safe = (jl_get_pgcstack() != nullptr);
            if (can_safe) gc_state = jl_gc_safe_enter(jl_current_task->ptls);
#endif
            {
                std::unique_lock<std::mutex> lock((*this->mtx_push.get())[this->cur_push_id]);
                ((*this->cnd_push.get())[this->cur_push_id])
                  .wait(lock,
                        [this]() {
                            return !(this->is_full(this->cur_push_id) &&
                                     !this->waiting_canceled->load(std::memory_order_relaxed));
                        });
            }
#ifdef SPU_JULIA
            if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, gc_state);
#endif
            if (this->push_stage_start_ts_ptr != nullptr)
                *this->push_stage_start_ts_ptr += (tools::read_tsc() - t_wait_start);
        }
    }

    if (this->waiting_canceled->load(std::memory_order_relaxed))
        throw tools::waiting_canceled(__FILE__, __LINE__, __func__);
}

void
Adaptor_m_to_n::wait_pull()
{
    if (this->tid_pull < 0)
    {
        std::stringstream message;
        message << "This instance is not a puller.";
        throw tools::runtime_error(__FILE__, __LINE__, __func__, message.str());
    }

    // When the inter-stage buffer is empty, enter GC_SAFE if this thread is registered with the Julia runtime
    // ('jl_get_pgcstack() != nullptr') so upstream Julia stages can run a Stop-The-World GC cycle without deadlocking.
    if (this->active_waiting)
    {
        if (this->is_empty(this->cur_pull_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
        {
#ifdef SPU_JULIA
            // Phase 1 (Fast-Spin): spin briefly with `cpu_relax()` before entering `jl_gc_safe_enter` so short
            // producer-consumer handoff delays do not incur two atomic TLS GC-state transitions per frame.
            static constexpr size_t SPU_FAST_SPIN_ITERS = 64;
            size_t spin_iters = 0;
            while (spin_iters < SPU_FAST_SPIN_ITERS && this->is_empty(this->cur_pull_id) &&
                   !this->waiting_canceled->load(std::memory_order_relaxed))
            {
                Adaptor_m_to_n::cpu_relax();
                spin_iters++;
            }
            if (this->is_empty(this->cur_pull_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
            {
                int8_t gc_state = 0;
                const bool can_safe = (jl_get_pgcstack() != nullptr);
                if (can_safe) gc_state = jl_gc_safe_enter(jl_current_task->ptls);
                while (this->is_empty(this->cur_pull_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
                    Adaptor_m_to_n::cpu_relax();
                if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, gc_state);
            }
#else
            while (this->is_empty(this->cur_pull_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
                Adaptor_m_to_n::cpu_relax();
#endif
        }
    }
    else // passive waiting
    {
        if (this->is_empty(this->cur_pull_id) && !this->waiting_canceled->load(std::memory_order_relaxed))
        {
#ifdef SPU_JULIA
            int8_t gc_state = 0;
            const bool can_safe = (jl_get_pgcstack() != nullptr);
            if (can_safe) gc_state = jl_gc_safe_enter(jl_current_task->ptls);
#endif
            {
                std::unique_lock<std::mutex> lock((*this->mtx_pull.get())[this->cur_pull_id]);
                ((*this->cnd_pull.get())[this->cur_pull_id])
                  .wait(lock,
                        [this]() {
                            return !(this->is_empty(this->cur_pull_id) &&
                                     !this->waiting_canceled->load(std::memory_order_relaxed));
                        });
            }
#ifdef SPU_JULIA
            if (can_safe) jl_gc_safe_leave(jl_current_task->ptls, gc_state);
#endif
        }
    }

    // Check the cheap `relaxed` cancellation flag first so `is_empty()` is not re-evaluated on every normal frame.
    if (this->waiting_canceled->load(std::memory_order_relaxed) && this->is_empty(this->cur_pull_id))
        throw tools::waiting_canceled(__FILE__, __LINE__, __func__);

    // When Real-Time pipeline statistics are enabled, record the exact timestamp at which `wait_pull()` unblocks
    // (so idle waiting time is excluded from this stage's own processing latency) and retrieve the frame's birth
    // timestamp from the shared ring buffer slot before `wake_up_pusher()` releases the slot.
    if (this->pull_stage_start_ts_ptr != nullptr)
    {
        const auto& ch = this->channels_ptr[this->cur_pull_id];
        *this->pull_stage_start_ts_ptr = tools::read_tsc();
        *this->pull_birth_ts_ptr = ch.pull_ts_base[ch.read_slot];
    }
}

[[noreturn]] void
Adaptor_m_to_n::throw_unallocated_buffers_error(const char* func_name) const
{
    std::stringstream message;
    message << "You need to call 'alloc_buffers()' before to change '" << func_name << "'.";
    throw tools::runtime_error(__FILE__, __LINE__, func_name, message.str());
}

void
Adaptor_m_to_n::wake_up()
{
    if (!this->active_waiting) // passive waiting
    {
        const size_t n_ch = this->channels_ptr[0].push_n_channels;
        for (size_t i = 0; i < n_ch; i++)
        {
            std::unique_lock<std::mutex> lock((*this->mtx_push.get())[i]);
            (*this->cnd_push.get())[i].notify_all();
        }
        for (size_t i = 0; i < n_ch; i++)
        {
            std::unique_lock<std::mutex> lock((*this->mtx_pull.get())[i]);
            (*this->cnd_pull.get())[i].notify_all();
        }
    }
}

void
Adaptor_m_to_n::cancel_waiting()
{
    this->send_cancel_signal();
    this->wake_up();
}
