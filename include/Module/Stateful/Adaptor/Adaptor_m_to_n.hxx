#include <cmath>
#include <sstream>
#include <string>

#include "Module/Stateful/Adaptor/Adaptor_m_to_n.hpp"
#include "Tools/Exception/exception.hpp"
#include "Tools/compute_bytes.h"

namespace spu
{
namespace module
{

Adaptor_m_to_n::Adaptor_m_to_n(const std::vector<size_t>& n_elmts,
                               const std::vector<std::type_index>& datatype,
                               const size_t buffer_size,
                               const bool active_waiting)
  : Stateful()
  , n_elmts(n_elmts)
  , n_bytes(tools::compute_bytes(n_elmts, datatype))
  , datatype(datatype)
  , buffer_size(buffer_size)
  , n_sockets(n_elmts.size())
  , buffer(new std::vector<int8_t*>(n_sockets * buffer_size, nullptr))
  , channels(new std::vector<SPSC_channel_sync>(1000))
  , channels_ptr(this->channels->data())
  , waiting_canceled(new std::atomic<bool>(false))
  , no_copy_pull(false)
  , no_copy_push(false)
  , active_waiting(active_waiting)
  , cnd_push(new std::vector<std::condition_variable>(1000))
  , mtx_push(new std::vector<std::mutex>(1000))
  , cnd_pull(new std::vector<std::condition_variable>(1000))
  , mtx_pull(new std::vector<std::mutex>(1000))
  , tid_push(0)
  , tid_pull(0)
  , n_pushers(new size_t(1))
  , n_pullers(new size_t(1))
  , n_clones(new size_t(0))
  , buffers_allocated(new bool(false))
  , cloned(false)
  , cur_push_id(0)
  , cur_pull_id(0)
  , ts_buffer(new std::vector<std::vector<uint64_t>>(1, std::vector<uint64_t>(buffer_size, 0)))
  , push_birth_ts_ptr(nullptr)
  , push_stage_start_ts_ptr(nullptr)
  , pull_birth_ts_ptr(nullptr)
  , pull_stage_start_ts_ptr(nullptr)
{
    const std::string name = "Adaptor_m_to_n";
    this->set_name(name);
    this->set_short_name(name);

    if (buffer_size == 0)
    {
        std::stringstream message;
        message << "'buffer_size' has to be greater than 0 ('buffer_size' = " << buffer_size << ").";
        throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }

    if (n_elmts.size() == 0)
    {
        std::stringstream message;
        message << "'n_elmts.size()' has to be greater than 0 ('n_elmts.size()' = " << n_elmts.size() << ").";
        throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }

    for (size_t e = 0; e < n_elmts.size(); e++)
    {
        if (n_elmts[e] == 0)
        {
            std::stringstream message;
            message << "'n_elmts[e]' has to be greater than 0 ('e' = " << e << ", 'n_elmts[e]' = " << n_elmts[e]
                    << ").";
            throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
        }
    }

    this->channels_ptr[0].push_slots_base = this->buffer->data();
    this->channels_ptr[0].pull_slots_base = this->buffer->data();
    this->channels_ptr[0].push_ts_base = (*this->ts_buffer)[0].data();
    this->channels_ptr[0].pull_ts_base = (*this->ts_buffer)[0].data();
    this->channels_ptr[0].push_n_channels = 1;
    this->channels_ptr[0].pull_n_channels = 1;

    this->init();
}

Adaptor_m_to_n::Adaptor_m_to_n(const size_t n_elmts,
                               const std::type_index datatype,
                               const size_t buffer_size,
                               const bool active_waiting)
  : Adaptor_m_to_n(std::vector<size_t>(1, n_elmts),
                   std::vector<std::type_index>(1, datatype),
                   buffer_size,
                   active_waiting)
{
}

void
Adaptor_m_to_n::init()
{
    const std::string name = "Adaptor_m_to_n";
    this->set_name(name);
    this->set_short_name(name);
    this->set_single_wave(true);

    auto& p1 = this->create_task("push");
    std::vector<size_t> p1s_in;
    for (size_t s = 0; s < this->n_sockets; s++)
        p1s_in.push_back(this->create_socket_in(p1, "in" + std::to_string(s), this->n_elmts[s], this->datatype[s]));

    // Capture the contiguous base socket index `p1s_in0` as a scalar instead of `std::vector<size_t> p1s_in`
    // so the lambda is `std::is_trivially_copyable` (8 bytes) and runs through `Task::call_fast_codelet`
    // without `std::function` overhead.
    const size_t n_scks = this->n_sockets;
    const size_t p1s_in0 = n_scks ? p1s_in[0] : 0;
    this->create_codelet(p1,
                         [p1s_in0, n_scks](Module& m, runtime::Task& t, const size_t frame_id) -> int
                         {
                             auto& adp = static_cast<Adaptor_m_to_n&>(m);
                             if (adp.is_no_copy_push())
                             {
                                 adp.wait_push();
                                 // In fast mode (`t.is_fast()`), `Sequence::apply_custom_process_rebind()` updates
                                 // both `t.sockets[s]->dataptr` (`rebind_sockets[rebind_id][s][0]`) and all upstream
                                 // bound sockets (`[1..N]`) in a single pass via `get_empty_buffer(s, swap_buff)`.
                                 // We only update `t.sockets[s]->dataptr` here when `!t.is_fast()` (e.g. debug mode).
                                 if (SPU_TASK_UNLIKELY(!t.is_fast()))
                                     for (size_t s = 0; s < n_scks; s++)
                                         t.sockets[p1s_in0 + s]->dataptr = adp.get_empty_buffer(s);
                             }
                             else
                             {
                                 adp.push(t, p1s_in0, frame_id);
                             }
                             return runtime::status_t::SUCCESS;
                         });

    auto& p2 = this->create_task("pull");
    std::vector<size_t> p2s_out;
    for (size_t s = 0; s < this->n_sockets; s++)
        p2s_out.push_back(this->create_socket_out(p2, "out" + std::to_string(s), this->n_elmts[s], this->datatype[s]));

    const size_t p2s_out0 = n_scks ? p2s_out[0] : 0;
    this->create_codelet(p2,
                         [p2s_out0, n_scks](Module& m, runtime::Task& t, const size_t frame_id) -> int
                         {
                             auto& adp = static_cast<Adaptor_m_to_n&>(m);
                             if (adp.is_no_copy_pull())
                             {
                                 adp.wait_pull();
                                 // In fast mode (`t.is_fast()`), `Sequence::apply_custom_process_rebind()` updates
                                 // both `t.sockets[s]->dataptr` (`rebind_sockets[rebind_id][s][0]`) and all downstream
                                 // bound sockets (`[1..N]`) in a single pass via `get_filled_buffer(s, swap_buff)`.
                                 // We only update `t.sockets[s]->dataptr` here when `!t.is_fast()` so `Task::_exec()`
                                 // sees the new output pointer before printing debug socket contents.
                                 if (SPU_TASK_UNLIKELY(!t.is_fast()))
                                     for (size_t s = 0; s < n_scks; s++)
                                         t.sockets[p2s_out0 + s]->dataptr = adp.get_filled_buffer(s);
                             }
                             else
                             {
                                 adp.pull(t, p2s_out0, frame_id);
                             }
                             return runtime::status_t::SUCCESS;
                         });
}

size_t
Adaptor_m_to_n::get_n_elmts(const size_t sid) const
{
    return this->n_elmts[sid];
}

size_t
Adaptor_m_to_n::get_n_bytes(const size_t sid) const
{
    return this->n_bytes[sid];
}

std::type_index
Adaptor_m_to_n::get_datatype(const size_t sid) const
{
    return this->datatype[sid];
}

bool
Adaptor_m_to_n::is_empty(const size_t id)
{
    return this->channels_ptr[id].is_empty();
}

bool
Adaptor_m_to_n::is_full(const size_t id)
{
    return this->channels_ptr[id].is_full(this->buffer_size);
}

size_t
Adaptor_m_to_n::n_fill_slots(const size_t id)
{
    return this->channels_ptr[id].n_fill_slots(this->buffer_size);
}

size_t
Adaptor_m_to_n::n_free_slots(const size_t id)
{
    return this->channels_ptr[id].n_free_slots(this->buffer_size);
}

bool
Adaptor_m_to_n::is_no_copy_pull() const noexcept
{
    return this->no_copy_pull;
}

bool
Adaptor_m_to_n::is_no_copy_push() const noexcept
{
    return this->no_copy_push;
}

void*
Adaptor_m_to_n::get_empty_buffer(const size_t sid)
{
#ifndef SPU_FAST
    if (SPU_TASK_UNLIKELY(!*this->buffers_allocated)) this->throw_unallocated_buffers_error("get_empty_buffer()");
#endif
    const auto& ch = this->channels_ptr[this->cur_push_id];
    return (void*)ch.push_slots_base[ch.write_slot * this->n_sockets + sid];
}

void*
Adaptor_m_to_n::get_filled_buffer(const size_t sid)
{
#ifndef SPU_FAST
    if (SPU_TASK_UNLIKELY(!*this->buffers_allocated)) this->throw_unallocated_buffers_error("get_filled_buffer()");
#endif
    const auto& ch = this->channels_ptr[this->cur_pull_id];
    return (void*)ch.pull_slots_base[ch.read_slot * this->n_sockets + sid];
}

void*
Adaptor_m_to_n::get_empty_buffer(const size_t sid, void* swap_buffer)
{
#ifndef SPU_FAST
    if (SPU_TASK_UNLIKELY(!*this->buffers_allocated)) this->throw_unallocated_buffers_error("get_empty_buffer()");
#endif
    const auto& ch = this->channels_ptr[this->cur_push_id];
    int8_t*& slot_ref = ch.push_slots_base[ch.write_slot * this->n_sockets + sid];
    void* empty_buffer = (void*)slot_ref;
    slot_ref = (int8_t*)swap_buffer;
    return empty_buffer;
}

void*
Adaptor_m_to_n::get_filled_buffer(const size_t sid, void* swap_buffer)
{
#ifndef SPU_FAST
    if (SPU_TASK_UNLIKELY(!*this->buffers_allocated)) this->throw_unallocated_buffers_error("get_filled_buffer()");
#endif
    const auto& ch = this->channels_ptr[this->cur_pull_id];
    int8_t*& slot_ref = ch.pull_slots_base[ch.read_slot * this->n_sockets + sid];
    void* filled_buffer = (void*)slot_ref;
    slot_ref = (int8_t*)swap_buffer;
    return filled_buffer;
}

void
Adaptor_m_to_n::wake_up_puller()
{
    auto& ch = this->channels_ptr[this->cur_push_id];
    // Forward the current frame's birth timestamp into the ring buffer slot before `publish_push()` publishes the slot
    // to the downstream puller thread (works identically in both `copy` and `no_copy` modes).
    if (this->push_birth_ts_ptr != nullptr) ch.push_ts_base[ch.write_slot] = *this->push_birth_ts_ptr;

    ch.publish_push(this->buffer_size);

    if (!this->active_waiting) // passive waiting
    {
        if (ch.n_fill_slots(this->buffer_size) > 0)
        {
            std::lock_guard<std::mutex> lock((*this->mtx_pull.get())[this->cur_push_id]);
            (*this->cnd_pull.get())[this->cur_push_id].notify_one();
        }
    }

    // Fast round-robin channel advancement without 64-bit integer modulo (`%`):
    // `ch.push_n_channels` resides on the Pusher's own cache line, avoiding a `this->buffer->size()` indirection.
    const size_t n_ch = ch.push_n_channels;
    if (n_ch > 1)
    {
        const size_t next_id = this->cur_push_id + *this->n_pushers;
        this->cur_push_id = (next_id >= n_ch) ? (next_id - n_ch) : next_id;
    }
}

void
Adaptor_m_to_n::wake_up_pusher()
{
    auto& ch = this->channels_ptr[this->cur_pull_id];
    ch.publish_pull(this->buffer_size);

    if (!this->active_waiting) // passive waiting
    {
        if (ch.n_free_slots(this->buffer_size) > 0)
        {
            std::lock_guard<std::mutex> lock((*this->mtx_push.get())[this->cur_pull_id]);
            (*this->cnd_push.get())[this->cur_pull_id].notify_one();
        }
    }

    // Fast round-robin channel advancement without 64-bit integer modulo (`%`):
    // `ch.pull_n_channels` resides on the Puller's own cache line, avoiding a `this->buffer->size()` indirection.
    const size_t n_ch = ch.pull_n_channels;
    if (n_ch > 1)
    {
        const size_t next_id = this->cur_pull_id + *this->n_pullers;
        this->cur_pull_id = (next_id >= n_ch) ? (next_id - n_ch) : next_id;
    }
}

}
}
