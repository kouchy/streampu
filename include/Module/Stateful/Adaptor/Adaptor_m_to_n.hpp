/*!
 * \file
 * \brief Class module::Adaptor_m_to_n.
 */
#ifndef ADAPTOR_M_TO_N_HPP_
#define ADAPTOR_M_TO_N_HPP_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <typeindex>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "Module/Stateful/Stateful.hpp"
#include "Runtime/Sequence/Sequence.hpp"
#include "Runtime/Socket/Socket.hpp"
#include "Runtime/Task/Task.hpp"
#include "Tools/Interface/Interface_waiting.hpp"

namespace spu
{
namespace module
{

// Portable cache-line / coherence-granule alignment size (128 bytes) used to separate the Pusher-owned half and the
// Puller-owned half of `SPSC_channel_sync`, as well as adjacent channels `d` and `d+1`:
// - x86 / x86_64: 128 bytes = 2 x 64-byte cache lines, which defeats the L2 spatial adjacent-line hardware prefetcher
//   and provides enough room (80 bytes used) to embed `inline_forward_slots[2]` and `inline_ts[2]` directly inside the
//   Pusher's half-struct and `inline_return_slots[2]` inside the Puller's half-struct.
// - ARM64 (AArch64): 128 bytes = native L2/SLC cache-line / coherence granule size on Apple Silicon and ARM servers.
// - ARM32 / RISC-V / other: 128 bytes = 2 x 64-byte cache lines, ensuring `sizeof` padding arithmetic never underflows.
#define SPU_CACHE_LINE_SIZE 128

class Adaptor_m_to_n
  : public Stateful
  , public tools::Interface_waiting
{
    friend runtime::Sequence; // Sequence is friend to enable the no copy mode (0 copy)

  public:
    // Portable hardware spin-wait hint (`cpu_relax`):
    // Throttles speculative out-of-order loads inside active spin-wait loops (`while (is_full)` / `while (is_empty)`),
    // preventing memory-order-violation pipeline flushes (~15-25 ns penalty on wakeup) and reducing interconnect
    // contention and power consumption while waiting for the peer core to update its atomic index:
    // - x86 / x86_64: `pause` instruction (`_mm_pause()`).
    // - ARM64 (AArch64): `yield` instruction.
    // - ARM32 (`armhf`): `yield` on ARMv6K/v7/v8, or `nop` on older ARM cores.
    // - RISC-V: `pause` hint encoded as `.insn i 0x0F, 0, x0, x0, 0x010` (`Zihintpause`, backward-compatible `FENCE`).
    // - Fallback (any other architecture): compiler/signal memory barrier `std::atomic_signal_fence`.
    static inline void cpu_relax() noexcept
    {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#elif defined(__arm__)
#if (defined(__ARM_ARCH) && (__ARM_ARCH >= 7)) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6KZ__)
        __asm__ __volatile__("yield" ::: "memory");
#else
        __asm__ __volatile__("nop" ::: "memory");
#endif
#elif defined(__riscv)
        __asm__ __volatile__(".insn i 0x0F, 0, x0, x0, 0x010" ::: "memory");
#else
        std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    }

    // Lock-free Single-Producer Single-Consumer (SPSC) synchronization and inline zero-copy slot state for one
    // inter-stage sub-channel.
    //
    // Architectural optimizations implemented in `SPSC_channel_sync`:
    // 1. Monotonic Release/Acquire Indices (`write_pos` / `read_pos`):
    //    Replaces the legacy shared `std::atomic<uint32_t> counter` (which forced both Pusher and Puller cores to
    //    execute `seq_cst` Read-Modify-Write `counter--` / `counter++` instructions on the same cache line on every
    //    frame) with two monotonically increasing 64-bit counters: `write_pos` written ONLY by the Pusher, and
    //    `read_pos` written ONLY by the Puller via lightweight `memory_order_release` stores.
    // 2. Shadow Indices (`read_pos_cache` private to Pusher, `write_pos_cache` private to Puller):
    //    - The Pusher only reloads `read_pos.load(memory_order_acquire)` when its cached view says the FIFO is full
    //      (`write_pos - read_pos_cache >= capacity`), i.e. at most once every `buffer_size` frames.
    //    - The Puller only reloads `write_pos.load(memory_order_acquire)` when its cached view says the FIFO is empty
    //      (`read_pos == write_pos_cache`), i.e. at most once every `buffer_size` frames.
    // 3. Unidirectional Zero-Copy Slot Exchange (`push_write_slots`/`pull_read_slots` vs.
    //    `pull_write_slots`/`push_read_slots`):
    //    - Historically, in zero-copy mode (`no_copy_push && no_copy_pull`), both the Pusher (`get_empty_buffer(sid,
    //      swap_buff)`) and the Puller (`get_filled_buffer(sid, swap_buff)`) overwrote the SAME slot `(*buffer)[idx]`.
    //      That caused a bidirectional Request-For-Ownership (RFO) write-write cache-line conflict on `(*buffer)[idx]`
    //      on every frame in addition to the `write_pos`/`read_pos` coherence traffic.
    //    - When both ends operate in zero-copy mode (`(*shared_no_copy_flags & 0x3) == 0x3`), buffer pointer handoff
    //      is split into two strictly unidirectional tables:
    //        * Forward table (`push_write_slots` == `pull_read_slots`): written ONLY by the Pusher (depositing the
    //          filled buffer) and read ONLY by the Puller (acquiring the filled buffer).
    //        * Return table (`pull_write_slots` == `push_read_slots`): written ONLY by the Puller (returning the
    //          consumed buffer) and read ONLY by the Pusher (acquiring a free buffer for the next frame).
    // 4. Inline Slot & Timestamp Embedding (`inline_forward_slots[2]`, `inline_return_slots[2]`, `inline_ts[2]`):
    //    - When `stride = buffer_size * n_sockets <= 2` (covering the latency-critical `buffer_size = 1` with 1 or 2
    //      sockets, or `buffer_size = 2` with 1 socket), `push_write_slots`/`pull_read_slots` point directly to
    //      `inline_forward_slots` (located at bytes 48..63 of the Pusher's cache line, right next to `write_pos`), and
    //      `pull_write_slots`/`push_read_slots` point directly to `inline_return_slots` (located at bytes 176..191 of
    //      the Puller's cache line, right next to `read_pos`).
    //    - Consequently, when the Pusher writes the filled buffer pointer into `inline_forward_slots` and publishes
    //      `write_pos + 1`, both stores hit the SAME 64-byte L1 cache line owned by the Pusher; when the Puller wakes
    //      up and reads `write_pos` + `inline_forward_slots`, a SINGLE cross-core cache-line transfer delivers both the
    //      synchronization counter and the data pointer!
    // 5. Dedicated 128-Byte Cache-Line Isolation (`SPU_CACHE_LINE_SIZE`):
    //    Pusher-owned fields (bytes 0..127) and Puller-owned fields (bytes 128..255) are isolated in separate 128-byte
    //    blocks so neither core's stores or hardware spatial prefetcher can invalidate the peer core's L1/L2 lines.
    struct SPSC_channel_sync
    {
        // --- Pusher-owned cache line (written exclusively by the upstream Pusher thread) ---
        std::atomic<uint64_t> write_pos; // Total frames published by the Pusher since reset (offset 0, 8B)
        uint64_t read_pos_cache;         // Pusher's private shadow copy of `read_pos` (offset 8, 8B)
        int8_t** push_read_slots;        // Slot table from which Pusher reads empty buffer (offset 16, 8B)
        int8_t** push_write_slots;       // Slot table into which Pusher writes filled swap_buffer (offset 24, 8B)
        uint64_t* push_ts_base;          // Direct pointer to channel's timestamp ring (offset 32, 8B)
        uint32_t write_slot;             // Current ring-buffer slot index in `[0, capacity - 1]` (offset 40, 4B)
        uint32_t push_n_channels;        // Cached total number of sub-channels (`ppcm`) (offset 44, 4B)
        int8_t* inline_forward_slots[2]; // Inline Pusher->Puller slot storage for stride <= 2 (offset 48..63, 16B)
        uint64_t inline_ts[2];           // Inline birth timestamp storage for buffer_size <= 2 (offset 64..79, 16B)
        uint8_t pad_pusher[SPU_CACHE_LINE_SIZE - sizeof(std::atomic<uint64_t>) - sizeof(uint64_t) -
                           2 * sizeof(int8_t**) - sizeof(uint64_t*) - 2 * sizeof(uint32_t) - 2 * sizeof(int8_t*) -
                           2 * sizeof(uint64_t)];

        // --- Puller-owned cache line (written exclusively by the downstream Puller thread) ---
        std::atomic<uint64_t> read_pos; // Total frames consumed by the Puller since reset (offset 128, 8B)
        uint64_t write_pos_cache;       // Puller's private shadow copy of `write_pos` (offset 136, 8B)
        int8_t** pull_read_slots;       // Slot table from which Puller reads filled buffer (offset 144, 8B)
        int8_t** pull_write_slots;      // Slot table into which Puller writes returned swap_buffer (offset 152, 8B)
        uint64_t* pull_ts_base;         // Direct pointer to channel's timestamp ring (offset 160, 8B)
        uint32_t read_slot;             // Current ring-buffer slot index in `[0, capacity - 1]` (offset 168, 4B)
        uint32_t pull_n_channels;       // Cached total number of sub-channels (`ppcm`) (offset 172, 4B)
        int8_t* inline_return_slots[2]; // Inline Puller->Pusher slot storage for stride <= 2 (offset 176..191, 16B)
        uint8_t pad_puller[SPU_CACHE_LINE_SIZE - sizeof(std::atomic<uint64_t>) - sizeof(uint64_t) -
                           2 * sizeof(int8_t**) - sizeof(uint64_t*) - 2 * sizeof(uint32_t) - 2 * sizeof(int8_t*)];

        inline SPSC_channel_sync() noexcept
          : write_pos(0)
          , read_pos_cache(0)
          , push_read_slots(nullptr)
          , push_write_slots(nullptr)
          , push_ts_base(nullptr)
          , write_slot(0)
          , push_n_channels(1)
          , inline_forward_slots{ nullptr, nullptr }
          , inline_ts{ 0, 0 }
          , pad_pusher{}
          , read_pos(0)
          , write_pos_cache(0)
          , pull_read_slots(nullptr)
          , pull_write_slots(nullptr)
          , pull_ts_base(nullptr)
          , read_slot(0)
          , pull_n_channels(1)
          , inline_return_slots{ nullptr, nullptr }
          , pad_puller{}
        {
        }

        inline SPSC_channel_sync(const SPSC_channel_sync& other) noexcept
          : write_pos(other.write_pos.load(std::memory_order_relaxed))
          , read_pos_cache(other.read_pos_cache)
          , push_read_slots(other.push_read_slots)
          , push_write_slots(other.push_write_slots)
          , push_ts_base(other.push_ts_base)
          , write_slot(other.write_slot)
          , push_n_channels(other.push_n_channels)
          , inline_forward_slots{ other.inline_forward_slots[0], other.inline_forward_slots[1] }
          , inline_ts{ other.inline_ts[0], other.inline_ts[1] }
          , pad_pusher{}
          , read_pos(other.read_pos.load(std::memory_order_relaxed))
          , write_pos_cache(other.write_pos_cache)
          , pull_read_slots(other.pull_read_slots)
          , pull_write_slots(other.pull_write_slots)
          , pull_ts_base(other.pull_ts_base)
          , read_slot(other.read_slot)
          , pull_n_channels(other.pull_n_channels)
          , inline_return_slots{ other.inline_return_slots[0], other.inline_return_slots[1] }
          , pad_puller{}
        {
        }

        inline SPSC_channel_sync& operator=(const SPSC_channel_sync& other) noexcept
        {
            if (this != &other)
            {
                this->write_pos.store(other.write_pos.load(std::memory_order_relaxed), std::memory_order_relaxed);
                this->read_pos_cache = other.read_pos_cache;
                this->push_read_slots = other.push_read_slots;
                this->push_write_slots = other.push_write_slots;
                this->push_ts_base = other.push_ts_base;
                this->write_slot = other.write_slot;
                this->push_n_channels = other.push_n_channels;
                this->inline_forward_slots[0] = other.inline_forward_slots[0];
                this->inline_forward_slots[1] = other.inline_forward_slots[1];
                this->inline_ts[0] = other.inline_ts[0];
                this->inline_ts[1] = other.inline_ts[1];
                this->read_pos.store(other.read_pos.load(std::memory_order_relaxed), std::memory_order_relaxed);
                this->write_pos_cache = other.write_pos_cache;
                this->pull_read_slots = other.pull_read_slots;
                this->pull_write_slots = other.pull_write_slots;
                this->pull_ts_base = other.pull_ts_base;
                this->read_slot = other.read_slot;
                this->pull_n_channels = other.pull_n_channels;
                this->inline_return_slots[0] = other.inline_return_slots[0];
                this->inline_return_slots[1] = other.inline_return_slots[1];
            }
            return *this;
        }

        inline void reset() noexcept
        {
            this->write_pos.store(0, std::memory_order_relaxed);
            this->read_pos_cache = 0;
            this->write_slot = 0;
            this->read_pos.store(0, std::memory_order_relaxed);
            this->write_pos_cache = 0;
            this->read_slot = 0;
        }

        // Called by the Pusher thread: checks if all `capacity` slots are occupied.
        // 1. Fast path: reads only the Pusher's own cache line (`write_pos` and `read_pos_cache`).
        // 2. Slow path: loads the Puller's `read_pos` with `memory_order_acquire` into a local register `r`.
        //    CRITICAL (Zero-Store Spin): while spinning (`w - r >= capacity`), we MUST NOT write `r` back to
        //    `this->read_pos_cache`. Writing to `read_pos_cache` on every spin iteration would dirty the Pusher's
        //    cache line (which also holds `write_pos`), forcing a `Shared -> Modified` coherence transition that
        //    invalidates the Puller's L1 cache while the Puller is trying to read `write_pos`. We only update
        //    `this->read_pos_cache = r` once a free slot is observed (`return false`).
        inline bool is_full(const size_t capacity) noexcept
        {
            const uint64_t w = this->write_pos.load(std::memory_order_relaxed);
            if (w - this->read_pos_cache < capacity) return false;
            const uint64_t r = this->read_pos.load(std::memory_order_acquire);
            if (w - r >= capacity) return true;
            this->read_pos_cache = r;
            return false;
        }

        // Called by the Puller thread: checks if no filled slot is available.
        // 1. Fast path: reads only the Puller's own cache line (`read_pos` and `write_pos_cache`).
        // 2. Slow path: loads the Pusher's `write_pos` with `memory_order_acquire` into a local register `w`.
        //    CRITICAL (Zero-Store Spin): while spinning (`r == w`), we MUST NOT write `w` back to
        //    `this->write_pos_cache`. Writing to `write_pos_cache` on every spin iteration would dirty the Puller's
        //    cache line (which also holds `read_pos`), forcing a `Shared -> Modified` coherence transition that
        //    invalidates the Pusher's L1 cache while the Pusher is trying to read `read_pos`. We only update
        //    `this->write_pos_cache = w` once at least one filled slot is observed (`return false`).
        inline bool is_empty() noexcept
        {
            const uint64_t r = this->read_pos.load(std::memory_order_relaxed);
            if (r != this->write_pos_cache) return false;
            const uint64_t w = this->write_pos.load(std::memory_order_acquire);
            if (r == w) return true;
            this->write_pos_cache = w;
            return false;
        }

        // Called by the Pusher thread after writing/swapping the slot at `write_slot`:
        // advances `write_slot` without integer division and publishes `write_pos + 1` with `memory_order_release`.
        inline void publish_push(const size_t capacity) noexcept
        {
            const uint32_t next_slot = this->write_slot + 1;
            this->write_slot = (next_slot == static_cast<uint32_t>(capacity)) ? 0 : next_slot;
            const uint64_t w = this->write_pos.load(std::memory_order_relaxed);
            this->write_pos.store(w + 1, std::memory_order_release);
        }

        // Called by the Puller thread after reading/swapping the slot at `read_slot`:
        // advances `read_slot` without integer division and publishes `read_pos + 1` with `memory_order_release`.
        inline void publish_pull(const size_t capacity) noexcept
        {
            const uint32_t next_slot = this->read_slot + 1;
            this->read_slot = (next_slot == static_cast<uint32_t>(capacity)) ? 0 : next_slot;
            const uint64_t r = this->read_pos.load(std::memory_order_relaxed);
            this->read_pos.store(r + 1, std::memory_order_release);
        }

        // Non-mutating observers (safe to call from any thread without touching the other thread's shadow cache).
        // Uses unsigned modular subtraction `w - r`, which remains exact even across unsigned integer wrap-around
        // (`UINT64_MAX -> 0`). If `r` was sampled slightly ahead of `w` due to a concurrent pull, `w - r` wraps to
        // a large unsigned value (`> capacity`) and is clamped to `0`.
        inline size_t n_fill_slots(const size_t capacity) const noexcept
        {
            const uint64_t w = this->write_pos.load(std::memory_order_acquire);
            const uint64_t r = this->read_pos.load(std::memory_order_acquire);
            const uint64_t diff = w - r;
            return (diff <= capacity) ? static_cast<size_t>(diff) : 0;
        }

        inline size_t n_free_slots(const size_t capacity) const noexcept
        {
            const size_t filled = this->n_fill_slots(capacity);
            return capacity - filled;
        }
    };

  protected:
    std::vector<size_t> n_elmts;
    std::vector<size_t> n_bytes;
    const std::vector<std::type_index> datatype;
    const size_t buffer_size;
    const size_t n_sockets;

    // Flat 1D slot-major buffer tables indexed as `(d * buffer_size + b) * n_sockets + s`.
    // When both `no_copy_push` and `no_copy_pull` are active, `buffer` serves as the unidirectional forward slot
    // table (written ONLY by Pusher, read ONLY by Puller) and `return_buffer` serves as the unidirectional return
    // slot table (written ONLY by Puller, read ONLY by Pusher), eliminating write-write RFO cache-line conflicts.
    // Furthermore, for `buffer_size * n_sockets <= 2`, slots are stored directly inside `SPSC_channel_sync`'s
    // `inline_forward_slots` (on the Pusher's cache line alongside `write_pos`) and `inline_return_slots` (on the
    // Puller's cache line alongside `read_pos`), achieving 1-cache-line zero-copy handoff per direction.
    std::shared_ptr<std::vector<int8_t*>> buffer;
    std::shared_ptr<std::vector<int8_t*>> return_buffer;
    std::vector<int8_t*> buffer_to_free;

    std::shared_ptr<std::vector<SPSC_channel_sync, tools::aligned_allocator<SPSC_channel_sync>>> channels;
    SPSC_channel_sync*
      channels_ptr; // Direct 128B-aligned pointer to `channels->data()` avoiding `shared_ptr` + `vector` indirection

    std::shared_ptr<std::atomic<bool>> waiting_canceled;
    std::atomic<bool>* waiting_canceled_ptr;
    std::shared_ptr<uint8_t> shared_no_copy_flags; // Bit 0: `no_copy_push`, Bit 1: `no_copy_pull`

    bool no_copy_pull;
    bool no_copy_push;

    const bool active_waiting;

    std::shared_ptr<std::vector<std::condition_variable>> cnd_push;
    std::shared_ptr<std::vector<std::mutex>> mtx_push;
    std::shared_ptr<std::vector<std::condition_variable>> cnd_pull;
    std::shared_ptr<std::vector<std::mutex>> mtx_pull;

    int tid_push;
    int tid_pull;
    std::shared_ptr<size_t> n_pushers;
    std::shared_ptr<size_t> n_pullers;
    std::shared_ptr<size_t> n_clones;

    std::shared_ptr<bool> buffers_allocated;
    bool cloned;

    // Shared ring buffer of frame birth timestamps (`[queue_id][slot_id]`) and per-thread pointers used when
    // `Pipeline::set_rt_stats(true)` is enabled.
    std::shared_ptr<std::vector<std::vector<uint64_t>>> ts_buffer;

    // Separate Pusher-local and Puller-local hot fields onto distinct cache lines so that when `Pipeline` shares
    // the master `Adaptor_m_to_n` instance `adp` (`t == 0`) between Stage `sta` (`push`) and Stage `sta + 1` (`pull`),
    // Pusher and Puller never suffer false sharing on `cur_push_id` vs. `cur_pull_id`.
    size_t cur_push_id;
    uint64_t* push_birth_ts_ptr;
    uint64_t* push_stage_start_ts_ptr;
    uint8_t pad_adp_push_pull[SPU_CACHE_LINE_SIZE - sizeof(size_t) - 2 * sizeof(uint64_t*)];

    size_t cur_pull_id;
    uint64_t* pull_birth_ts_ptr;
    uint64_t* pull_stage_start_ts_ptr;

  public:
    inline Adaptor_m_to_n(const size_t n_elmts,
                          const std::type_index datatype,
                          const size_t buffer_size = 1,
                          const bool active_waiting = true);
    inline Adaptor_m_to_n(const std::vector<size_t>& n_elmts,
                          const std::vector<std::type_index>& datatype,
                          const size_t buffer_size = 1,
                          const bool active_waiting = true);
    inline size_t get_n_elmts(const size_t sid = 0) const;
    inline size_t get_n_bytes(const size_t sid = 0) const;
    inline std::type_index get_datatype(const size_t sid = 0) const;
    void reset();
    virtual ~Adaptor_m_to_n();
    virtual Adaptor_m_to_n* clone() const;
    virtual void set_n_frames(const size_t n_frames);

    virtual void wake_up();
    void cancel_waiting();

    void add_pusher();
    void add_puller();
    void alloc_buffers();

    inline void set_push_rt_stats_ptrs(uint64_t* birth_ts_ptr, uint64_t* stage_start_ts_ptr = nullptr) noexcept
    {
        this->push_birth_ts_ptr = birth_ts_ptr;
        this->push_stage_start_ts_ptr = stage_start_ts_ptr;
    }
    inline void set_pull_rt_stats_ptrs(uint64_t* birth_ts_ptr, uint64_t* stage_start_ts_ptr) noexcept
    {
        this->pull_birth_ts_ptr = birth_ts_ptr;
        this->pull_stage_start_ts_ptr = stage_start_ts_ptr;
    }

  protected:
    void send_cancel_signal();
    inline void init();
    void deep_copy(const Adaptor_m_to_n& m);
    inline bool is_full(const size_t id);
    inline bool is_empty(const size_t id);
    inline size_t n_free_slots(const size_t id);
    inline size_t n_fill_slots(const size_t id);
    inline void update_slot_pointers() noexcept;

    void set_no_copy_push(const bool no_copy_push);
    void set_no_copy_pull(const bool no_copy_pull);
    inline bool is_no_copy_push() const noexcept;
    inline bool is_no_copy_pull() const noexcept;
    void reset_buffer();

    void push(const runtime::Task& t, const size_t base_sck, const size_t frame_id);
    void pull(runtime::Task& t, const size_t base_sck, const size_t frame_id);

    inline void wait_push();
    inline void wait_pull();
    void wait_push_slow();
    void wait_pull_slow();
    inline void* get_empty_buffer(const size_t sid);
    inline void* get_filled_buffer(const size_t sid);
    inline void* get_empty_buffer(const size_t sid, void* swap_buffer);
    inline void* get_filled_buffer(const size_t sid, void* swap_buffer);
    inline void wake_up_pusher();
    inline void wake_up_puller();

  private:
    [[noreturn]] void throw_unallocated_buffers_error(const char* func_name) const;
    [[noreturn]] void throw_invalid_pusher_error() const;
    [[noreturn]] void throw_invalid_puller_error() const;
};
}
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include "Module/Stateful/Adaptor/Adaptor_m_to_n.hxx"
#endif

#endif /* ADAPTOR_M_TO_N_HPP_ */
