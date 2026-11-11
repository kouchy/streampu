/*!
 * \file
 * \brief Class tools::Statistics.
 */
#ifndef STATISTICS_HPP_
#define STATISTICS_HPP_

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "Module/Module.hpp"
#include "Runtime/Task/Task.hpp"

namespace spu
{
namespace runtime
{
class Sequence;
class Pipeline;
}
namespace tools
{
// Low-overhead, architecture-portable timestamp counter for Real-Time (RT) latency and jitter measurements:
// - x86 / x86_64: reads the invariant Time Stamp Counter via `rdtsc` (~2.5 ns, 1 instruction).
// - ARM64 (AArch64): reads the EL0 virtual timer counter register `cntvct_el0` (~1-2 ns, 1 instruction).
// - ARM32 (`armhf`, ARMv8 AArch32 `__ARM_ARCH >= 8`): reads the 64-bit Generic Timer `CNTVCT` via `mrrc p15, 1, c14`
//   (1 instruction; on older ARMv7-A where `CNTVCT` is optional, e.g. Cortex-A9 Zynq-7000, falls through to
//   `steady_clock::now()` which uses the Linux vDSO without risking `SIGILL`).
// - RISC-V (RV64 / RV32 `Zicntr`): reads the `time` CSR via `rdtime` (~1-2 ns, 1 instruction).
// - Fallback (ARMv7-A and any other architecture): uses C++11 `std::chrono::steady_clock::now()` in nanoseconds.
// Ticks are converted to nanoseconds after execution using the wall-clock/tick ratio calibrated over the run.
inline uint64_t
read_tsc() noexcept
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif defined(__x86_64__) || defined(__i386__)
    // Serialized read: `rdtscp` waits until all previous instructions have executed, and the trailing `lfence`
    // prevents later instructions from starting before the counter is read (plain `rdtsc` can be reordered).
    uint32_t lo, hi, aux;
    __asm__ __volatile__("rdtscp\n\tlfence" : "=a"(lo), "=d"(hi), "=c"(aux) : : "memory");
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
    // `isb` flushes the pipeline so that the counter read is not speculated before previous instructions.
    uint64_t val;
    __asm__ __volatile__("isb\n\tmrs %0, cntvct_el0" : "=r"(val) : : "memory");
    return val;
#elif defined(__arm__) && defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
    uint32_t lo, hi;
    __asm__ __volatile__("mrrc p15, 1, %0, %1, c14" : "=r"(lo), "=r"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__riscv) && (__riscv_xlen == 64)
    uint64_t val;
    __asm__ __volatile__("rdtime %0" : "=r"(val));
    return val;
#elif defined(__riscv) && (__riscv_xlen == 32)
    uint32_t hi, lo, tmp;
    __asm__ __volatile__("1:\n\t"
                         "rdtimeh %0\n\t"
                         "rdtime  %1\n\t"
                         "rdtimeh %2\n\t"
                         "bne %0, %2, 1b"
                         : "=&r"(hi), "=&r"(lo), "=&r"(tmp));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#else
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count());
#endif
}

// High-Dynamic-Range (HDR) sub-bucketed base-2 histogram and online accumulator for Real-Time (RT) latency & jitter.
// Uses 128 linear sub-buckets per power-of-2 octave (`SUB_BITS = 7`, 4096 buckets = 16 KB total, fitting inside L1d
// cache):
// - Values in `[0..255]` ticks are mapped with exact 1-tick resolution.
// - Values `>= 256` ticks are mapped with `<= 0.39%` relative quantization error up to `2^38` ticks.
// Recording a sample in the hot loop requires no heap allocation, no branches, and no floating-point division.
struct Rt_stats
{
    static constexpr size_t SUB_BITS = 7;
    static constexpr size_t SUB_COUNT = (size_t)1 << SUB_BITS; // 128
    static constexpr size_t LINEAR_LIMIT = SUB_COUNT * 2;      // 256
    static constexpr size_t N_BUCKETS = 4096;

    uint64_t n_calls;
    uint64_t tot_ticks;
    double sum_sq_ticks;
    uint64_t min_ticks;
    uint64_t max_ticks;
    uint32_t buckets[N_BUCKETS];

    Rt_stats() noexcept { this->reset(); }

    inline void reset() noexcept
    {
        this->n_calls = 0;
        this->tot_ticks = 0;
        this->sum_sq_ticks = 0.0;
        this->min_ticks = std::numeric_limits<uint64_t>::max();
        this->max_ticks = 0;
        std::memset(this->buckets, 0, sizeof(this->buckets));
    }

    static inline size_t tick_to_bucket(const uint64_t ticks) noexcept
    {
        if (ticks < LINEAR_LIMIT) return static_cast<size_t>(ticks);
#if defined(__GNUC__) || defined(__clang__)
        const size_t msb = 63 - static_cast<size_t>(__builtin_clzll(ticks));
#else
        size_t msb = 7;
        uint64_t tmp = ticks >> 8;
        while (tmp != 0)
        {
            tmp >>= 1;
            msb++;
        }
#endif
        const size_t sub = static_cast<size_t>((ticks >> (msb - SUB_BITS)) & (SUB_COUNT - 1));
        const size_t idx = ((msb - (SUB_BITS - 1)) << SUB_BITS) | sub;
        return idx < N_BUCKETS ? idx : (N_BUCKETS - 1);
    }

    static inline double bucket_to_ticks(const size_t idx) noexcept
    {
        if (idx < LINEAR_LIMIT) return static_cast<double>(idx);
        const size_t msb = (idx >> SUB_BITS) + (SUB_BITS - 1);
        const size_t sub = idx & (SUB_COUNT - 1);
        const uint64_t step = uint64_t(1) << (msb - SUB_BITS);
        const uint64_t low = (uint64_t(SUB_COUNT) + sub) << (msb - SUB_BITS);
        return static_cast<double>(low) + 0.5 * static_cast<double>(step);
    }

    inline void add_sample(const uint64_t ticks) noexcept
    {
        this->n_calls++;
        this->tot_ticks += ticks;
        const double d = static_cast<double>(ticks);
        this->sum_sq_ticks += d * d;
        if (ticks < this->min_ticks) this->min_ticks = ticks;
        if (ticks > this->max_ticks) this->max_ticks = ticks;
        this->buckets[tick_to_bucket(ticks)]++;
    }

    inline void merge(const Rt_stats& other) noexcept
    {
        if (other.n_calls == 0) return;
        this->n_calls += other.n_calls;
        this->tot_ticks += other.tot_ticks;
        this->sum_sq_ticks += other.sum_sq_ticks;
        if (other.min_ticks < this->min_ticks) this->min_ticks = other.min_ticks;
        if (other.max_ticks > this->max_ticks) this->max_ticks = other.max_ticks;
        for (size_t i = 0; i < N_BUCKETS; i++)
            this->buckets[i] += other.buckets[i];
    }

    inline double get_avg_ticks() const noexcept
    {
        return this->n_calls ? (static_cast<double>(this->tot_ticks) / static_cast<double>(this->n_calls)) : 0.0;
    }

    inline double get_std_ticks() const noexcept
    {
        if (this->n_calls <= 1) return 0.0;
        const double mean = this->get_avg_ticks();
        const double var = (this->sum_sq_ticks / static_cast<double>(this->n_calls)) - (mean * mean);
        return var > 0.0 ? std::sqrt(var) : 0.0;
    }

    inline double get_percentile_ticks(const double p) const noexcept
    {
        if (this->n_calls == 0) return 0.0;
        const double frac = std::min(1.0, std::max(0.0, p * 0.01));
        const uint64_t target =
          std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(frac * static_cast<double>(this->n_calls))));
        uint64_t cum = 0;
        for (size_t i = 0; i < N_BUCKETS; i++)
        {
            cum += this->buckets[i];
            if (cum >= target)
            {
                const double val = bucket_to_ticks(i);
                const double lo = static_cast<double>(this->min_ticks);
                const double hi = static_cast<double>(this->max_ticks);
                return std::min(hi, std::max(lo, val));
            }
        }
        return static_cast<double>(this->max_ticks);
    }
};

class Statistics
{
  protected:
    Statistics() = default;

  public:
    virtual ~Statistics() = default;

    template<class MODULE_OR_TASK>
    static void show(std::vector<MODULE_OR_TASK*> modules_or_tasks,
                     const bool ordered = false,
                     const bool display_thr = true,
                     std::ostream& stream = std::cout);
    template<class MODULE_OR_TASK>
    static void show(std::vector<std::vector<MODULE_OR_TASK*>> modules_or_tasks,
                     const bool ordered = false,
                     const bool display_thr = true,
                     std::ostream& stream = std::cout);

    /*!
     * \brief Displays Real-Time (RT) frame-level latency percentiles (Min, Avg, Std, P50, P90, P99, P99.9, P99.99,
     *        Max) and throughput tail metrics (Avg, 1% low, 0.1% low, 0.01% low, Min) for a Sequence.
     *        Requires `sequence.set_rt_stats(true, n_skip_frames)` to be called prior to execution.
     * \param sequence:       Sequence whose RT statistics are displayed.
     * \param display_thr:    true to display the throughput columns (default: true).
     * \param thr_in_frames:  false to display throughput in streams/s (`strm/s`), true to display in frames/s (`frm/s`,
     *                        multiplying the stream rate by `sequence.get_n_frames()`) (default: false).
     * \param stream:         Output stream (default: `std::cout`).
     */
    static void show_rt(const runtime::Sequence& sequence,
                        const bool display_thr = true,
                        const bool thr_in_frames = false,
                        std::ostream& stream = std::cout);

    static inline void show_rt(const runtime::Sequence& sequence, const bool display_thr, std::ostream& stream)
    {
        Statistics::show_rt(sequence, display_thr, false, stream);
    }

    /*!
     * \brief Displays Real-Time (RT) frame-level latency percentiles and throughput tail metrics for a Pipeline:
     *        1) Per-stage processing statistics (excluding inter-stage `wait_pull`/`wait_push` blocking time).
     *        2) End-to-End (`PIPELINE E2E`) statistics from the start of stage 0 to the end of the last stage
     *           (including inter-stage wait and FIFO queuing time).
     *        Requires `pipeline.set_rt_stats(true, n_skip_frames)` to be called prior to execution.
     * \param pipeline:       Pipeline whose RT statistics are displayed.
     * \param display_thr:    true to display the throughput columns (default: true).
     * \param thr_in_frames:  false to display throughput in streams/s (`strm/s`), true to display in frames/s (`frm/s`,
     *                        multiplying the stream rate by `stage->get_n_frames()`) (default: false).
     * \param stream:         Output stream (default: `std::cout`).
     */
    static void show_rt(const runtime::Pipeline& pipeline,
                        const bool display_thr = true,
                        const bool thr_in_frames = false,
                        std::ostream& stream = std::cout);

    static inline void show_rt(const runtime::Pipeline& pipeline, const bool display_thr, std::ostream& stream)
    {
        Statistics::show_rt(pipeline, display_thr, false, stream);
    }

  private:
    template<class MODULE = const module::Module>
    static void show_modules(std::vector<MODULE*> modules,
                             const bool ordered = false,
                             const bool display_thr = true,
                             std::ostream& stream = std::cout);
    template<class TASK = const runtime::Task>
    static void show_tasks(std::vector<TASK*> tasks,
                           const bool ordered = false,
                           const bool display_thr = true,
                           std::ostream& stream = std::cout);
    template<class MODULE = const module::Module>
    static void show_modules(std::vector<std::vector<MODULE*>> modules,
                             const bool ordered = false,
                             const bool display_thr = true,
                             std::ostream& stream = std::cout);
    template<class TASK = const runtime::Task>
    static void show_tasks(std::vector<std::vector<TASK*>> tasks,
                           const bool ordered = false,
                           const bool display_thr = true,
                           std::ostream& stream = std::cout);

    static void separation1(const bool display_thr = true, std::ostream& stream = std::cout);

    static void separation2(const bool display_thr = true, std::ostream& stream = std::cout);

    static void show_header(const bool display_thr = true, std::ostream& stream = std::cout);

    static void show_task(const float total_sec,
                          const std::string& module_sname,
                          const std::string& task_name,
                          const bool task_replicability,
                          const int task_order,
                          const size_t task_n_elmts,
                          const uint32_t task_n_calls,
                          const std::chrono::nanoseconds task_tot_duration,
                          const std::chrono::nanoseconds task_min_duration,
                          const std::chrono::nanoseconds task_max_duration,
                          const bool display_thr = true,
                          std::ostream& stream = std::cout);

    static void show_timer(const float total_sec,
                           const uint32_t task_n_calls,
                           const size_t timer_n_elmts,
                           const std::string& timer_name,
                           const uint32_t timer_n_calls,
                           const std::chrono::nanoseconds timer_tot_duration,
                           const std::chrono::nanoseconds timer_min_duration,
                           const std::chrono::nanoseconds timer_max_duration,
                           std::ostream& stream = std::cout);
};

using Stats = Statistics;
}
}

#endif /* STATISTICS_HPP_ */
