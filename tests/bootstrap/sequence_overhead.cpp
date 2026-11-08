#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <streampu.hpp>

using namespace spu;
using namespace spu::runtime;

#if defined(_MSC_VER)
#define SPU_BENCH_NOINLINE __declspec(noinline)
#else
#define SPU_BENCH_NOINLINE __attribute__((noinline))
#endif

// Empty out-of-line kernels with compiler memory/register barriers so the compiler cannot inline, constant-fold, or
// dead-code-eliminate the function call or the socket pointer materializations in the codelet.
SPU_BENCH_NOINLINE void
empty_kernel_0(const size_t frame_id) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(frame_id) : "memory");
#else
    (void)frame_id;
#endif
}

SPU_BENCH_NOINLINE void
empty_kernel_1(const void* p0, const size_t frame_id) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(p0), "r"(frame_id) : "memory");
#else
    (void)p0;
    (void)frame_id;
#endif
}

SPU_BENCH_NOINLINE void
empty_kernel_2(const void* p0, const void* p1, const size_t frame_id) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(p0), "r"(p1), "r"(frame_id) : "memory");
#else
    (void)p0;
    (void)p1;
    (void)frame_id;
#endif
}

SPU_BENCH_NOINLINE void
empty_kernel_n(const void* const* ptrs, const size_t n_ptrs, const size_t frame_id) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(ptrs), "r"(n_ptrs), "r"(frame_id) : "memory");
#else
    (void)ptrs;
    (void)n_ptrs;
    (void)frame_id;
#endif
}

struct Bench_result
{
    std::string name;
    size_t n_tasks;
    double ns_per_exec;
    double ns_per_task;
    double ticks_per_task;
    double std_ns_per_task;
    double p50_ns_per_task;
    double p99_ns_per_task;
    double p999_ns_per_task;
    double p9999_ns_per_task;
};

static void
print_table_header(const std::string& title)
{
    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold
              << "====================================================================================================="
                 "======================"
              << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold << " " << title << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold
              << "-----------------------------------------------------------------------------------------------------"
                 "----------------------"
              << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold
              << " BENCHMARK CASE                          | TASKS ||  NS/FRAME |   NS/TASK | TICKS/TSK ||   STD(ns) | "
                 "  P50(ns) |   P99(ns) | P99.9(ns) |P99.99(ns)"
              << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold
              << "-----------------------------------------|-------||-----------|-----------|-----------||-----------|-"
                 "----------|-----------|-----------|----------"
              << rang::style::reset << std::endl;
}

static void
print_row(const Bench_result& r)
{
    std::cout << "#  " << std::left << std::setw(39) << r.name.substr(0, 39) << std::right << " | " << std::setw(5)
              << r.n_tasks << " || " << std::fixed << std::setprecision(2) << std::setw(9) << r.ns_per_exec << " | "
              << std::setw(9) << r.ns_per_task << " | " << std::setw(9) << r.ticks_per_task << " || " << std::setw(9)
              << r.std_ns_per_task << " | " << std::setw(9) << r.p50_ns_per_task << " | " << std::setw(9)
              << r.p99_ns_per_task << " | " << std::setw(9) << r.p999_ns_per_task << " | " << std::setw(9)
              << r.p9999_ns_per_task << std::endl;
}

static Bench_result
run_sequence_bench(const std::string& name,
                   runtime::Sequence& seq,
                   const size_t n_exec,
                   const bool use_statuses = false,
                   const size_t task_divisor_override = 0)
{
    const size_t n_tasks = task_divisor_override ? task_divisor_override : seq.get_tasks_per_types().size();

    // Warmup run (without RT stats) to warm caches and branch predictors
    {
        size_t w = 0;
        const size_t n_warm = std::min<size_t>(n_exec / 10 + 100, 10000);
        seq.set_rt_stats(false);
        if (!use_statuses)
            seq.exec([&w, n_warm]() { return ++w >= n_warm; });
        else
            seq.exec([&w, n_warm](const std::vector<const int*>&) { return ++w >= n_warm; });
    }

    // 1. Uninstrumented run for pure wall-clock ns/task and ticks/task
    size_t c1 = 0;
    const auto t0_clk = std::chrono::steady_clock::now();
    const uint64_t t0_tsc = tools::read_tsc();
    if (!use_statuses)
        seq.exec([&c1, n_exec]() { return ++c1 >= n_exec; });
    else
        seq.exec([&c1, n_exec](const std::vector<const int*>&) { return ++c1 >= n_exec; });
    const uint64_t t1_tsc = tools::read_tsc();
    const auto t1_clk = std::chrono::steady_clock::now();

    const double dt_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_clk - t0_clk).count();
    const double dt_ticks = (double)(t1_tsc - t0_tsc);
    const double ns_per_exec = dt_ns / (double)n_exec;
    const double ns_per_task = ns_per_exec / (double)n_tasks;
    const double ticks_per_task = (dt_ticks / (double)n_exec) / (double)n_tasks;

    // 2. RT-stats run to collect HDR latency/jitter percentiles per frame (normalized per task)
    seq.reset_rt_stats();
    seq.set_rt_stats(true);
    size_t c2 = 0;
    if (!use_statuses)
        seq.exec([&c2, n_exec]() { return ++c2 >= n_exec; });
    else
        seq.exec([&c2, n_exec](const std::vector<const int*>&) { return ++c2 >= n_exec; });

    const auto& st = seq.get_rt_stats_per_thread()[0];
    const double ns_per_tick = seq.get_rt_ns_per_tick() / (double)n_tasks;
    seq.set_rt_stats(false);

    Bench_result res;
    res.name = name;
    res.n_tasks = n_tasks;
    res.ns_per_exec = ns_per_exec;
    res.ns_per_task = ns_per_task;
    res.ticks_per_task = ticks_per_task;
    res.std_ns_per_task = st.get_std_ticks() * ns_per_tick;
    res.p50_ns_per_task = st.get_percentile_ticks(50.0) * ns_per_tick;
    res.p99_ns_per_task = st.get_percentile_ticks(99.0) * ns_per_tick;
    res.p999_ns_per_task = st.get_percentile_ticks(99.9) * ns_per_tick;
    res.p9999_ns_per_task = st.get_percentile_ticks(99.99) * ns_per_tick;
    return res;
}

int
main(int argc, char** argv)
{
    tools::Signal_handler::init();

    option longopts[] = { { "n-exec", required_argument, NULL, 'e' },
                          { "data-length", required_argument, NULL, 'd' },
                          { "rt-stats", no_argument, NULL, 'r' },
                          { "help", no_argument, NULL, 'h' },
                          { NULL, 0, NULL, 0 } };

    size_t n_exec = 500000;
    size_t data_length = 64;
    bool show_full_rt_tables = false;

    while (1)
    {
        const int opt = getopt_long(argc, argv, "e:d:rh", longopts, 0);
        if (opt == -1) break;
        switch (opt)
        {
            case 'e':
                n_exec = atoi(optarg);
                break;
            case 'd':
                data_length = atoi(optarg);
                break;
            case 'r':
                show_full_rt_tables = true;
                break;
            case 'h':
                std::cout << "usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "  -e, --n-exec       Number of sequence executions per benchmark [" << n_exec << "]"
                          << std::endl;
                std::cout << "  -d, --data-length  Socket buffer length in bytes               [" << data_length << "]"
                          << std::endl;
                std::cout << "  -r, --rt-stats     Also print full tools::Stats::show_rt tables [false]" << std::endl;
                return 0;
            default:
                break;
        }
    }

    std::cout << "########################################################" << std::endl;
    std::cout << "# StreamPU Empty-Task Runtime Overhead Micro-Benchmark #" << std::endl;
    std::cout << "########################################################" << std::endl;
    std::cout << "#   - n_exec      = " << n_exec << std::endl;
    std::cout << "#   - data_length = " << data_length << " bytes" << std::endl;

    // =================================================================================================================
    // 0. Bare C++ noinline function call baseline (8 calls per iteration to match an 8-task chain)
    // =================================================================================================================
    print_table_header("1. BASELINE & SOCKET TYPES (8 empty tasks per sequence)");
    {
        const size_t n_tasks = 8;
        for (size_t i = 0; i < 10000; i++)
            for (size_t t = 0; t < n_tasks; t++)
                empty_kernel_0(t);

        tools::Rt_stats raw_stats;
        const auto t0_clk = std::chrono::steady_clock::now();
        const uint64_t t0_tsc = tools::read_tsc();
        for (size_t i = 0; i < n_exec; i++)
        {
            for (size_t t = 0; t < n_tasks; t++)
                empty_kernel_0(t);
        }
        const uint64_t t1_tsc = tools::read_tsc();
        const auto t1_clk = std::chrono::steady_clock::now();
        const double dt_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_clk - t0_clk).count();
        const double dt_ticks = (double)(t1_tsc - t0_tsc);
        const double ns_per_tick = (dt_ticks > 0.0) ? (dt_ns / dt_ticks) : 1.0;

        for (size_t i = 0; i < n_exec; i++)
        {
            const uint64_t s0 = tools::read_tsc();
            for (size_t t = 0; t < n_tasks; t++)
                empty_kernel_0(t);
            const uint64_t s1 = tools::read_tsc();
            raw_stats.add_sample(s1 - s0);
        }

        Bench_result r;
        r.name = "Bare C++ noinline call (no StreamPU)";
        r.n_tasks = n_tasks;
        r.ns_per_exec = dt_ns / (double)n_exec;
        r.ns_per_task = r.ns_per_exec / (double)n_tasks;
        r.ticks_per_task = (dt_ticks / (double)n_exec) / (double)n_tasks;
        r.std_ns_per_task = raw_stats.get_std_ticks() * (ns_per_tick / n_tasks);
        r.p50_ns_per_task = raw_stats.get_percentile_ticks(50.0) * (ns_per_tick / n_tasks);
        r.p99_ns_per_task = raw_stats.get_percentile_ticks(99.0) * (ns_per_tick / n_tasks);
        r.p999_ns_per_task = raw_stats.get_percentile_ticks(99.9) * (ns_per_tick / n_tasks);
        r.p9999_ns_per_task = raw_stats.get_percentile_ticks(99.99) * (ns_per_tick / n_tasks);
        print_row(r);
    }

    // 1a. 8-task SFWD chain (1x SOUT -> 6x SFWD -> 1x SIN) in no_copy_mode
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(8);
        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless());
            mods[i]->set_custom_name("EmptyFwd" + std::to_string(i));
            if (i == 0)
            {
                auto& t = mods[i]->create_task("exec");
                auto s_out = mods[i]->create_socket_out<uint8_t>(t, "out", data_length);
                mods[i]->create_codelet(t,
                                        [s_out](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[s_out].get_dataptr(), fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
            else if (i < 7)
            {
                auto& t = mods[i]->create_task("exec");
                auto s_fwd = mods[i]->create_socket_fwd<uint8_t>(t, "fwd", data_length);
                mods[i]->create_codelet(t,
                                        [s_fwd](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[s_fwd].get_dataptr(), fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
            else
            {
                auto& t = mods[i]->create_task("exec");
                auto s_in = mods[i]->create_socket_in<uint8_t>(t, "in", data_length);
                mods[i]->create_codelet(t,
                                        [s_in](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[s_in].get_dataptr(), fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
        }
        (*mods[1])["exec::fwd"] = (*mods[0])["exec::out"];
        for (size_t i = 1; i < 6; i++)
            (*mods[i + 1])["exec::fwd"] = (*mods[i])["exec::fwd"];
        (*mods[7])["exec::in"] = (*mods[6])["exec::fwd"];

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_no_copy_mode(true);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        print_row(run_sequence_bench("SFWD chain (1x SOUT, 6x SFWD, 1x SIN)", seq, n_exec));
        if (show_full_rt_tables)
        {
            seq.reset_rt_stats();
            seq.set_rt_stats(true);
            size_t c = 0;
            seq.exec([&c, n_exec]() { return ++c >= n_exec; });
            tools::Stats::show_rt(seq, true);
        }
    }

    // 1b. 8-task SIN+SOUT chain (1x SOUT -> 6x [SIN+SOUT] -> 1x SIN) in no_copy_mode vs copy_mode
    for (bool no_copy : { true, false })
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(8);
        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless());
            if (i == 0)
            {
                auto& t = mods[i]->create_task("exec");
                auto s_out = mods[i]->create_socket_out<uint8_t>(t, "out", data_length);
                mods[i]->create_codelet(t,
                                        [s_out](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[s_out].get_dataptr(), fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
            else if (i < 7)
            {
                auto& t = mods[i]->create_task("exec");
                auto s_in = mods[i]->create_socket_in<uint8_t>(t, "in", data_length);
                auto s_out = mods[i]->create_socket_out<uint8_t>(t, "out", data_length);
                mods[i]->create_codelet(
                  t,
                  [s_in, s_out](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                  {
                      empty_kernel_2(tsk[s_in].get_dataptr(), tsk[s_out].get_dataptr(), fid);
                      return runtime::status_t::SUCCESS;
                  });
            }
            else
            {
                auto& t = mods[i]->create_task("exec");
                auto s_in = mods[i]->create_socket_in<uint8_t>(t, "in", data_length);
                mods[i]->create_codelet(t,
                                        [s_in](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[s_in].get_dataptr(), fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
        }
        (*mods[1])["exec::in"] = (*mods[0])["exec::out"];
        for (size_t i = 1; i < 6; i++)
            (*mods[i + 1])["exec::in"] = (*mods[i])["exec::out"];
        (*mods[7])["exec::in"] = (*mods[6])["exec::out"];

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_no_copy_mode(no_copy);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        print_row(run_sequence_bench(
          std::string("SIN+SOUT chain (") + (no_copy ? "no_copy_mode=true)" : "no_copy_mode=false)"), seq, n_exec));
    }

    // 1c. Task-to-task binding (0 data sockets on intermediate tasks, only status socket)
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(8);
        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless());
            auto& t = mods[i]->create_task("exec");
            mods[i]->create_codelet(t,
                                    [](module::Module& /*m*/, runtime::Task& /*tsk*/, const size_t fid) -> int
                                    {
                                        empty_kernel_0(fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        for (size_t i = 0; i < 7; i++)
            (*mods[i + 1])("exec") = (*mods[i])("exec");

        runtime::Sequence seq((*mods[0])("exec"), 1);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        print_row(run_sequence_bench("0-socket task-bound chain (status only)", seq, n_exec));
    }

    // =================================================================================================================
    // 2. Number of Sockets per Task (1, 2, 4, 8, 16 sockets)
    // =================================================================================================================
    print_table_header("2. NUMBER OF SOCKETS PER TASK (8 tasks, SFWD in no_copy_mode)");
    for (size_t n_scks : { 1, 2, 4, 8, 16 })
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(8);
        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless());
            auto& t = mods[i]->create_task("exec");
            if (i == 0)
            {
                for (size_t s = 0; s < n_scks; s++)
                    mods[i]->create_socket_out<uint8_t>(t, "out" + std::to_string(s), data_length);
            }
            else if (i < 7)
            {
                for (size_t s = 0; s < n_scks; s++)
                    mods[i]->create_socket_fwd<uint8_t>(t, "fwd" + std::to_string(s), data_length);
            }
            else
            {
                for (size_t s = 0; s < n_scks; s++)
                    mods[i]->create_socket_in<uint8_t>(t, "in" + std::to_string(s), data_length);
            }
            mods[i]->create_codelet(t,
                                    [n_scks](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                    {
                                        const void* ptrs[16];
                                        for (size_t s = 0; s < n_scks; s++)
                                            ptrs[s] = tsk[s].get_dataptr();
                                        empty_kernel_n(ptrs, n_scks, fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        for (size_t s = 0; s < n_scks; s++)
        {
            (*mods[1])["exec::fwd" + std::to_string(s)] = (*mods[0])["exec::out" + std::to_string(s)];
            for (size_t i = 1; i < 6; i++)
                (*mods[i + 1])["exec::fwd" + std::to_string(s)] = (*mods[i])["exec::fwd" + std::to_string(s)];
            (*mods[7])["exec::in" + std::to_string(s)] = (*mods[6])["exec::fwd" + std::to_string(s)];
        }

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_no_copy_mode(true);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        print_row(run_sequence_bench(std::to_string(n_scks) + " socket(s) per task", seq, n_exec));
    }

    // =================================================================================================================
    // 3. Codelet Lambda Captures (0B, 8B, 16B, 64B trivial -> fast_codelet vs non-trivial std::function fallback)
    // =================================================================================================================
    print_table_header("3. LAMBDA CAPTURE SIZE & TYPE (fast_codelet_fn vs std::function)");
    for (int mode = 0; mode < 5; mode++)
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(8);
        std::string label;
        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless());
            auto& t = mods[i]->create_task("exec");
            if (i == 0)
                mods[i]->create_socket_out<uint8_t>(t, "s", data_length);
            else if (i < 7)
                mods[i]->create_socket_fwd<uint8_t>(t, "s", data_length);
            else
                mods[i]->create_socket_in<uint8_t>(t, "s", data_length);

            if (mode == 0)
            {
                label = "0-byte capture [] (fast_codelet)";
                mods[i]->create_codelet(t,
                                        [](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[0].get_dataptr(), fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
            else if (mode == 1)
            {
                label = "8-byte capture [ptr] (fast_codelet)";
                void* dummy_ptr = mods[i].get();
                mods[i]->create_codelet(t,
                                        [dummy_ptr](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_2(tsk[0].get_dataptr(), dummy_ptr, fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
            else if (mode == 2)
            {
                label = "16-byte capture [2x size_t] (fast)";
                size_t s0 = 0, s1 = 1;
                mods[i]->create_codelet(t,
                                        [s0, s1](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[s0].get_dataptr(), fid + s1);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
            else if (mode == 3)
            {
                label = "64-byte capture [8x size_t] (fast max)";
                struct Cap64
                {
                    size_t a[8];
                } cap = { { 0, 1, 2, 3, 4, 5, 6, 7 } };
                mods[i]->create_codelet(t,
                                        [cap](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[cap.a[0]].get_dataptr(), fid + cap.a[7]);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
            else
            {
                label = "Non-trivial capture (std::function)";
                std::shared_ptr<size_t> sp(new size_t(0));
                mods[i]->create_codelet(t,
                                        [sp](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                        {
                                            empty_kernel_1(tsk[*sp].get_dataptr(), fid);
                                            return runtime::status_t::SUCCESS;
                                        });
            }
        }
        for (size_t i = 0; i < 7; i++)
            (*mods[i + 1])["exec::s"] = (*mods[i])["exec::s"];

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_no_copy_mode(true);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        print_row(run_sequence_bench(label, seq, n_exec));
    }

    // =================================================================================================================
    // 4. Task & Sequence Execution Modes (fast vs slow, statuses vs without_statuses, stats vs rt_stats)
    // =================================================================================================================
    print_table_header("4. TASK & SEQUENCE EXECUTION MODES (8 empty tasks)");
    for (int mode = 0; mode < 4; mode++)
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(8);
        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless());
            auto& t = mods[i]->create_task("exec");
            if (i == 0)
                mods[i]->create_socket_out<uint8_t>(t, "s", data_length);
            else if (i < 7)
                mods[i]->create_socket_fwd<uint8_t>(t, "s", data_length);
            else
                mods[i]->create_socket_in<uint8_t>(t, "s", data_length);

            mods[i]->create_codelet(t,
                                    [](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                    {
                                        empty_kernel_1(tsk[0].get_dataptr(), fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        for (size_t i = 0; i < 7; i++)
            (*mods[i + 1])["exec::s"] = (*mods[i])["exec::s"];

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_no_copy_mode(true);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
            {
                t->set_fast(mode != 2);
                t->set_stats(mode == 3);
            }

        if (mode == 0)
            print_row(run_sequence_bench("set_fast(true), _exec_without_statuses", seq, n_exec, false));
        else if (mode == 1)
            print_row(run_sequence_bench("set_fast(true), _exec (with statuses)", seq, n_exec, true));
        else if (mode == 2)
            print_row(run_sequence_bench("set_fast(false) (full checks & fill)", seq, n_exec, false));
        else if (mode == 3)
            print_row(run_sequence_bench("set_stats(true) (per-task steady_clock)", seq, n_exec / 5, false));
    }

    // =================================================================================================================
    // 5. Sequence Chain Length (2, 4, 8, 16, 32, 64, 128 tasks)
    // =================================================================================================================
    print_table_header("5. SEQUENCE CHAIN LENGTH SCALING (amortization of Sequence loop)");
    for (size_t chain_len : { 2, 4, 8, 16, 32, 64, 128 })
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(chain_len);
        for (size_t i = 0; i < chain_len; i++)
        {
            mods[i].reset(new module::Stateless());
            auto& t = mods[i]->create_task("exec");
            if (i == 0)
                mods[i]->create_socket_out<uint8_t>(t, "s", data_length);
            else if (i + 1 < chain_len)
                mods[i]->create_socket_fwd<uint8_t>(t, "s", data_length);
            else
                mods[i]->create_socket_in<uint8_t>(t, "s", data_length);

            mods[i]->create_codelet(t,
                                    [](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                    {
                                        empty_kernel_1(tsk[0].get_dataptr(), fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        for (size_t i = 0; i + 1 < chain_len; i++)
            (*mods[i + 1])["exec::s"] = (*mods[i])["exec::s"];

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_no_copy_mode(true);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        const size_t scaled_exec = std::max<size_t>(10000, (n_exec * 8) / chain_len);
        print_row(run_sequence_bench("Chain length = " + std::to_string(chain_len) + " tasks", seq, scaled_exec));
    }

    // =================================================================================================================
    // 6. Multi-Wave Execution (n_frames = 1, 2, 4, 8 with n_frames_per_wave = 1 -> n_waves = 1, 2, 4, 8)
    // =================================================================================================================
    print_table_header("6. MULTI-WAVE EXECUTION (n_waves = 1, 2, 4, 8; normalized per task*wave)");
    for (size_t n_waves : { 1, 2, 4, 8 })
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(8);
        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless());
            auto& t = mods[i]->create_task("exec");
            if (i == 0)
                mods[i]->create_socket_out<uint8_t>(t, "s", data_length);
            else if (i < 7)
                mods[i]->create_socket_fwd<uint8_t>(t, "s", data_length);
            else
                mods[i]->create_socket_in<uint8_t>(t, "s", data_length);

            mods[i]->create_codelet(t,
                                    [](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                                    {
                                        empty_kernel_1(tsk[0].get_dataptr(), fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        for (size_t i = 0; i < 7; i++)
            (*mods[i + 1])["exec::s"] = (*mods[i])["exec::s"];

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_n_frames(n_waves);
        seq.set_no_copy_mode(true);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        const size_t scaled_exec = std::max<size_t>(10000, n_exec / n_waves);
        print_row(run_sequence_bench(
          "Multi-wave: n_waves = " + std::to_string(n_waves) + " (per wave)", seq, scaled_exec, false, 8 * n_waves));
    }

    // =================================================================================================================
    // 7. Control Flow (Switcher commute + select)
    // =================================================================================================================
    print_table_header("7. CONTROL FLOW (Switcher commute + select)");
    for (bool no_copy : { true, false })
    {
        module::Stateless src, branch0, branch1, snk;
        auto& t_src = src.create_task("exec");
        auto s_src_data = src.create_socket_out<uint8_t>(t_src, "data", data_length);
        auto s_src_ctrl = src.create_socket_out<int8_t>(t_src, "ctrl", 1);
        src.create_codelet(t_src,
                           [s_src_data, s_src_ctrl](module::Module&, runtime::Task& tsk, const size_t fid) -> int
                           {
                               int8_t* ctrl = static_cast<int8_t*>(tsk[s_src_ctrl].get_dataptr());
                               ctrl[0] = 0;
                               empty_kernel_2(tsk[s_src_data].get_dataptr(), ctrl, fid);
                               return runtime::status_t::SUCCESS;
                           });

        module::Switcher switcher(2, data_length, typeid(uint8_t));

        auto& t_b0 = branch0.create_task("exec");
        auto s_b0_in = branch0.create_socket_in<uint8_t>(t_b0, "in", data_length);
        auto s_b0_out = branch0.create_socket_out<uint8_t>(t_b0, "out", data_length);
        branch0.create_codelet(t_b0,
                               [s_b0_in, s_b0_out](module::Module&, runtime::Task& tsk, const size_t fid) -> int
                               {
                                   empty_kernel_2(tsk[s_b0_in].get_dataptr(), tsk[s_b0_out].get_dataptr(), fid);
                                   return runtime::status_t::SUCCESS;
                               });

        auto& t_b1 = branch1.create_task("exec");
        auto s_b1_in = branch1.create_socket_in<uint8_t>(t_b1, "in", data_length);
        auto s_b1_out = branch1.create_socket_out<uint8_t>(t_b1, "out", data_length);
        branch1.create_codelet(t_b1,
                               [s_b1_in, s_b1_out](module::Module&, runtime::Task& tsk, const size_t fid) -> int
                               {
                                   empty_kernel_2(tsk[s_b1_in].get_dataptr(), tsk[s_b1_out].get_dataptr(), fid);
                                   return runtime::status_t::SUCCESS;
                               });

        auto& t_snk = snk.create_task("exec");
        auto s_snk_in = snk.create_socket_in<uint8_t>(t_snk, "in", data_length);
        snk.create_codelet(t_snk,
                           [s_snk_in](module::Module&, runtime::Task& tsk, const size_t fid) -> int
                           {
                               empty_kernel_1(tsk[s_snk_in].get_dataptr(), fid);
                               return runtime::status_t::SUCCESS;
                           });

        switcher["commute::in_data"] = src["exec::data"];
        switcher["commute::in_ctrl"] = src["exec::ctrl"];
        branch0["exec::in"] = switcher["commute::out_data0"];
        branch1["exec::in"] = switcher["commute::out_data1"];
        switcher["select::in_data0"] = branch0["exec::out"];
        switcher["select::in_data1"] = branch1["exec::out"];
        snk["exec::in"] = switcher["select::out_data"];

        runtime::Sequence seq(src("exec"), 1);
        seq.set_no_copy_mode(no_copy);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        // 5 tasks executed along path 0: src, commute, branch0, select, snk
        print_row(
          run_sequence_bench(std::string("Switcher commute+select (") + (no_copy ? "no_copy=true)" : "no_copy=false)"),
                             seq,
                             n_exec,
                             false,
                             5));
    }

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << rang::fg::green << "Overhead breakdown completed!" << rang::style::reset
              << std::endl;
    return 0;
}
