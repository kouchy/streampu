#include <algorithm>
#include <atomic>
#include <chrono>
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

// =====================================================================================================================
// Non-inlinable empty kernels with compiler memory/register barriers so the compiler cannot optimize away the call
// or dead-strip the socket pointer loads, isolating pure StreamPU Pipeline & Adaptor_m_to_n runtime overhead.
// =====================================================================================================================
#if defined(_MSC_VER)
#define SPU_BENCH_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define SPU_BENCH_NOINLINE __attribute__((noinline))
#else
#define SPU_BENCH_NOINLINE
#endif

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

struct Pipe_bench_result
{
    std::string name;
    size_t n_stages;
    double stage_ns_per_task;
    double out_ns_per_frame;
    double e2e_avg_us;
    double e2e_std_us;
    double e2e_p50_us;
    double e2e_p99_us;
    double e2e_p999_us;
    double e2e_p9999_us;
};

static void
print_table_header(const std::string& title)
{
    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold
              << "====================================================================================================="
                 "============================="
              << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold << " " << title << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold
              << "-----------------------------------------------------------------------------------------------------"
                 "-----------------------------"
              << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold
              << " PIPELINE BENCHMARK CASE                 | STGS || STG NS/TSK | OUT NS/FRM || E2E AVG(us)| E2E "
                 "STD(us)| E2E P50(us)| E2E P99(us)|P99.9(us) |P99.99(us)"
              << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold
              << "-----------------------------------------|------||------------|------------||------------|-----------"
                 "-|------------|------------|----------|----------"
              << rang::style::reset << std::endl;
}

static void
print_row(const Pipe_bench_result& r)
{
    std::cout << "#  " << std::left << std::setw(39) << r.name.substr(0, 39) << std::right << " | " << std::setw(4)
              << r.n_stages << " || " << std::fixed << std::setprecision(2) << std::setw(10) << r.stage_ns_per_task
              << " | " << std::setw(10) << r.out_ns_per_frame << " || " << std::setw(10) << r.e2e_avg_us << " | "
              << std::setw(10) << r.e2e_std_us << " | " << std::setw(10) << r.e2e_p50_us << " | " << std::setw(10)
              << r.e2e_p99_us << " | " << std::setw(8) << r.e2e_p999_us << " | " << std::setw(8) << r.e2e_p9999_us
              << std::endl;
}

static Pipe_bench_result
measure_pipeline(const std::string& name, runtime::Pipeline& pipe, const size_t n_exec, const size_t n_skip_frames = 0)
{
    for (auto* m : pipe.get_modules<module::Module>(false))
        for (auto& t : m->tasks)
            t->set_fast(true);

    // Warmup run (16 frames)
    {
        std::atomic<size_t> w(0);
        pipe.set_rt_stats(false);
        pipe.exec([&w]() { return ++w >= 16; });
    }

    pipe.reset_rt_stats();
    pipe.set_rt_stats(true, n_skip_frames);
    std::atomic<size_t> cnt(0);
    pipe.exec([&cnt, n_exec]() { return ++cnt >= n_exec; });

    const auto stages = pipe.get_stages();
    double sum_stage_ns_per_task = 0.0;
    for (const auto* stg : stages)
    {
        tools::Rt_stats stg_merged;
        for (const auto& s : stg->get_rt_stats_per_thread())
            stg_merged.merge(s);
        const size_t n_t = std::max<size_t>(1, stg->get_tasks_per_types().size());
        sum_stage_ns_per_task += (stg_merged.get_avg_ticks() * stg->get_rt_ns_per_tick()) / (double)n_t;
    }
    const double avg_stage_ns_per_task = sum_stage_ns_per_task / (double)stages.size();

    tools::Rt_stats e2e_merged;
    for (const auto& s : stages.back()->get_pipeline_e2e_rt_stats_per_thread())
        e2e_merged.merge(s);

    tools::Rt_stats inter_merged;
    for (const auto& s : stages.back()->get_pipeline_inter_rt_stats_per_thread())
        inter_merged.merge(s);

    const double ns_per_tick = stages.back()->get_rt_ns_per_tick();
    const double us_per_tick = ns_per_tick * 1e-3;
    const double last_threads = (double)std::max<size_t>(1, stages.back()->get_n_threads());

    Pipe_bench_result res;
    res.name = name;
    res.n_stages = stages.size();
    res.stage_ns_per_task = avg_stage_ns_per_task;
    res.out_ns_per_frame = (inter_merged.get_avg_ticks() * ns_per_tick) / last_threads;
    res.e2e_avg_us = e2e_merged.get_avg_ticks() * us_per_tick;
    res.e2e_std_us = e2e_merged.get_std_ticks() * us_per_tick;
    res.e2e_p50_us = e2e_merged.get_percentile_ticks(50.0) * us_per_tick;
    res.e2e_p99_us = e2e_merged.get_percentile_ticks(99.0) * us_per_tick;
    res.e2e_p999_us = e2e_merged.get_percentile_ticks(99.9) * us_per_tick;
    res.e2e_p9999_us = e2e_merged.get_percentile_ticks(99.99) * us_per_tick;
    return res;
}

static Pipe_bench_result
run_linear_pipeline_case(const std::string& name,
                         const size_t n_stages,
                         const size_t tasks_per_stage,
                         const std::vector<size_t>& threads_per_stage,
                         const size_t buffer_size,
                         const bool active_wait,
                         const bool no_copy_mode,
                         const size_t data_length,
                         const size_t n_exec,
                         const bool show_rt_table = false,
                         const size_t n_skip_frames = 0)
{
    const size_t total_tasks = n_stages * tasks_per_stage;
    std::vector<std::unique_ptr<module::Stateless>> mods(total_tasks);
    for (size_t i = 0; i < total_tasks; i++)
    {
        mods[i].reset(new module::Stateless());
        mods[i]->set_custom_name("Empty_S" + std::to_string(i / tasks_per_stage) + "_" +
                                 std::to_string(i % tasks_per_stage));
        auto& t = mods[i]->create_task("exec");
        if (i == 0)
        {
            auto s_out = mods[i]->create_socket_out<uint8_t>(t, "s", data_length);
            mods[i]->create_codelet(t,
                                    [s_out](module::Module& /*m*/, runtime::Task& tk, const size_t fid) -> int
                                    {
                                        empty_kernel_1(tk[s_out].get_dataptr(), fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        else if (i + 1 < total_tasks)
        {
            auto s_fwd = mods[i]->create_socket_fwd<uint8_t>(t, "s", data_length);
            mods[i]->create_codelet(t,
                                    [s_fwd](module::Module& /*m*/, runtime::Task& tk, const size_t fid) -> int
                                    {
                                        empty_kernel_1(tk[s_fwd].get_dataptr(), fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        else
        {
            auto s_in = mods[i]->create_socket_in<uint8_t>(t, "s", data_length);
            mods[i]->create_codelet(t,
                                    [s_in](module::Module& /*m*/, runtime::Task& tk, const size_t fid) -> int
                                    {
                                        empty_kernel_1(tk[s_in].get_dataptr(), fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
    }

    for (size_t i = 0; i + 1 < total_tasks; i++)
        (*mods[i + 1])["exec::s"] = (*mods[i])["exec::s"];

    std::vector<std::pair<std::vector<runtime::Task*>, std::vector<runtime::Task*>>> stage_defs(n_stages);
    for (size_t s = 0; s < n_stages; s++)
    {
        runtime::Task* first_t = &(*mods[s * tasks_per_stage])("exec");
        runtime::Task* last_t = &(*mods[(s + 1) * tasks_per_stage - 1])("exec");
        stage_defs[s] = { { first_t }, { last_t } };
    }

    std::vector<size_t> buf_sizes(n_stages - 1, buffer_size);
    std::vector<bool> waits(n_stages - 1, active_wait);

    runtime::Pipeline pipe((*mods[0])("exec"), stage_defs, threads_per_stage, buf_sizes, waits);
    for (auto* stg : pipe.get_stages())
        stg->set_no_copy_mode(no_copy_mode);

    Pipe_bench_result res = measure_pipeline(name, pipe, n_exec, n_skip_frames);

    if (show_rt_table)
    {
        std::cout << "#" << std::endl;
        tools::Stats::show_rt(pipe, true);
    }

    pipe.unbind_adaptors();
    for (size_t i = 0; i + 1 < total_tasks; i++)
        (*mods[i + 1])["exec::s"].unbind((*mods[i])["exec::s"]);

    return res;
}

int
main(int argc, char** argv)
{
    tools::Signal_handler::init();

    size_t n_exec = 100000;
    size_t data_length = 64;
    size_t n_skip_frames = 0;
    bool show_full_rt_tables = false;

    const option longopts[] = {
        { "n-exec", required_argument, NULL, 'e' },  { "data-length", required_argument, NULL, 'd' },
        { "rt-skip", required_argument, NULL, 'k' }, { "rt-stats", no_argument, NULL, 'r' },
        { "help", no_argument, NULL, 'h' },          { 0, 0, 0, 0 }
    };

    while (true)
    {
        const int opt = getopt_long(argc, argv, "e:d:k:R:rh", longopts, 0);
        if (opt == -1) break;
        switch (opt)
        {
            case 'e':
                n_exec = std::atoi(optarg);
                break;
            case 'd':
                data_length = std::atoi(optarg);
                break;
            case 'k':
            case 'R':
                n_skip_frames = std::atoi(optarg);
                break;
            case 'r':
                show_full_rt_tables = true;
                break;
            case 'h':
                std::cout << "usage: " << argv[0] << " [-e n_exec] [-d data_length] [-k n_skip_frames] [-r]"
                          << std::endl;
                return 0;
            default:
                break;
        }
    }

    std::cout << "#################################################################" << std::endl;
    std::cout << "# StreamPU Empty-Task Pipeline Runtime Overhead Micro-Benchmark #" << std::endl;
    std::cout << "#################################################################" << std::endl;
    std::cout << "#   - n_exec        = " << n_exec << std::endl;
    std::cout << "#   - data_length   = " << data_length << " bytes" << std::endl;
    std::cout << "#   - n_skip_frames = " << n_skip_frames << std::endl;

    // =================================================================================================================
    // 1. Waiting Strategy: Active Spin-Wait vs Passive Condition Variable (2 stages, 2 tasks/stage)
    // =================================================================================================================
    print_table_header("1. WAITING STRATEGY: ACTIVE SPIN-WAIT vs PASSIVE MUTEX/CONDVAR (2 stages)");
    for (size_t bsize : { (size_t)1, (size_t)16 })
    {
        for (bool active : { true, false })
        {
            const size_t cur_exec = active ? n_exec : std::min<size_t>(n_exec, 50000);
            const std::string label =
              std::string(active ? "active_wait=true" : "active_wait=false") + ", buffer_size=" + std::to_string(bsize);
            print_row(run_linear_pipeline_case(
              label, 2, 2, { 1, 1 }, bsize, active, true, data_length, cur_exec, false, n_skip_frames));
        }
    }

    // =================================================================================================================
    // 2. Inter-Stage FIFO Buffer Size (buffer_size = 1, 2, 4, 8, 16, 64; 2 stages, active_wait=true)
    // =================================================================================================================
    print_table_header("2. INTER-STAGE FIFO BUFFER SIZE (2 stages, active_wait=true, no_copy=true)");
    for (size_t bsize : { 1, 2, 4, 8, 16, 64 })
    {
        print_row(run_linear_pipeline_case("buffer_size = " + std::to_string(bsize) + " slot(s)",
                                           2,
                                           2,
                                           { 1, 1 },
                                           bsize,
                                           true,
                                           true,
                                           data_length,
                                           n_exec,
                                           false,
                                           n_skip_frames));
    }

    // =================================================================================================================
    // 3. Number of Pipeline Stages (2, 3, 4, 8 stages in series; buffer_size = 1 and 16)
    // =================================================================================================================
    print_table_header("3. NUMBER OF PIPELINE STAGES IN SERIES (active_wait=true, no_copy=true)");
    for (size_t bsize : { (size_t)1, (size_t)16 })
    {
        for (size_t n_stg : { 2, 3, 4, 8 })
        {
            std::vector<size_t> thrs(n_stg, 1);
            print_row(
              run_linear_pipeline_case(std::to_string(n_stg) + " stages (buffer_size=" + std::to_string(bsize) + ")",
                                       n_stg,
                                       2,
                                       thrs,
                                       bsize,
                                       true,
                                       true,
                                       data_length,
                                       n_exec,
                                       show_full_rt_tables && (n_stg == 4 && bsize == 1),
                                       n_skip_frames));
        }
    }

    // =================================================================================================================
    // 4. Data-Parallel Multi-Threading on Middle Stage (1->N->1 Adaptor_m_to_n contention)
    // =================================================================================================================
    print_table_header("4. DATA-PARALLEL MULTI-THREADING (3 stages: 1 -> N -> 1 threads, active_wait)");
    for (size_t mid_t : { 1, 2, 4 })
    {
        print_row(run_linear_pipeline_case("3 stages: 1 -> " + std::to_string(mid_t) + " -> 1 thread(s) (buf=16)",
                                           3,
                                           2,
                                           { 1, mid_t, 1 },
                                           16,
                                           true,
                                           true,
                                           data_length,
                                           n_exec,
                                           false,
                                           n_skip_frames));
    }

    // =================================================================================================================
    // 5. Copy Mode vs Zero-Copy Mode across Payload Sizes (64 B, 1 KB, 16 KB, 64 KB)
    // =================================================================================================================
    print_table_header("5. ZERO-COPY (swap_buffers) vs COPY (std::copy) ACROSS PAYLOAD SIZES (2 stages)");
    for (size_t bytes : { (size_t)64, (size_t)1024, (size_t)16384, (size_t)65536 })
    {
        for (bool no_copy : { true, false })
        {
            const std::string label =
              std::string(no_copy ? "no_copy=true" : "no_copy=false") + ", payload=" + std::to_string(bytes) + " B";
            print_row(
              run_linear_pipeline_case(label, 2, 2, { 1, 1 }, 4, true, no_copy, bytes, n_exec, false, n_skip_frames));
        }
    }

    // =================================================================================================================
    // 6. Number of Sockets Crossing Adaptor_m_to_n Boundary (1, 2, 4, 8 sockets)
    // =================================================================================================================
    print_table_header("6. NUMBER OF SOCKETS CROSSING ADAPTOR BOUNDARY (2 stages, active_wait, buf=4)");
    for (size_t n_scks : { 1, 2, 4, 8 })
    {
        std::vector<std::unique_ptr<module::Stateless>> mods(4);
        std::vector<std::vector<size_t>> sck_ids(4, std::vector<size_t>(n_scks));
        for (size_t i = 0; i < 4; i++)
        {
            mods[i].reset(new module::Stateless());
            auto& t = mods[i]->create_task("exec");
            for (size_t s = 0; s < n_scks; s++)
            {
                if (i == 0)
                    sck_ids[i][s] = mods[i]->create_socket_out<uint8_t>(t, "s" + std::to_string(s), data_length);
                else if (i < 3)
                    sck_ids[i][s] = mods[i]->create_socket_fwd<uint8_t>(t, "s" + std::to_string(s), data_length);
                else
                    sck_ids[i][s] = mods[i]->create_socket_in<uint8_t>(t, "s" + std::to_string(s), data_length);
            }
            const auto ids = sck_ids[i];
            mods[i]->create_codelet(t,
                                    [ids, n_scks](module::Module& /*m*/, runtime::Task& tk, const size_t fid) -> int
                                    {
                                        const void* ptrs[8];
                                        for (size_t s = 0; s < n_scks; s++)
                                            ptrs[s] = tk[ids[s]].get_dataptr();
                                        empty_kernel_n(ptrs, n_scks, fid);
                                        return runtime::status_t::SUCCESS;
                                    });
        }
        for (size_t s = 0; s < n_scks; s++)
            for (size_t i = 0; i < 3; i++)
                (*mods[i + 1])["exec::s" + std::to_string(s)] = (*mods[i])["exec::s" + std::to_string(s)];

        runtime::Pipeline pipe((*mods[0])("exec"),
                               {
                                 { { &(*mods[0])("exec") }, { &(*mods[1])("exec") } },
                                 { { &(*mods[2])("exec") }, { &(*mods[3])("exec") } },
                               },
                               { 1, 1 },
                               { 4 },
                               { true });
        for (auto* stg : pipe.get_stages())
            stg->set_no_copy_mode(true);

        print_row(
          measure_pipeline(std::to_string(n_scks) + " socket(s) across Adaptor_m_to_n", pipe, n_exec, n_skip_frames));

        pipe.unbind_adaptors();
        for (size_t s = 0; s < n_scks; s++)
            for (size_t i = 0; i < 3; i++)
                (*mods[i + 1])["exec::s" + std::to_string(s)].unbind((*mods[i])["exec::s" + std::to_string(s)]);
    }

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << rang::fg::green << "Pipeline overhead breakdown completed!"
              << rang::style::reset << std::endl;
    return 0;
}
