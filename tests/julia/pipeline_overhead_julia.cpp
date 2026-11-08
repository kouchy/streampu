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
              << " JULIA PIPELINE BENCHMARK CASE           | STGS || STG NS/TSK | OUT NS/FRM || E2E AVG(us)| E2E "
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
measure_pipeline(const std::string& name, runtime::Pipeline& pipe, const size_t n_exec, const size_t n_skip_frames)
{
    for (auto* m : pipe.get_modules<module::Module>(false))
        for (auto& t : m->tasks)
            t->set_fast(true);

    // Warmup run (16 frames) to trigger Julia JIT compilation on all worker threads
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
run_linear_jl_pipeline_case(const std::string& name,
                            const size_t n_stages,
                            const size_t tasks_per_stage,
                            const std::vector<size_t>& threads_per_stage,
                            const size_t buffer_size,
                            const bool active_wait,
                            const bool jl_safe,
                            const bool no_copy_mode,
                            const size_t data_length,
                            const size_t n_exec,
                            const size_t n_skip_frames,
                            const bool show_rt_table = false)
{
    const size_t total_tasks = n_stages * tasks_per_stage;
    std::vector<std::unique_ptr<module::Stateless_Julia>> mods(total_tasks);
    for (size_t i = 0; i < total_tasks; i++)
    {
        mods[i].reset(new module::Stateless_Julia());
        mods[i]->set_custom_name("JlPipe_S" + std::to_string(i / tasks_per_stage) + "_" +
                                 std::to_string(i % tasks_per_stage));
        mods[i]->set_jl_safety(jl_safe);
        auto& t = mods[i]->create_task("exec");
        if (i == 0)
            mods[i]->create_socket_out<uint8_t>(t, "s", data_length);
        else if (i + 1 < total_tasks)
            mods[i]->create_socket_fwd<uint8_t>(t, "s", data_length);
        else
            mods[i]->create_socket_in<uint8_t>(t, "s", data_length);

        mods[i]->create_codelet(t, R""""(
            function exec(s, rnt_frame_id, rnt_n_frames_per_wave)
                return 0
            end
        )"""");
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
    tools::Julia::init(256);

    size_t n_exec = 50000;
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
        const int opt = getopt_long(argc, argv, "e:d:k:rh", longopts, 0);
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
                n_skip_frames = std::atoi(optarg);
                break;
            case 'r':
                show_full_rt_tables = true;
                break;
            case 'h':
                std::cout << "usage: " << argv[0] << " [-e n_exec] [-d data_length] [-k rt_skip_frames] [-r]"
                          << std::endl;
                return 0;
            default:
                break;
        }
    }

    std::cout << "#######################################################################" << std::endl;
    std::cout << "# StreamPU Julia Empty-Task Pipeline Runtime Overhead Micro-Benchmark #" << std::endl;
    std::cout << "#######################################################################" << std::endl;
    std::cout << "#   - n_exec         = " << n_exec << std::endl;
    std::cout << "#   - data_length    = " << data_length << " bytes" << std::endl;
    std::cout << "#   - rt_skip_frames = " << n_skip_frames << std::endl;

    // =================================================================================================================
    // 1. Julia Call Mode & Waiting Strategy (unsafe @cfunction vs safe jl_call; active vs passive; buf=1 & 16)
    // =================================================================================================================
    print_table_header("1. JULIA CALL MODE & WAITING STRATEGY (2 stages, 2 empty Julia tasks/stage)");
    for (bool jl_safe : { false, true })
    {
        for (bool active : { true, false })
        {
            const size_t cur_exec = active ? n_exec : std::min<size_t>(n_exec, 25000);
            const std::string label = std::string(jl_safe ? "safe (jl_call), " : "unsafe (@cfunction), ") +
                                      (active ? "active, buf=1" : "passive, buf=1");
            print_row(run_linear_jl_pipeline_case(
              label, 2, 2, { 1, 1 }, 1, active, jl_safe, true, data_length, cur_exec, n_skip_frames, false));
        }
    }

    // =================================================================================================================
    // 2. Inter-Stage FIFO Buffer Size (buffer_size = 1, 2, 4, 8, 16, 64; unsafe @cfunction, active_wait=true)
    // =================================================================================================================
    print_table_header("2. INTER-STAGE FIFO BUFFER SIZE (2 stages, unsafe @cfunction, active_wait=true)");
    for (size_t bsize : { 1, 2, 4, 8, 16, 64 })
    {
        print_row(run_linear_jl_pipeline_case("buffer_size = " + std::to_string(bsize) + " slot(s)",
                                              2,
                                              2,
                                              { 1, 1 },
                                              bsize,
                                              true,
                                              false,
                                              true,
                                              data_length,
                                              n_exec,
                                              n_skip_frames,
                                              false));
    }

    // =================================================================================================================
    // 3. Number of Julia Pipeline Stages (2, 3, 4 stages in series; buffer_size = 1 and 16)
    // =================================================================================================================
    print_table_header("3. NUMBER OF JULIA PIPELINE STAGES IN SERIES (unsafe @cfunction, active_wait=true)");
    for (size_t bsize : { (size_t)1, (size_t)16 })
    {
        for (size_t n_stg : { 2, 3, 4 })
        {
            std::vector<size_t> thrs(n_stg, 1);
            print_row(
              run_linear_jl_pipeline_case(std::to_string(n_stg) + " stages (buffer_size=" + std::to_string(bsize) + ")",
                                          n_stg,
                                          2,
                                          thrs,
                                          bsize,
                                          true,
                                          false,
                                          true,
                                          data_length,
                                          n_exec,
                                          n_skip_frames,
                                          show_full_rt_tables && (n_stg == 4 && bsize == 1)));
        }
    }

    // =================================================================================================================
    // 4. Multi-Threaded Julia Stages (Thread_pool_Julia: 1 -> N -> 1 threads, unsafe @cfunction)
    // =================================================================================================================
    print_table_header("4. MULTI-THREADED JULIA STAGES (3 stages: 1 -> N -> 1 threads, active_wait)");
    for (size_t mid_t : { 1, 2, 4 })
    {
        print_row(run_linear_jl_pipeline_case("3 stages: 1 -> " + std::to_string(mid_t) + " -> 1 thread(s) (buf=16)",
                                              3,
                                              2,
                                              { 1, mid_t, 1 },
                                              16,
                                              true,
                                              false,
                                              true,
                                              data_length,
                                              n_exec,
                                              n_skip_frames,
                                              false));
    }

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << rang::fg::green << "Julia pipeline overhead breakdown completed!"
              << rang::style::reset << std::endl;
    return 0;
}
