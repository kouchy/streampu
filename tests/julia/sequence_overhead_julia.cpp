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
                   const size_t task_divisor_override = 0)
{
    const size_t n_tasks = task_divisor_override ? task_divisor_override : seq.get_tasks_per_types().size();

    // Julia JIT compilation warmup + cache warmup
    {
        size_t w = 0;
        const size_t n_warm = std::min<size_t>(n_exec / 10 + 100, 10000);
        seq.set_rt_stats(false);
        seq.exec([&w, n_warm]() { return ++w >= n_warm; });
    }

    // 1. Uninstrumented run
    size_t c1 = 0;
    const auto t0_clk = std::chrono::steady_clock::now();
    const uint64_t t0_tsc = tools::read_tsc();
    seq.exec([&c1, n_exec]() { return ++c1 >= n_exec; });
    const uint64_t t1_tsc = tools::read_tsc();
    const auto t1_clk = std::chrono::steady_clock::now();

    const double dt_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_clk - t0_clk).count();
    const double dt_ticks = (double)(t1_tsc - t0_tsc);
    const double ns_per_exec = dt_ns / (double)n_exec;
    const double ns_per_task = ns_per_exec / (double)n_tasks;
    const double ticks_per_task = (dt_ticks / (double)n_exec) / (double)n_tasks;

    // 2. RT-stats run
    seq.reset_rt_stats();
    seq.set_rt_stats(true);
    size_t c2 = 0;
    seq.exec([&c2, n_exec]() { return ++c2 >= n_exec; });

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
    tools::Julia::init(256);

    option longopts[] = { { "n-exec", required_argument, NULL, 'e' },
                          { "data-length", required_argument, NULL, 'd' },
                          { "rt-stats", no_argument, NULL, 'r' },
                          { "rt-skip", required_argument, NULL, 'R' },
                          { "rt-frames", no_argument, NULL, 'F' },
                          { "help", no_argument, NULL, 'h' },
                          { NULL, 0, NULL, 0 } };

    size_t n_exec = 200000;
    size_t data_length = 64;
    size_t rt_skip = 0;
    bool rt_frames = false;
    bool show_full_rt_tables = false;

    while (1)
    {
        const int opt = getopt_long(argc, argv, "e:d:R:Frh", longopts, 0);
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
            case 'R':
                rt_skip = atoi(optarg);
                break;
            case 'F':
                rt_frames = true;
                break;
            case 'h':
                std::cout << "usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "  -e, --n-exec       Number of sequence executions per benchmark [" << n_exec << "]"
                          << std::endl;
                std::cout << "  -d, --data-length  Socket buffer length in bytes               [" << data_length << "]"
                          << std::endl;
                std::cout << "  -r, --rt-stats     Also print full tools::Stats::show_rt tables [false]" << std::endl;
                std::cout << "  -R, --rt-skip      Number of initial frames to skip in RT      [" << rt_skip << "]"
                          << std::endl;
                std::cout << "  -F, --rt-frames    Display RT throughput in frames/s           [false]" << std::endl;
                return 0;
            default:
                break;
        }
    }

    std::cout << "##############################################################" << std::endl;
    std::cout << "# StreamPU Julia Empty-Task Runtime Overhead Micro-Benchmark #" << std::endl;
    std::cout << "##############################################################" << std::endl;
    std::cout << "#   - n_exec      = " << n_exec << std::endl;
    std::cout << "#   - data_length = " << data_length << " bytes" << std::endl;

    // =================================================================================================================
    // 1. Stateless_Julia Empty Task: SFWD vs SIN+SOUT, unsafe (@cfunction) vs safe (jl_call)
    // =================================================================================================================
    print_table_header("1. JULIA CALL DISPATCH & SOCKET TYPES (8 empty Julia tasks)");
    for (bool jl_safe : { false, true })
    {
        for (bool use_fwd : { true, false })
        {
            std::vector<std::unique_ptr<module::Stateless_Julia>> mods(8);
            for (size_t i = 0; i < 8; i++)
            {
                mods[i].reset(new module::Stateless_Julia());
                mods[i]->set_custom_name("JlEmpty" + std::to_string(i));
                mods[i]->set_jl_safety(jl_safe);
                auto& t = mods[i]->create_task("exec");
                if (i == 0)
                {
                    mods[i]->create_socket_out<uint8_t>(t, "out", data_length);
                    mods[i]->create_codelet(t, R""""(
                        function exec(sck_out, rnt_frame_id, rnt_n_frames_per_wave)
                            return 0
                        end
                    )"""");
                }
                else if (i < 7)
                {
                    if (use_fwd)
                    {
                        mods[i]->create_socket_fwd<uint8_t>(t, "fwd", data_length);
                        mods[i]->create_codelet(t, R""""(
                            function exec(sck_fwd, rnt_frame_id, rnt_n_frames_per_wave)
                                return 0
                            end
                        )"""");
                    }
                    else
                    {
                        mods[i]->create_socket_in<uint8_t>(t, "in", data_length);
                        mods[i]->create_socket_out<uint8_t>(t, "out", data_length);
                        mods[i]->create_codelet(t, R""""(
                            function exec(sck_in, sck_out, rnt_frame_id, rnt_n_frames_per_wave)
                                return 0
                            end
                        )"""");
                    }
                }
                else
                {
                    mods[i]->create_socket_in<uint8_t>(t, "in", data_length);
                    mods[i]->create_codelet(t, R""""(
                        function exec(sck_in, rnt_frame_id, rnt_n_frames_per_wave)
                            return 0
                        end
                    )"""");
                }
            }
            if (use_fwd)
            {
                (*mods[1])["exec::fwd"] = (*mods[0])["exec::out"];
                for (size_t i = 1; i < 6; i++)
                    (*mods[i + 1])["exec::fwd"] = (*mods[i])["exec::fwd"];
                (*mods[7])["exec::in"] = (*mods[6])["exec::fwd"];
            }
            else
            {
                (*mods[1])["exec::in"] = (*mods[0])["exec::out"];
                for (size_t i = 1; i < 6; i++)
                    (*mods[i + 1])["exec::in"] = (*mods[i])["exec::out"];
                (*mods[7])["exec::in"] = (*mods[6])["exec::out"];
            }

            runtime::Sequence seq((*mods[0])("exec"), 1);
            seq.set_no_copy_mode(true);
            for (auto* m : seq.get_modules<module::Module>(false))
                for (auto& t : m->tasks)
                    t->set_fast(true);

            const size_t cur_exec = jl_safe ? std::min<size_t>(n_exec, 50000) : n_exec;
            const std::string label = std::string(jl_safe ? "Stateless_Julia (safe, " : "Stateless_Julia (unsafe, ") +
                                      (use_fwd ? "SFWD)" : "SIN+SOUT)");
            print_row(run_sequence_bench(label, seq, cur_exec));

            if (show_full_rt_tables && !jl_safe && use_fwd)
            {
                seq.reset_rt_stats();
                seq.set_rt_stats(true, rt_skip);
                size_t c = 0;
                seq.exec([&c, cur_exec]() { return ++c >= cur_exec; });
                tools::Stats::show_rt(seq, true, rt_frames);
            }
        }
    }

    // =================================================================================================================
    // 2. Number of Julia Sockets per Task (1, 2, 4, 8 sockets, unsafe @cfunction)
    // =================================================================================================================
    print_table_header("2. NUMBER OF JULIA SOCKETS PER TASK (8 tasks, unsafe @cfunction)");
    for (size_t n_scks : { 1, 2, 4, 8 })
    {
        std::vector<std::unique_ptr<module::Stateless_Julia>> mods(8);
        std::string arg_list;
        for (size_t s = 0; s < n_scks; s++)
            arg_list += "s" + std::to_string(s) + ", ";
        const std::string jl_code =
          "function exec(" + arg_list + "rnt_frame_id, rnt_n_frames_per_wave)\n    return 0\nend\n";

        for (size_t i = 0; i < 8; i++)
        {
            mods[i].reset(new module::Stateless_Julia());
            mods[i]->set_jl_safety(false);
            auto& t = mods[i]->create_task("exec");
            for (size_t s = 0; s < n_scks; s++)
            {
                if (i == 0)
                    mods[i]->create_socket_out<uint8_t>(t, "s" + std::to_string(s), data_length);
                else if (i < 7)
                    mods[i]->create_socket_fwd<uint8_t>(t, "s" + std::to_string(s), data_length);
                else
                    mods[i]->create_socket_in<uint8_t>(t, "s" + std::to_string(s), data_length);
            }
            mods[i]->create_codelet(t, jl_code);
        }
        for (size_t s = 0; s < n_scks; s++)
        {
            for (size_t i = 0; i < 7; i++)
                (*mods[i + 1])["exec::s" + std::to_string(s)] = (*mods[i])["exec::s" + std::to_string(s)];
        }

        runtime::Sequence seq((*mods[0])("exec"), 1);
        seq.set_no_copy_mode(true);
        for (auto* m : seq.get_modules<module::Module>(false))
            for (auto& t : m->tasks)
                t->set_fast(true);

        print_row(run_sequence_bench(std::to_string(n_scks) + " Julia socket(s) per task", seq, n_exec));
    }

    // =================================================================================================================
    // 3. Stateful_Julia States & Constants (0, 1, 2 states; 0, 1, 2, 4 constants)
    // =================================================================================================================
    print_table_header("3. STATEFUL_JULIA STATES & CONSTANTS (8 tasks, unsafe @cfunction)");
    for (size_t n_states : { 0, 1, 2 })
    {
        for (size_t n_consts : { 0, 2 })
        {
            std::vector<std::unique_ptr<module::Stateful_Julia>> mods(8);
            std::string arg_list;
            for (size_t st = 0; st < n_states; st++)
                arg_list += "stt_" + std::to_string(st) + ", ";
            arg_list += "s0, ";
            for (size_t c = 0; c < n_consts; c++)
                arg_list += "cst_" + std::to_string(c) + ", ";
            const std::string jl_code =
              "function exec(" + arg_list + "rnt_frame_id, rnt_n_frames_per_wave)\n    return 0\nend\n";

            for (size_t i = 0; i < 8; i++)
            {
                mods[i].reset(new module::Stateful_Julia());
                mods[i]->set_jl_safety(false);
                for (size_t st = 0; st < n_states; st++)
                {
                    mods[i]->create_state("EmptyBenchState(" + std::to_string(st) + ")", false, R""""(
                        mutable struct EmptyBenchState
                            v::Int64
                        end
                    )"""");
                }
                auto& t = mods[i]->create_task("exec");
                if (i == 0)
                    mods[i]->create_socket_out<uint8_t>(t, "s0", data_length);
                else if (i < 7)
                    mods[i]->create_socket_fwd<uint8_t>(t, "s0", data_length);
                else
                    mods[i]->create_socket_in<uint8_t>(t, "s0", data_length);
                for (size_t c = 0; c < n_consts; c++)
                    mods[i]->create_constant<uint32_t>(t, (uint32_t)c);
                mods[i]->create_codelet(t, jl_code);
            }
            for (size_t i = 0; i < 7; i++)
                (*mods[i + 1])["exec::s0"] = (*mods[i])["exec::s0"];

            runtime::Sequence seq((*mods[0])("exec"), 1);
            seq.set_no_copy_mode(true);
            for (auto* m : seq.get_modules<module::Module>(false))
                for (auto& t : m->tasks)
                    t->set_fast(true);

            print_row(run_sequence_bench("Stateful_Julia: " + std::to_string(n_states) + " state(s), " +
                                           std::to_string(n_consts) + " const(s)",
                                         seq,
                                         n_exec));
        }
    }

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << rang::fg::green << "Julia overhead breakdown completed!"
              << rang::style::reset << std::endl;
    return 0;
}
