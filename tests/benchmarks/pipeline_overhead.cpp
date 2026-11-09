#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
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
    std::string backend;
    size_t n_stages = 0;
    double stage_ns_per_task = -1.0;
    double stage_std_ns_per_task = 0.0;
    double stage_p50_ns_per_task = 0.0;
    double stage_p99_ns_per_task = 0.0;
    double stage_p999_ns_per_task = 0.0;
    double stage_p9999_ns_per_task = 0.0;
    double stage_max_ns_per_task = 0.0;
    double out_ns_per_stream = -1.0;
    double e2e_avg_ns = -1.0;
    double e2e_std_ns = -1.0;
    double e2e_p50_ns = -1.0;
    double e2e_p99_ns = -1.0;
    double e2e_p999_ns = -1.0;
    double e2e_p9999_ns = -1.0;
    double e2e_max_ns = -1.0;
};

struct Pipe_csv_row
{
    std::string section;
    std::string case_name;
    Pipe_bench_result res;
};

static std::string
fmt_ns(const double ns_val)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << std::setw(9) << ns_val << " ns";
    return oss.str();
}

static void
print_multi_header(const std::string& title, const std::vector<std::string>& cols)
{
    std::ostringstream col_hdr;
    std::ostringstream sep_line;
    for (size_t i = 0; i < cols.size(); i++)
    {
        if (i > 0)
        {
            col_hdr << " | ";
            sep_line << "-+-";
        }
        col_hdr << std::right << std::setw(12) << cols[i];
        sep_line << "------------";
    }
    const std::string col_hdr_str = col_hdr.str();
    const size_t total_w = 60 + 4 + col_hdr_str.size();

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << std::string(total_w, '=') << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << title << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold << std::string(total_w, '-') << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << std::left << std::setw(39) << "PIPELINE BENCHMARK CASE" << std::right
              << " | STGS | METRIC     || " << col_hdr_str << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << std::string(39, '-') << "-+------+------------++-" << sep_line.str()
              << rang::style::reset << std::endl;
}

static void
print_multi_row(const std::string& label, const size_t n_stages, const std::vector<Pipe_bench_result>& results)
{
    // Line 1: End-to-End latency per stream (E2E ns/stm)
    std::cout << "#  " << std::left << std::setw(39) << label.substr(0, 39) << std::right << " | " << std::setw(4)
              << n_stages << " | E2E ns/stm || ";
    for (size_t i = 0; i < results.size(); i++)
    {
        if (i > 0) std::cout << " | ";
        if (results[i].e2e_avg_ns < 0.0)
            std::cout << std::setw(12) << "N/A";
        else
            std::cout << std::setw(12) << fmt_ns(results[i].e2e_avg_ns);
    }
    std::cout << std::endl;

    // Line 2: Stage latency per task (STG ns/tsk)
    std::cout << "#  " << std::left << std::setw(39) << "" << std::right << " | " << std::setw(4) << n_stages
              << " | STG ns/tsk || ";
    for (size_t i = 0; i < results.size(); i++)
    {
        if (i > 0) std::cout << " | ";
        if (results[i].stage_ns_per_task < 0.0)
            std::cout << std::setw(12) << "N/A";
        else
            std::cout << std::setw(12) << fmt_ns(results[i].stage_ns_per_task);
    }
    std::cout << std::endl;
}

static void
print_single_rt_header(const std::string& title, const std::string& backend)
{
    const std::string hdr_line =
      "PIPELINE BENCHMARK CASE                 | STGS | METRIC     ||    AVG(ns) | OUT NS/STM ||    STD(ns) | "
      "   P50(ns) |    P99(ns) |  P99.9(ns) | P99.99(ns) |    MAX(ns)";
    const std::string sep_line =
      "----------------------------------------+------+------------++------------+------------++------------+-"
      "-----------+------------+------------+------------+-----------";
    const size_t total_w = 1 + hdr_line.size();

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << std::string(total_w, '=') << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << title << " [Backend: " << backend << "]" << rang::style::reset
              << std::endl;
    std::cout << "# " << rang::style::bold << std::string(total_w, '-') << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << hdr_line << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << sep_line << rang::style::reset << std::endl;
}

static void
print_single_rt_row(const std::string& label, const Pipe_bench_result& r)
{
    if (r.stage_ns_per_task < 0.0) return;

    // Line 1: End-to-End pipeline stream latency (E2E ns/stm)
    std::cout << "#  " << std::left << std::setw(39) << label.substr(0, 39) << std::right << " | " << std::setw(4)
              << r.n_stages << " | E2E ns/stm || " << std::fixed << std::setprecision(2) << std::setw(10)
              << r.e2e_avg_ns << " | " << std::setw(10) << r.out_ns_per_stream << " || " << std::setw(10)
              << r.e2e_std_ns << " | " << std::setw(10) << r.e2e_p50_ns << " | " << std::setw(10) << r.e2e_p99_ns
              << " | " << std::setw(10) << r.e2e_p999_ns << " | " << std::setw(10) << r.e2e_p9999_ns << " | "
              << std::setw(10) << r.e2e_max_ns << std::endl;

    // Line 2: Stage execution latency per task (STG ns/tsk)
    std::cout << "#  " << std::left << std::setw(39) << "" << std::right << " | " << std::setw(4) << r.n_stages
              << " | STG ns/tsk || " << std::fixed << std::setprecision(2) << std::setw(10) << r.stage_ns_per_task
              << " | " << std::setw(10) << r.out_ns_per_stream << " || " << std::setw(10) << r.stage_std_ns_per_task
              << " | " << std::setw(10) << r.stage_p50_ns_per_task << " | " << std::setw(10) << r.stage_p99_ns_per_task
              << " | " << std::setw(10) << r.stage_p999_ns_per_task << " | " << std::setw(10)
              << r.stage_p9999_ns_per_task << " | " << std::setw(10) << r.stage_max_ns_per_task << std::endl;
}

static void
print_section_header(const std::string& title, const std::vector<std::string>& backends)
{
    if (backends.size() == 1)
        print_single_rt_header(title, backends[0]);
    else
        print_multi_header(title, backends);
}

static void
print_section_row(const std::string& label, const size_t n_stages, const std::vector<Pipe_bench_result>& results)
{
    if (results.size() == 1)
        print_single_rt_row(label, results[0]);
    else
        print_multi_row(label, n_stages, results);
}

#ifdef SPU_JULIA
static std::string
get_julia_empty_fn_name(const size_t n_scks)
{
    static std::set<size_t> defined_scks;
    const std::string fn_name = "_spu_pipe_empty_jl_" + std::to_string(n_scks);
    if (defined_scks.find(n_scks) == defined_scks.end())
    {
        std::string args;
        for (size_t s = 0; s < n_scks; s++)
            args += "s" + std::to_string(s) + ", ";
        args += "rnt_frame_id, rnt_n_frames_per_wave";
        const std::string code = fn_name + "(" + args + ") = Int32(0)";
        tools::Julia::eval_string(code);
        defined_scks.insert(n_scks);
    }
    return fn_name;
}
#endif

static Pipe_bench_result
measure_pipeline(const std::string& name,
                 const std::string& backend,
                 runtime::Pipeline& pipe,
                 const size_t n_exec,
                 const size_t n_skip_frames = 0)
{
    for (auto* m : pipe.get_modules<module::Module>(false))
        for (auto& t : m->tasks)
            t->set_fast(true);

    // Warmup run (16 frames) to warm caches and trigger Julia JIT compilation on all worker threads
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
    double sum_stage_std_ns = 0.0;
    double sum_stage_p50_ns = 0.0;
    double sum_stage_p99_ns = 0.0;
    double sum_stage_p999_ns = 0.0;
    double sum_stage_p9999_ns = 0.0;
    double max_stage_ns = 0.0;

    for (const auto* stg : stages)
    {
        tools::Rt_stats stg_merged;
        for (const auto& s : stg->get_rt_stats_per_thread())
            stg_merged.merge(s);
        const size_t n_t = std::max<size_t>(1, stg->get_tasks_per_types().size());
        const double scale = stg->get_rt_ns_per_tick() / (double)n_t;
        sum_stage_ns_per_task += stg_merged.get_avg_ticks() * scale;
        sum_stage_std_ns += stg_merged.get_std_ticks() * scale;
        sum_stage_p50_ns += stg_merged.get_percentile_ticks(50.0) * scale;
        sum_stage_p99_ns += stg_merged.get_percentile_ticks(99.0) * scale;
        sum_stage_p999_ns += stg_merged.get_percentile_ticks(99.9) * scale;
        sum_stage_p9999_ns += stg_merged.get_percentile_ticks(99.99) * scale;
        const double stg_max = (stg_merged.n_calls > 0) ? ((double)stg_merged.max_ticks * scale) : 0.0;
        if (stg_max > max_stage_ns) max_stage_ns = stg_max;
    }
    const double n_stg_d = (double)std::max<size_t>(1, stages.size());

    tools::Rt_stats e2e_merged;
    for (const auto& s : stages.back()->get_pipeline_e2e_rt_stats_per_thread())
        e2e_merged.merge(s);

    tools::Rt_stats inter_merged;
    for (const auto& s : stages.back()->get_pipeline_inter_rt_stats_per_thread())
        inter_merged.merge(s);

    const double ns_per_tick = stages.back()->get_rt_ns_per_tick();
    const double last_threads = (double)std::max<size_t>(1, stages.back()->get_n_threads());

    Pipe_bench_result res;
    res.name = name;
    res.backend = backend;
    res.n_stages = stages.size();
    res.stage_ns_per_task = sum_stage_ns_per_task / n_stg_d;
    res.stage_std_ns_per_task = sum_stage_std_ns / n_stg_d;
    res.stage_p50_ns_per_task = sum_stage_p50_ns / n_stg_d;
    res.stage_p99_ns_per_task = sum_stage_p99_ns / n_stg_d;
    res.stage_p999_ns_per_task = sum_stage_p999_ns / n_stg_d;
    res.stage_p9999_ns_per_task = sum_stage_p9999_ns / n_stg_d;
    res.stage_max_ns_per_task = max_stage_ns;
    res.out_ns_per_stream = (inter_merged.get_avg_ticks() * ns_per_tick) / last_threads;
    res.e2e_avg_ns = e2e_merged.get_avg_ticks() * ns_per_tick;
    res.e2e_std_ns = e2e_merged.get_std_ticks() * ns_per_tick;
    res.e2e_p50_ns = e2e_merged.get_percentile_ticks(50.0) * ns_per_tick;
    res.e2e_p99_ns = e2e_merged.get_percentile_ticks(99.0) * ns_per_tick;
    res.e2e_p999_ns = e2e_merged.get_percentile_ticks(99.9) * ns_per_tick;
    res.e2e_p9999_ns = e2e_merged.get_percentile_ticks(99.99) * ns_per_tick;
    res.e2e_max_ns = (e2e_merged.n_calls > 0) ? ((double)e2e_merged.max_ticks * ns_per_tick) : 0.0;
    return res;
}

static size_t
choose_pipe_n_exec(const size_t base_n_exec, const std::string& backend, const bool active_wait)
{
    size_t cur = base_n_exec;
    if (!active_wait) cur = std::min<size_t>(cur, 25000);
    if (backend == "JL_SAFE") cur = std::min<size_t>(cur, 25000);
    return std::max<size_t>(16, cur);
}

static Pipe_bench_result
run_linear_pipeline_case(const std::string& name,
                         const std::string& backend,
                         const size_t n_stages,
                         const size_t tasks_per_stage,
                         const std::vector<size_t>& threads_per_stage,
                         const size_t buffer_size,
                         const bool active_wait,
                         const bool no_copy_mode,
                         const size_t data_length,
                         const size_t n_exec,
                         const size_t n_scks_per_task = 1,
                         const bool show_rt_table = false,
                         const size_t n_skip_frames = 0,
                         const bool rt_frames = false)
{
    const size_t total_tasks = n_stages * tasks_per_stage;
    std::vector<std::unique_ptr<module::Module>> mods(total_tasks);

    for (size_t i = 0; i < total_tasks; i++)
    {
#ifdef SPU_JULIA
        if (backend == "JL_UNSAFE" || backend == "JL_SAFE")
        {
            std::unique_ptr<module::Stateless_Julia> m(new module::Stateless_Julia());
            m->set_custom_name(backend + "_S" + std::to_string(i / tasks_per_stage) + "_" +
                               std::to_string(i % tasks_per_stage));
            m->set_jl_safety(backend == "JL_SAFE");
            auto& t = m->create_task("exec");
            for (size_t s = 0; s < n_scks_per_task; s++)
            {
                const std::string sname = "s" + std::to_string(s);
                if (i == 0)
                    m->create_socket_out<uint8_t>(t, sname, data_length);
                else if (i + 1 < total_tasks)
                    m->create_socket_fwd<uint8_t>(t, sname, data_length);
                else
                    m->create_socket_in<uint8_t>(t, sname, data_length);
            }
            m->create_codelet(t, "", get_julia_empty_fn_name(n_scks_per_task));
            mods[i].reset(m.release());
            continue;
        }
#endif
        std::unique_ptr<module::Stateless> m(new module::Stateless());
        m->set_custom_name("CPP_S" + std::to_string(i / tasks_per_stage) + "_" + std::to_string(i % tasks_per_stage));
        auto& t = m->create_task("exec");
        for (size_t s = 0; s < n_scks_per_task; s++)
        {
            const std::string sname = "s" + std::to_string(s);
            if (i == 0)
                m->create_socket_out<uint8_t>(t, sname, data_length);
            else if (i + 1 < total_tasks)
                m->create_socket_fwd<uint8_t>(t, sname, data_length);
            else
                m->create_socket_in<uint8_t>(t, sname, data_length);
        }
        if (n_scks_per_task == 1)
        {
            m->create_codelet(t,
                              [](module::Module& /*m*/, runtime::Task& tk, const size_t fid) -> int
                              {
                                  empty_kernel_1(tk[0].get_dataptr(), fid);
                                  return runtime::status_t::SUCCESS;
                              });
        }
        else
        {
            m->create_codelet(t,
                              [n_scks_per_task](module::Module& /*m*/, runtime::Task& tk, const size_t fid) -> int
                              {
                                  const void* ptrs[16];
                                  for (size_t s = 0; s < n_scks_per_task; s++)
                                      ptrs[s] = tk[s].get_dataptr();
                                  empty_kernel_n(ptrs, n_scks_per_task, fid);
                                  return runtime::status_t::SUCCESS;
                              });
        }
        mods[i].reset(m.release());
    }

    for (size_t s = 0; s < n_scks_per_task; s++)
    {
        const std::string sname = "exec::s" + std::to_string(s);
        for (size_t i = 0; i + 1 < total_tasks; i++)
            (*mods[i + 1])[sname] = (*mods[i])[sname];
    }

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

    const size_t cur_exec = choose_pipe_n_exec(n_exec, backend, active_wait);
    Pipe_bench_result res = measure_pipeline(name, backend, pipe, cur_exec, n_skip_frames);

    if (show_rt_table)
    {
        std::cout << "#" << std::endl;
        tools::Stats::show_rt(pipe, true, rt_frames);
    }

    pipe.unbind_adaptors();
    for (size_t s = 0; s < n_scks_per_task; s++)
    {
        const std::string sname = "exec::s" + std::to_string(s);
        for (size_t i = 0; i + 1 < total_tasks; i++)
            (*mods[i + 1])[sname].unbind((*mods[i])[sname]);
    }

    return res;
}

static std::vector<std::string>
parse_backends(const std::string& backends_str)
{
    std::vector<std::string> out;
    std::stringstream ss(backends_str);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }),
                   item.end());
        std::transform(item.begin(), item.end(), item.begin(), [](unsigned char c) { return std::toupper(c); });
        if (item == "JL") item = "JL_UNSAFE";
        if (item == "CPP") out.push_back(item);
#ifdef SPU_JULIA
        else if (item == "JL_UNSAFE" || item == "JL_SAFE")
            out.push_back(item);
#endif
    }
    return out;
}

int
main(int argc, char** argv)
{
    tools::Signal_handler::init();

#ifdef SPU_JULIA
    std::string backends_str = "CPP,JL_UNSAFE,JL_SAFE";
    const bool has_julia = true;
#else
    std::string backends_str = "CPP";
    const bool has_julia = false;
#endif

    size_t n_exec = 50000;
    size_t data_length = 64;
    size_t n_skip_frames = 0;
    bool rt_frames = false;
    bool show_full_rt_tables = false;
    std::string csv_path;

    const option longopts[] = { { "n-exec", required_argument, NULL, 'e' },
                                { "data-length", required_argument, NULL, 'd' },
                                { "backends", required_argument, NULL, 'b' },
                                { "csv", required_argument, NULL, 'C' },
                                { "rt-skip", required_argument, NULL, 'k' },
                                { "rt-frames", no_argument, NULL, 'F' },
                                { "rt-stats", no_argument, NULL, 'r' },
                                { "help", no_argument, NULL, 'h' },
                                { 0, 0, 0, 0 } };

    while (true)
    {
        const int opt = getopt_long(argc, argv, "e:d:b:C:k:R:Frh", longopts, 0);
        if (opt == -1) break;
        switch (opt)
        {
            case 'e':
                n_exec = std::atoi(optarg);
                break;
            case 'd':
                data_length = std::atoi(optarg);
                break;
            case 'b':
                backends_str = optarg;
                break;
            case 'C':
                csv_path = optarg;
                break;
            case 'k':
            case 'R':
                n_skip_frames = std::atoi(optarg);
                break;
            case 'F':
                rt_frames = true;
                break;
            case 'r':
                show_full_rt_tables = true;
                break;
            case 'h':
                std::cout << "usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "  -e, --n-exec       Base number of pipeline frames per benchmark    [" << n_exec << "]"
                          << std::endl;
                std::cout << "  -d, --data-length  Socket buffer length in bytes                   [" << data_length
                          << "]" << std::endl;
                std::cout << "  -b, --backends     Comma-separated list of backends                [" << backends_str
                          << "]" << std::endl;
                std::cout << "                     (Specify a single backend e.g. -b CPP or -b JL_UNSAFE" << std::endl;
                std::cout << "                      to display full RT percentiles + MAX for every case)" << std::endl;
                std::cout << "  -C, --csv          Optional path to export benchmark results (CSV) [none]" << std::endl;
                std::cout << "  -r, --rt-stats     Also print full tools::Stats::show_rt tables    [false]"
                          << std::endl;
                std::cout << "  -k, --rt-skip      Number of initial frames to skip in RT          [" << n_skip_frames
                          << "]" << std::endl;
                std::cout << "  -F, --rt-frames    Display RT throughput in frames/s               [false]"
                          << std::endl;
                return 0;
            default:
                break;
        }
    }

    const std::vector<std::string> backends = parse_backends(backends_str);
    if (backends.empty())
    {
        std::cerr << "# No valid backends selected from '" << backends_str << "'." << std::endl;
        return 1;
    }

#ifdef SPU_JULIA
    for (const auto& b : backends)
    {
        if (b == "JL_UNSAFE" || b == "JL_SAFE")
        {
            tools::Julia::init(256);
            break;
        }
    }
#endif

    std::string active_backends_joined;
    for (size_t i = 0; i < backends.size(); i++)
        active_backends_joined += (i > 0 ? ", " : "") + backends[i];

    std::cout << "#######################################################################" << std::endl;
    std::cout << "# StreamPU Empty-Task Pipeline Overhead Benchmark (pipeline_overhead) #" << std::endl;
    std::cout << "#######################################################################" << std::endl;
    std::cout << "#   - base n_exec   = " << n_exec << " (scaled for passive wait / JL_SAFE)" << std::endl;
    std::cout << "#   - data_length   = " << data_length << " bytes" << std::endl;
    std::cout << "#   - n_skip_frames = " << n_skip_frames << std::endl;
    std::cout << "#   - has_julia     = " << (has_julia ? "true" : "false") << std::endl;
    std::cout << "#   - backends      = " << active_backends_joined << std::endl;
    if (backends.size() == 1)
        std::cout << "#   - display_mode  = Single-Backend Full RT Percentiles (E2E ns/stm + STG ns/tsk)" << std::endl;
    else
        std::cout << "#   - display_mode  = Multi-Backend Average Comparison (use -b <BACKEND> for full RT percentiles)"
                  << std::endl;

    std::vector<Pipe_csv_row> csv_rows;

    // =================================================================================================================
    // SECTION 1: Waiting Strategy: Active Spin-Wait vs Passive Condition Variable (2 stages, 2 tasks/stage)
    // =================================================================================================================
    print_section_header("1. WAITING STRATEGY: ACTIVE SPIN-WAIT vs PASSIVE CONDVAR (2 stages)", backends);
    for (size_t bsize : { (size_t)1, (size_t)16 })
    {
        for (bool active : { true, false })
        {
            const std::string label =
              std::string(active ? "active_wait=true" : "active_wait=false") + ", buffer_size=" + std::to_string(bsize);
            std::vector<Pipe_bench_result> row_res;
            for (const auto& b : backends)
            {
                const std::string rt_label = std::string(active ? "active_wait=true" : "active_wait=false") +
                                             ", buf=" + std::to_string(bsize) + " (" + b + ")";
                auto r = run_linear_pipeline_case(
                  rt_label, b, 2, 2, { 1, 1 }, bsize, active, true, data_length, n_exec, 1, false, n_skip_frames);
                row_res.push_back(r);
                csv_rows.push_back({ "1_waiting_strategy", label, r });
            }
            print_section_row(label, 2, row_res);
        }
    }

    // =================================================================================================================
    // SECTION 2: Inter-Stage FIFO Buffer Size (buffer_size = 1, 2, 4, 8, 16, 64; 2 stages, active_wait=true)
    // =================================================================================================================
    print_section_header("2. INTER-STAGE FIFO BUFFER SIZE (2 stages, active_wait=true, no_copy=true)", backends);
    for (size_t bsize : { 1, 2, 4, 8, 16, 64 })
    {
        const std::string label = "buffer_size = " + std::to_string(bsize) + " slot(s)";
        std::vector<Pipe_bench_result> row_res;
        for (const auto& b : backends)
        {
            auto r = run_linear_pipeline_case(label + " (" + b + ")",
                                              b,
                                              2,
                                              2,
                                              { 1, 1 },
                                              bsize,
                                              true,
                                              true,
                                              data_length,
                                              n_exec,
                                              1,
                                              false,
                                              n_skip_frames);
            row_res.push_back(r);
            csv_rows.push_back({ "2_buffer_size", label, r });
        }
        print_section_row(label, 2, row_res);
    }

    // =================================================================================================================
    // SECTION 3: Number of Pipeline Stages (2, 3, 4, 8 stages in series; buffer_size = 1 and 16)
    // =================================================================================================================
    print_section_header("3. NUMBER OF PIPELINE STAGES IN SERIES (active_wait=true, no_copy=true)", backends);
    for (size_t bsize : { (size_t)1, (size_t)16 })
    {
        for (size_t n_stg : { 2, 3, 4, 8 })
        {
            const std::string label = std::to_string(n_stg) + " stages (buffer_size=" + std::to_string(bsize) + ")";
            std::vector<size_t> thrs(n_stg, 1);
            std::vector<Pipe_bench_result> row_res;
            for (const auto& b : backends)
            {
                const bool show_rt =
                  show_full_rt_tables && (n_stg == 4 && bsize == 1 && (b == "CPP" || b == "JL_UNSAFE"));
                auto r = run_linear_pipeline_case(label + " (" + b + ")",
                                                  b,
                                                  n_stg,
                                                  2,
                                                  thrs,
                                                  bsize,
                                                  true,
                                                  true,
                                                  data_length,
                                                  n_exec,
                                                  1,
                                                  show_rt,
                                                  n_skip_frames,
                                                  rt_frames);
                row_res.push_back(r);
                csv_rows.push_back({ "3_n_stages", label, r });
            }
            print_section_row(label, n_stg, row_res);
        }
    }

    // =================================================================================================================
    // SECTION 4: Data-Parallel Multi-Threading on Middle Stage (1->N->1 Adaptor_m_to_n contention)
    // =================================================================================================================
    print_section_header("4. DATA-PARALLEL MULTI-THREADING (3 stages: 1 -> N -> 1 threads, active_wait)", backends);
    for (size_t mid_t : { 1, 2, 4 })
    {
        const std::string label = "3 stages: 1 -> " + std::to_string(mid_t) + " -> 1 thread(s) (buf=16)";
        std::vector<Pipe_bench_result> row_res;
        for (const auto& b : backends)
        {
            const std::string rt_label = "3 stages: 1->" + std::to_string(mid_t) + "->1 threads (" + b + ")";
            auto r = run_linear_pipeline_case(
              rt_label, b, 3, 2, { 1, mid_t, 1 }, 16, true, true, data_length, n_exec, 1, false, n_skip_frames);
            row_res.push_back(r);
            csv_rows.push_back({ "4_multithread_stage", label, r });
        }
        print_section_row(label, 3, row_res);
    }

    // =================================================================================================================
    // SECTION 5: Copy Mode vs Zero-Copy Mode across Payload Sizes (64 B, 1 KB, 16 KB, 64 KB)
    // =================================================================================================================
    print_section_header("5. ZERO-COPY (swap_buffers) vs COPY (std::copy) ACROSS PAYLOAD SIZES", backends);
    for (size_t bytes : { (size_t)64, (size_t)1024, (size_t)16384, (size_t)65536 })
    {
        for (bool no_copy : { true, false })
        {
            const std::string label =
              std::string(no_copy ? "no_copy=true" : "no_copy=false") + ", payload=" + std::to_string(bytes) + " B";
            std::vector<Pipe_bench_result> row_res;
            for (const auto& b : backends)
            {
                auto r = run_linear_pipeline_case(
                  label + " (" + b + ")", b, 2, 2, { 1, 1 }, 4, true, no_copy, bytes, n_exec, 1, false, n_skip_frames);
                row_res.push_back(r);
                csv_rows.push_back({ "5_copy_vs_zerocopy", label, r });
            }
            print_section_row(label, 2, row_res);
        }
    }

    // =================================================================================================================
    // SECTION 6: Number of Sockets Crossing Adaptor_m_to_n Boundary (1, 2, 4, 8 sockets)
    // =================================================================================================================
    print_section_header("6. NUMBER OF SOCKETS CROSSING ADAPTOR BOUNDARY (2 stages, active_wait, buf=4)", backends);
    for (size_t n_scks : { 1, 2, 4, 8 })
    {
        const std::string label = std::to_string(n_scks) + " socket(s) across Adaptor_m_to_n";
        std::vector<Pipe_bench_result> row_res;
        for (const auto& b : backends)
        {
            auto r = run_linear_pipeline_case(label + " (" + b + ")",
                                              b,
                                              2,
                                              2,
                                              { 1, 1 },
                                              4,
                                              true,
                                              true,
                                              data_length,
                                              n_exec,
                                              n_scks,
                                              false,
                                              n_skip_frames);
            row_res.push_back(r);
            csv_rows.push_back({ "6_adaptor_sockets", label, r });
        }
        print_section_row(label, 2, row_res);
    }

    if (!csv_path.empty())
    {
        std::ofstream ofs(csv_path);
        ofs << "section,case,backend,n_stages,stage_ns_per_task,stage_std_ns_per_task,stage_p50_ns_per_task,"
               "stage_p99_ns_per_task,stage_p999_ns_per_task,stage_p9999_ns_per_task,stage_max_ns_per_task,"
               "out_ns_per_stream,e2e_avg_ns,e2e_std_ns,e2e_p50_ns,e2e_p99_ns,e2e_p999_ns,e2e_p9999_ns,e2e_max_ns\n";
        for (const auto& row : csv_rows)
        {
            ofs << row.section << ",\"" << row.case_name << "\"," << row.res.backend << "," << row.res.n_stages << ","
                << row.res.stage_ns_per_task << "," << row.res.stage_std_ns_per_task << ","
                << row.res.stage_p50_ns_per_task << "," << row.res.stage_p99_ns_per_task << ","
                << row.res.stage_p999_ns_per_task << "," << row.res.stage_p9999_ns_per_task << ","
                << row.res.stage_max_ns_per_task << "," << row.res.out_ns_per_stream << "," << row.res.e2e_avg_ns << ","
                << row.res.e2e_std_ns << "," << row.res.e2e_p50_ns << "," << row.res.e2e_p99_ns << ","
                << row.res.e2e_p999_ns << "," << row.res.e2e_p9999_ns << "," << row.res.e2e_max_ns << "\n";
        }
        std::cout << "# Wrote CSV results to " << csv_path << std::endl;
    }

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << rang::fg::green << "Empty-task pipeline overhead breakdown completed!"
              << rang::style::reset << std::endl;
    return 0;
}
