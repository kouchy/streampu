#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

struct Seq_bench_result
{
    std::string name;
    std::string backend;
    size_t n_tasks = 0;
    double ns_per_frame = -1.0;
    double ns_per_task = -1.0;
    double ticks_per_frame = 0.0;
    double ticks_per_task = 0.0;
    double std_ns_per_frame = 0.0;
    double p50_ns_per_frame = 0.0;
    double p99_ns_per_frame = 0.0;
    double p999_ns_per_frame = 0.0;
    double p9999_ns_per_frame = 0.0;
    double max_ns_per_frame = 0.0;
};

struct Csv_row
{
    std::string section;
    std::string case_name;
    Seq_bench_result res;
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
    const size_t total_w = 57 + 4 + col_hdr_str.size();

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << std::string(total_w, '=') << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << title << rang::style::reset << std::endl;
    std::cout << "# " << rang::style::bold << std::string(total_w, '-') << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << std::left << std::setw(39) << "BENCHMARK CASE" << std::right
              << " | TASKS | UNIT   || " << col_hdr_str << rang::style::reset << std::endl;
    std::cout << "#  " << rang::style::bold << std::string(39, '-') << "-+-------+--------++-" << sep_line.str()
              << rang::style::reset << std::endl;
}

static void
print_multi_row(const std::string& label, const size_t n_tasks, const std::vector<Seq_bench_result>& results)
{
    // Line 1: Per Stream (ns/stm)
    std::cout << "#  " << std::left << std::setw(39) << label.substr(0, 39) << std::right << " | " << std::setw(5)
              << n_tasks << " | ns/stm || ";
    for (size_t i = 0; i < results.size(); i++)
    {
        if (i > 0) std::cout << " | ";
        if (results[i].ns_per_frame < 0.0)
            std::cout << std::setw(12) << "N/A";
        else
            std::cout << std::setw(12) << fmt_ns(results[i].ns_per_frame);
    }
    std::cout << std::endl;

    // Line 2: Per Task (ns/tsk)
    std::cout << "#  " << std::left << std::setw(39) << "" << std::right << " | " << std::setw(5) << n_tasks
              << " | ns/tsk || ";
    for (size_t i = 0; i < results.size(); i++)
    {
        if (i > 0) std::cout << " | ";
        if (results[i].ns_per_task < 0.0)
            std::cout << std::setw(12) << "N/A";
        else
            std::cout << std::setw(12) << fmt_ns(results[i].ns_per_task);
    }
    std::cout << std::endl;
}

static void
print_single_rt_header(const std::string& title, const std::string& backend)
{
    const std::string hdr_line =
      "BENCHMARK CASE                          | TASKS | UNIT   ||   AVG(ns) |     TICKS ||   STD(ns) | "
      "  P50(ns) |   P99(ns) | P99.9(ns) |P99.99(ns) |   MAX(ns)";
    const std::string sep_line =
      "----------------------------------------+-------+--------++-----------+-----------++-----------+-"
      "----------+-----------+-----------+-----------+----------";
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
print_single_rt_row(const std::string& label, const Seq_bench_result& r)
{
    if (r.ns_per_task < 0.0) return;
    const double d = (double)std::max<size_t>(1, r.n_tasks);

    // Line 1: Per Stream (ns/stm)
    std::cout << "#  " << std::left << std::setw(39) << label.substr(0, 39) << std::right << " | " << std::setw(5)
              << r.n_tasks << " | ns/stm || " << std::fixed << std::setprecision(2) << std::setw(9) << r.ns_per_frame
              << " | " << std::setw(9) << r.ticks_per_frame << " || " << std::setw(9) << r.std_ns_per_frame << " | "
              << std::setw(9) << r.p50_ns_per_frame << " | " << std::setw(9) << r.p99_ns_per_frame << " | "
              << std::setw(9) << r.p999_ns_per_frame << " | " << std::setw(9) << r.p9999_ns_per_frame << " | "
              << std::setw(9) << r.max_ns_per_frame << std::endl;

    // Line 2: Per Task (ns/tsk)
    std::cout << "#  " << std::left << std::setw(39) << "" << std::right << " | " << std::setw(5) << r.n_tasks
              << " | ns/tsk || " << std::fixed << std::setprecision(2) << std::setw(9) << r.ns_per_task << " | "
              << std::setw(9) << r.ticks_per_task << " || " << std::setw(9) << (r.std_ns_per_frame / d) << " | "
              << std::setw(9) << (r.p50_ns_per_frame / d) << " | " << std::setw(9) << (r.p99_ns_per_frame / d) << " | "
              << std::setw(9) << (r.p999_ns_per_frame / d) << " | " << std::setw(9) << (r.p9999_ns_per_frame / d)
              << " | " << std::setw(9) << (r.max_ns_per_frame / d) << std::endl;
}

static void
print_header(const std::string& title, const std::vector<std::string>& backends)
{
    if (backends.size() == 1)
        print_single_rt_header(title, backends[0]);
    else
        print_multi_header(title, backends);
}

static void
print_row(const std::string& label, const size_t n_tasks, const std::vector<Seq_bench_result>& results)
{
    if (results.size() == 1)
        print_single_rt_row(label, results[0]);
    else
        print_multi_row(label, n_tasks, results);
}

#ifdef SPU_JULIA
static std::string
get_julia_empty_fn_name(const size_t n_scks)
{
    static std::set<size_t> defined_scks;
    const std::string fn_name = "_spu_bench_empty_jl_" + std::to_string(n_scks);
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

enum class Sck_dir : uint8_t
{
    IN,
    OUT,
    FWD
};

static std::unique_ptr<module::Module>
make_empty_module(const std::string& backend,
                  const std::vector<std::pair<Sck_dir, std::string>>& socket_spec,
                  const size_t data_length,
                  const size_t n_frames = 1,
                  const size_t n_frames_per_wave = 1,
                  const int capture_mode = 0)
{
    const size_t n_scks = socket_spec.size();

#ifdef SPU_JULIA
    if (backend == "JL_UNSAFE" || backend == "JL_SAFE")
    {
        std::unique_ptr<module::Stateless_Julia> mdl(new module::Stateless_Julia());
        mdl->set_n_frames(n_frames);
        mdl->set_n_frames_per_wave(n_frames_per_wave);
        mdl->set_jl_safety(backend == "JL_SAFE");
        auto& t = mdl->create_task("exec");
        t.set_replicability(true);
        for (size_t s = 0; s < n_scks; s++)
        {
            if (socket_spec[s].first == Sck_dir::IN)
                mdl->create_socket_in<uint8_t>(t, socket_spec[s].second, data_length);
            else if (socket_spec[s].first == Sck_dir::OUT)
                mdl->create_socket_out<uint8_t>(t, socket_spec[s].second, data_length);
            else
                mdl->create_socket_fwd<uint8_t>(t, socket_spec[s].second, data_length);
        }
        const std::string fn_name = get_julia_empty_fn_name(n_scks);
        mdl->create_codelet(t, "", fn_name);
        return std::unique_ptr<module::Module>(mdl.release());
    }
#endif

    std::unique_ptr<module::Stateless> mdl(new module::Stateless());
    mdl->set_n_frames(n_frames);
    mdl->set_n_frames_per_wave(n_frames_per_wave);
    auto& t = mdl->create_task("exec");
    t.set_replicability(true);
    for (size_t s = 0; s < n_scks; s++)
    {
        if (socket_spec[s].first == Sck_dir::IN)
            mdl->create_socket_in<uint8_t>(t, socket_spec[s].second, data_length);
        else if (socket_spec[s].first == Sck_dir::OUT)
            mdl->create_socket_out<uint8_t>(t, socket_spec[s].second, data_length);
        else
            mdl->create_socket_fwd<uint8_t>(t, socket_spec[s].second, data_length);
    }

    if (capture_mode == 1)
    {
        void* dummy_ptr = mdl.get();
        mdl->create_codelet(t,
                            [dummy_ptr](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                            {
                                empty_kernel_2(tsk[0].get_dataptr(), dummy_ptr, fid);
                                return runtime::status_t::SUCCESS;
                            });
    }
    else if (capture_mode == 2)
    {
        size_t s0 = 0, s1 = 1;
        mdl->create_codelet(t,
                            [s0, s1](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                            {
                                empty_kernel_1(tsk[s0].get_dataptr(), fid + s1);
                                return runtime::status_t::SUCCESS;
                            });
    }
    else if (capture_mode == 3)
    {
        struct Cap64
        {
            size_t a[8];
        } cap = { { 0, 1, 2, 3, 4, 5, 6, 7 } };
        mdl->create_codelet(t,
                            [cap](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                            {
                                empty_kernel_1(tsk[cap.a[0]].get_dataptr(), fid + cap.a[7]);
                                return runtime::status_t::SUCCESS;
                            });
    }
    else if (capture_mode == 4)
    {
        std::shared_ptr<size_t> sp(new size_t(0));
        mdl->create_codelet(t,
                            [sp](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                            {
                                empty_kernel_1(tsk[*sp].get_dataptr(), fid);
                                return runtime::status_t::SUCCESS;
                            });
    }
    else if (n_scks == 0)
    {
        mdl->create_codelet(t,
                            [](module::Module& /*m*/, runtime::Task& /*tsk*/, const size_t fid) -> int
                            {
                                empty_kernel_0(fid);
                                return runtime::status_t::SUCCESS;
                            });
    }
    else if (n_scks == 1)
    {
        mdl->create_codelet(t,
                            [](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                            {
                                empty_kernel_1(tsk[0].get_dataptr(), fid);
                                return runtime::status_t::SUCCESS;
                            });
    }
    else if (n_scks == 2)
    {
        mdl->create_codelet(t,
                            [](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                            {
                                empty_kernel_2(tsk[0].get_dataptr(), tsk[1].get_dataptr(), fid);
                                return runtime::status_t::SUCCESS;
                            });
    }
    else
    {
        mdl->create_codelet(t,
                            [n_scks](module::Module& /*m*/, runtime::Task& tsk, const size_t fid) -> int
                            {
                                const void* ptrs[16];
                                for (size_t s = 0; s < n_scks; s++)
                                    ptrs[s] = tsk[s].get_dataptr();
                                empty_kernel_n(ptrs, n_scks, fid);
                                return runtime::status_t::SUCCESS;
                            });
    }

    return std::unique_ptr<module::Module>(mdl.release());
}

struct Built_chain
{
    std::vector<std::unique_ptr<module::Module>> mods;
    std::unique_ptr<runtime::Sequence> seq;
};

static Built_chain
build_chain(const std::vector<std::string>& backends,
            const std::string& topology,
            const size_t data_length,
            const size_t n_scks_per_task = 1,
            const size_t n_threads = 1,
            const bool no_copy_mode = true,
            const bool fast_mode = true,
            const bool stats_mode = false,
            const size_t n_frames = 1,
            const size_t n_frames_per_wave = 1,
            const int capture_mode = 0)
{
    const size_t chain_len = backends.size();
    Built_chain out;
    out.mods.resize(chain_len);

    for (size_t i = 0; i < chain_len; i++)
    {
        std::vector<std::pair<Sck_dir, std::string>> spec;
        if (topology == "0SCK")
        {
            // No data sockets
        }
        else if (topology == "SFWD")
        {
            for (size_t s = 0; s < n_scks_per_task; s++)
            {
                const std::string sname = "s" + std::to_string(s);
                if (i == 0)
                    spec.push_back({ Sck_dir::OUT, sname });
                else if (i + 1 < chain_len)
                    spec.push_back({ Sck_dir::FWD, sname });
                else
                    spec.push_back({ Sck_dir::IN, sname });
            }
        }
        else if (topology == "SINOUT")
        {
            if (i == 0)
                spec.push_back({ Sck_dir::OUT, "s0" });
            else if (i + 1 < chain_len)
            {
                spec.push_back({ Sck_dir::IN, "s0" });
                spec.push_back({ Sck_dir::OUT, "s1" });
            }
            else
                spec.push_back({ Sck_dir::IN, "s0" });
        }

        out.mods[i] = make_empty_module(backends[i], spec, data_length, n_frames, n_frames_per_wave, capture_mode);
        out.mods[i]->set_custom_name(backends[i] + "_" + std::to_string(i));
    }

    for (size_t i = 0; i + 1 < chain_len; i++)
    {
        if (topology == "0SCK")
        {
            (*out.mods[i + 1])("exec") = (*out.mods[i])("exec");
        }
        else if (topology == "SFWD")
        {
            for (size_t s = 0; s < n_scks_per_task; s++)
            {
                const std::string sname = "exec::s" + std::to_string(s);
                (*out.mods[i + 1])[sname] = (*out.mods[i])[sname];
            }
        }
        else if (topology == "SINOUT")
        {
            const std::string out_sck = (i == 0) ? "exec::s0" : "exec::s1";
            (*out.mods[i + 1])["exec::s0"] = (*out.mods[i])[out_sck];
        }
    }

    out.seq.reset(new runtime::Sequence((*out.mods[0])("exec"), n_threads));
    out.seq->set_n_frames(n_frames);
    out.seq->set_no_copy_mode(no_copy_mode);
    for (auto* m : out.seq->get_modules<module::Module>(false))
    {
        for (auto& t : m->tasks)
        {
            t->set_fast(fast_mode);
            t->set_stats(stats_mode);
        }
    }
    return out;
}

static size_t
choose_n_exec(const size_t base_n_exec, const std::vector<std::string>& backends, const double scale = 1.0)
{
    bool has_jl_safe = false;
    for (const auto& b : backends)
        if (b == "JL_SAFE") has_jl_safe = true;

    const size_t raw = std::max<size_t>(1, (size_t)((double)base_n_exec * scale));
    if (base_n_exec < 5000) return raw;
    if (has_jl_safe) return std::max<size_t>(5000, std::min<size_t>(raw, 50000));
    return std::max<size_t>(10000, raw);
}

static Seq_bench_result
measure_sequence_bench(const std::string& name,
                       const std::string& backend,
                       runtime::Sequence& seq,
                       const size_t n_exec,
                       const bool collect_rt_percentiles,
                       const size_t n_warmup = 256,
                       const size_t rt_skip = 0,
                       const bool with_statuses = false,
                       const size_t task_divisor_override = 0,
                       const bool per_thread = false)
{
    const size_t n_tasks = task_divisor_override ? task_divisor_override : seq.get_tasks_per_types().size();
    const size_t n_threads = std::max<size_t>(1, seq.get_n_threads());
    seq.set_rt_stats(false);

    Seq_bench_result res;
    res.name = name;
    res.backend = backend;
    res.n_tasks = n_tasks;

    if (per_thread && n_threads > 1)
    {
        if (n_warmup > 0)
        {
            std::atomic<size_t> w_cnt(0);
            if (!with_statuses)
                seq.exec([&w_cnt, n_warmup]() { return ++w_cnt >= n_warmup; });
            else
                seq.exec([&w_cnt, n_warmup](const std::vector<const int*>&) { return ++w_cnt >= n_warmup; });
        }

        std::atomic<size_t> cnt(0);
        const auto t0_clk = std::chrono::steady_clock::now();
        const uint64_t t0_tsc = tools::read_tsc();
        if (!with_statuses)
            seq.exec([&cnt, n_exec]() { return ++cnt >= n_exec; });
        else
            seq.exec([&cnt, n_exec](const std::vector<const int*>&) { return ++cnt >= n_exec; });
        const uint64_t t1_tsc = tools::read_tsc();
        const auto t1_clk = std::chrono::steady_clock::now();

        const double dt_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_clk - t0_clk).count();
        const double dt_ticks = (double)(t1_tsc - t0_tsc);
        res.ns_per_frame = dt_ns / (double)n_exec;
        res.ns_per_task = res.ns_per_frame / (double)n_tasks;
        res.ticks_per_frame = dt_ticks / (double)n_exec;
        res.ticks_per_task = res.ticks_per_frame / (double)n_tasks;

        if (collect_rt_percentiles)
        {
            seq.reset_rt_stats();
            seq.set_rt_stats(true, rt_skip);
            std::atomic<size_t> c2(0);
            if (!with_statuses)
                seq.exec([&c2, n_exec]() { return ++c2 >= n_exec; });
            else
                seq.exec([&c2, n_exec](const std::vector<const int*>&) { return ++c2 >= n_exec; });

            tools::Rt_stats st;
            for (const auto& s : seq.get_rt_stats_per_thread())
                st.merge(s);
            const double ns_per_tick = seq.get_rt_ns_per_tick() / (double)n_threads;
            seq.set_rt_stats(false);

            res.std_ns_per_frame = st.get_std_ticks() * ns_per_tick;
            res.p50_ns_per_frame = st.get_percentile_ticks(50.0) * ns_per_tick;
            res.p99_ns_per_frame = st.get_percentile_ticks(99.0) * ns_per_tick;
            res.p999_ns_per_frame = st.get_percentile_ticks(99.9) * ns_per_tick;
            res.p9999_ns_per_frame = st.get_percentile_ticks(99.99) * ns_per_tick;
            res.max_ns_per_frame = (double)st.max_ticks * ns_per_tick;
        }
        return res;
    }

    if (n_warmup > 0)
    {
        size_t w = 0;
        if (!with_statuses)
            seq.exec([&w, n_warmup]() { return ++w >= n_warmup; });
        else
            seq.exec([&w, n_warmup](const std::vector<const int*>&) { return ++w >= n_warmup; });
    }

    size_t c1 = 0;
    const auto t0_clk = std::chrono::steady_clock::now();
    const uint64_t t0_tsc = tools::read_tsc();
    if (!with_statuses)
        seq.exec([&c1, n_exec]() { return ++c1 >= n_exec; });
    else
        seq.exec([&c1, n_exec](const std::vector<const int*>&) { return ++c1 >= n_exec; });
    const uint64_t t1_tsc = tools::read_tsc();
    const auto t1_clk = std::chrono::steady_clock::now();

    const double dt_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_clk - t0_clk).count();
    const double dt_ticks = (double)(t1_tsc - t0_tsc);
    res.ns_per_frame = dt_ns / (double)n_exec;
    res.ns_per_task = res.ns_per_frame / (double)n_tasks;
    res.ticks_per_frame = dt_ticks / (double)n_exec;
    res.ticks_per_task = res.ticks_per_frame / (double)n_tasks;

    if (collect_rt_percentiles)
    {
        seq.reset_rt_stats();
        seq.set_rt_stats(true, rt_skip);
        size_t c2 = 0;
        if (!with_statuses)
            seq.exec([&c2, n_exec]() { return ++c2 >= n_exec; });
        else
            seq.exec([&c2, n_exec](const std::vector<const int*>&) { return ++c2 >= n_exec; });

        const auto& st = seq.get_rt_stats_per_thread()[0];
        const double ns_per_tick = seq.get_rt_ns_per_tick();
        seq.set_rt_stats(false);

        res.std_ns_per_frame = st.get_std_ticks() * ns_per_tick;
        res.p50_ns_per_frame = st.get_percentile_ticks(50.0) * ns_per_tick;
        res.p99_ns_per_frame = st.get_percentile_ticks(99.0) * ns_per_tick;
        res.p999_ns_per_frame = st.get_percentile_ticks(99.9) * ns_per_tick;
        res.p9999_ns_per_frame = st.get_percentile_ticks(99.99) * ns_per_tick;
        res.max_ns_per_frame = (double)st.max_ticks * ns_per_tick;
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

    option longopts[] = {
        { "n-exec", required_argument, NULL, 'e' },   { "data-length", required_argument, NULL, 'd' },
        { "backends", required_argument, NULL, 'b' }, { "csv", required_argument, NULL, 'C' },
        { "warmup", required_argument, NULL, 'W' },   { "rt-stats", no_argument, NULL, 'r' },
        { "rt-skip", required_argument, NULL, 'R' },  { "rt-frames", no_argument, NULL, 'F' },
        { "help", no_argument, NULL, 'h' },           { NULL, 0, NULL, 0 }
    };

    size_t n_exec = 300000;
    size_t data_length = 64;
    size_t n_warmup = 256;
    size_t rt_skip = 0;
    bool rt_frames = false;
    bool show_full_rt_tables = false;
    std::string csv_path;

    while (1)
    {
        const int opt = getopt_long(argc, argv, "e:d:b:C:W:R:Frh", longopts, 0);
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
            case 'W':
                n_warmup = std::atoi(optarg);
                break;
            case 'r':
                show_full_rt_tables = true;
                break;
            case 'R':
                rt_skip = std::atoi(optarg);
                break;
            case 'F':
                rt_frames = true;
                break;
            case 'h':
                std::cout << "usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "  -e, --n-exec       Base number of sequence executions per benchmark [" << n_exec << "]"
                          << std::endl;
                std::cout << "  -d, --data-length  Socket buffer length in bytes                    [" << data_length
                          << "]" << std::endl;
                std::cout << "  -b, --backends     Backends (1 backend => full RT percentiles/MAX)  [" << backends_str
                          << "]" << std::endl;
                std::cout << "  -C, --csv          Optional path to export benchmark results as CSV [none]"
                          << std::endl;
                std::cout << "  -W, --warmup       Number of warmup streams before each benchmark   [" << n_warmup
                          << "]" << std::endl;
                std::cout << "  -r, --rt-stats     Also print full tools::Stats::show_rt tables     [false]"
                          << std::endl;
                std::cout << "  -R, --rt-skip      Number of initial frames to skip in RT           [" << rt_skip << "]"
                          << std::endl;
                std::cout << "  -F, --rt-frames    Display RT throughput in frames/s                [false]"
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

    const bool single_backend_rt_mode = (backends.size() == 1);

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
    std::cout << "# StreamPU Empty-Task Runtime Overhead Benchmark (sequence_overhead)  #" << std::endl;
    std::cout << "#######################################################################" << std::endl;
    std::cout << "#   - base n_exec  = " << n_exec << " (scaled for JL_SAFE / long chains)" << std::endl;
    std::cout << "#   - data_length  = " << data_length << " bytes" << std::endl;
    std::cout << "#   - n_warmup     = " << n_warmup << std::endl;
    std::cout << "#   - rt_skip      = " << rt_skip << std::endl;
    std::cout << "#   - has_julia    = " << (has_julia ? "true" : "false") << std::endl;
    std::cout << "#   - backends     = " << active_backends_joined << std::endl;
    std::cout << "#   - display mode = "
              << (single_backend_rt_mode ? "Single-Backend Full RT Percentiles (AVG, STD, P50..P99.99, MAX)"
                                         : "Multi-Backend Average Comparison (pass -b <BACKEND> for RT percentiles)")
              << std::endl;

    std::vector<Csv_row> csv_rows;
    const bool has_cpp = std::find(backends.begin(), backends.end(), "CPP") != backends.end();

    // =================================================================================================================
    // SECTION 1: Baseline & Socket Types (8 empty tasks per sequence)
    // =================================================================================================================
    print_header("1. BASELINE & SOCKET TYPES (8 empty tasks per sequence)", backends);
    if (has_cpp)
    {
        const size_t n_tasks = 8;
        for (size_t i = 0; i < n_warmup; i++)
            for (size_t t = 0; t < n_tasks; t++)
                empty_kernel_0(t);

        const auto t0_clk = std::chrono::steady_clock::now();
        const uint64_t t0_tsc = tools::read_tsc();
        for (size_t i = 0; i < n_exec; i++)
            for (size_t t = 0; t < n_tasks; t++)
                empty_kernel_0(t);
        const uint64_t t1_tsc = tools::read_tsc();
        const auto t1_clk = std::chrono::steady_clock::now();
        const double dt_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_clk - t0_clk).count();
        const double dt_ticks = (double)(t1_tsc - t0_tsc);
        const double ns_per_tick = (dt_ticks > 0.0) ? (dt_ns / dt_ticks) : 1.0;

        Seq_bench_result bare_r;
        bare_r.name = "Bare call (C++ noinline)";
        bare_r.backend = "CPP";
        bare_r.n_tasks = n_tasks;
        bare_r.ns_per_frame = dt_ns / (double)n_exec;
        bare_r.ns_per_task = bare_r.ns_per_frame / (double)n_tasks;
        bare_r.ticks_per_frame = dt_ticks / (double)n_exec;
        bare_r.ticks_per_task = bare_r.ticks_per_frame / (double)n_tasks;

        if (single_backend_rt_mode)
        {
            tools::Rt_stats raw_stats;
            for (size_t i = 0; i < n_exec; i++)
            {
                const uint64_t s0 = tools::read_tsc();
                for (size_t t = 0; t < n_tasks; t++)
                    empty_kernel_0(t);
                const uint64_t s1 = tools::read_tsc();
                if (i >= rt_skip) raw_stats.add_sample(s1 - s0);
            }
            bare_r.std_ns_per_frame = raw_stats.get_std_ticks() * ns_per_tick;
            bare_r.p50_ns_per_frame = raw_stats.get_percentile_ticks(50.0) * ns_per_tick;
            bare_r.p99_ns_per_frame = raw_stats.get_percentile_ticks(99.0) * ns_per_tick;
            bare_r.p999_ns_per_frame = raw_stats.get_percentile_ticks(99.9) * ns_per_tick;
            bare_r.p9999_ns_per_frame = raw_stats.get_percentile_ticks(99.99) * ns_per_tick;
            bare_r.max_ns_per_frame = (double)raw_stats.max_ticks * ns_per_tick;
        }

        std::vector<Seq_bench_result> baseline_cells;
        for (const auto& b : backends)
            baseline_cells.push_back(b == "CPP" ? bare_r : Seq_bench_result());
        print_row("Bare call (C++ noinline)", 8, baseline_cells);
        csv_rows.push_back({ "1_socket_types", "Bare call (C++ noinline)", bare_r });
    }

    struct Sec1_case
    {
        std::string label;
        std::string topo;
        bool no_copy;
    };
    const std::vector<Sec1_case> cases_sec1 = {
        { "0-socket task-bound chain (status only)", "0SCK", true },
        { "SFWD chain (1x OUT, 6x FWD, 1x IN)", "SFWD", true },
        { "SIN+SOUT chain (no_copy_mode=true)", "SINOUT", true },
        { "SIN+SOUT chain (no_copy_mode=false)", "SINOUT", false },
    };
    for (const auto& c : cases_sec1)
    {
        std::vector<Seq_bench_result> row_vals;
        for (const auto& b : backends)
        {
            std::vector<std::string> chain_b(8, b);
            auto bc = build_chain(chain_b, c.topo, data_length, 1, 1, c.no_copy);
            const size_t ne = choose_n_exec(n_exec, chain_b);
            const auto res = measure_sequence_bench(c.label, b, *bc.seq, ne, single_backend_rt_mode, n_warmup, rt_skip);
            row_vals.push_back(res);
            csv_rows.push_back({ "1_socket_types", c.label, res });

            if (show_full_rt_tables && c.topo == "SFWD" && (b == "CPP" || b == "JL_UNSAFE"))
            {
                bc.seq->reset_rt_stats();
                bc.seq->set_rt_stats(true, rt_skip);
                size_t cnt = 0;
                bc.seq->exec([&cnt, ne]() { return ++cnt >= ne; });
                tools::Stats::show_rt(*bc.seq, true, rt_frames);
            }
        }
        print_row(c.label, 8, row_vals);
    }

    // =================================================================================================================
    // SECTION 2: Number of Sockets per Task (1, 2, 4, 8, 16 sockets, SFWD no_copy_mode=true)
    // =================================================================================================================
    print_header("2. NUMBER OF SOCKETS PER TASK (8 tasks, SFWD no_copy_mode=true)", backends);
    for (size_t n_scks : { 1, 2, 4, 8, 16 })
    {
        const std::string label = std::to_string(n_scks) + " socket(s) per task";
        std::vector<Seq_bench_result> row_vals;
        for (const auto& b : backends)
        {
            std::vector<std::string> chain_b(8, b);
            auto bc = build_chain(chain_b, "SFWD", data_length, n_scks, 1, true);
            const size_t ne = choose_n_exec(n_exec, chain_b);
            const auto res = measure_sequence_bench(label, b, *bc.seq, ne, single_backend_rt_mode, n_warmup, rt_skip);
            row_vals.push_back(res);
            csv_rows.push_back({ "2_n_sockets", label, res });
        }
        print_row(label, 8, row_vals);
    }

    // =================================================================================================================
    // SECTION 3: C++ Lambda Capture Size & Julia Stateful Arguments (8 tasks, SFWD)
    // =================================================================================================================
    print_header("3. C++ LAMBDA CAPTURE & JULIA STATEFUL ARGS (8 tasks, SFWD)", backends);
    if (has_cpp)
    {
        const std::vector<std::pair<int, std::string>> capture_cases = {
            { 0, "0-byte capture [] (fast_codelet)" },    { 1, "8-byte capture [ptr] (fast_codelet)" },
            { 2, "16-byte capture [2x size_t] (fast)" },  { 3, "64-byte capture [8x size_t] (fast max)" },
            { 4, "Non-trivial capture (std::function)" },
        };
        for (const auto& cc : capture_cases)
        {
            std::vector<std::string> chain_b(8, "CPP");
            auto bc = build_chain(chain_b, "SFWD", data_length, 1, 1, true, true, false, 1, 1, cc.first);
            const size_t ne = choose_n_exec(n_exec, chain_b);
            const auto res =
              measure_sequence_bench(cc.second, "CPP", *bc.seq, ne, single_backend_rt_mode, n_warmup, rt_skip);

            std::vector<Seq_bench_result> row_vals;
            for (const auto& b : backends)
                row_vals.push_back(b == "CPP" ? res : Seq_bench_result());
            print_row(cc.second, 8, row_vals);
            csv_rows.push_back({ "3_cpp_capture_and_jl_state", cc.second, res });
        }
    }

#ifdef SPU_JULIA
    {
        const bool has_jl_any = std::find(backends.begin(), backends.end(), "JL_UNSAFE") != backends.end() ||
                                std::find(backends.begin(), backends.end(), "JL_SAFE") != backends.end();
        if (has_jl_any)
        {
            tools::Julia::eval_string(R""""(
                mutable struct _SpuEmptyBenchState
                    v::Int64
                end
            )"""");
            for (size_t n_states : { 0, 1, 2 })
            {
                for (size_t n_consts : { 0, 2 })
                {
                    const std::string label = "Stateful_Julia: " + std::to_string(n_states) + " state(s), " +
                                              std::to_string(n_consts) + " const(s)";
                    const std::string fn_name =
                      "_spu_bench_st_" + std::to_string(n_states) + "_" + std::to_string(n_consts);
                    std::string arg_list;
                    for (size_t st = 0; st < n_states; st++)
                        arg_list += "stt_" + std::to_string(st) + ", ";
                    arg_list += "s0, ";
                    for (size_t c = 0; c < n_consts; c++)
                        arg_list += "cst_" + std::to_string(c) + ", ";
                    tools::Julia::eval_string(fn_name + "(" + arg_list +
                                              "rnt_frame_id, rnt_n_frames_per_wave) = Int32(0)");

                    std::vector<Seq_bench_result> row_vals;
                    for (const auto& b : backends)
                    {
                        if (b != "JL_UNSAFE" && b != "JL_SAFE")
                        {
                            row_vals.push_back(Seq_bench_result());
                            continue;
                        }
                        std::vector<std::unique_ptr<module::Stateful_Julia>> mods(8);
                        for (size_t i = 0; i < 8; i++)
                        {
                            mods[i].reset(new module::Stateful_Julia());
                            mods[i]->set_jl_safety(b == "JL_SAFE");
                            for (size_t st = 0; st < n_states; st++)
                                mods[i]->create_state("_SpuEmptyBenchState(" + std::to_string(st) + ")", false);
                            auto& t = mods[i]->create_task("exec");
                            if (i == 0)
                                mods[i]->create_socket_out<uint8_t>(t, "s0", data_length);
                            else if (i < 7)
                                mods[i]->create_socket_fwd<uint8_t>(t, "s0", data_length);
                            else
                                mods[i]->create_socket_in<uint8_t>(t, "s0", data_length);
                            for (size_t c = 0; c < n_consts; c++)
                                mods[i]->create_constant<uint32_t>(t, (uint32_t)c);
                            mods[i]->create_codelet(t, "", fn_name);
                        }
                        for (size_t i = 0; i < 7; i++)
                            (*mods[i + 1])["exec::s0"] = (*mods[i])["exec::s0"];

                        runtime::Sequence seq((*mods[0])("exec"), 1);
                        seq.set_no_copy_mode(true);
                        for (auto* m : seq.get_modules<module::Module>(false))
                            for (auto& t : m->tasks)
                                t->set_fast(true);

                        const size_t ne = choose_n_exec(n_exec, { b });
                        const auto res =
                          measure_sequence_bench(label, b, seq, ne, single_backend_rt_mode, n_warmup, rt_skip);
                        row_vals.push_back(res);
                        csv_rows.push_back({ "3_cpp_capture_and_jl_state", label, res });
                    }
                    print_row(label, 8, row_vals);
                }
            }
        }
    }
#endif

    // =================================================================================================================
    // SECTION 4: Task & Sequence Execution Modes (8 empty tasks, SFWD)
    // =================================================================================================================
    print_header("4. TASK & SEQUENCE EXECUTION MODES (8 empty tasks, SFWD)", backends);
    struct Mode_case
    {
        std::string label;
        bool fast_m;
        bool with_st;
        bool stats_m;
        double scale;
    };
    const std::vector<Mode_case> mode_cases = {
        { "fast=true, _exec_without_statuses", true, false, false, 1.0 },
        { "fast=true, _exec (with statuses)", true, true, false, 1.0 },
        { "fast=false (full checks & fill)", false, false, false, 1.0 },
        { "stats=true (per-task steady_clock)", true, false, true, 0.25 },
    };
    for (const auto& mc : mode_cases)
    {
        std::vector<Seq_bench_result> row_vals;
        for (const auto& b : backends)
        {
            std::vector<std::string> chain_b(8, b);
            auto bc = build_chain(chain_b, "SFWD", data_length, 1, 1, true, mc.fast_m, mc.stats_m);
            const size_t ne = choose_n_exec(n_exec, chain_b, mc.scale);
            const auto res =
              measure_sequence_bench(mc.label, b, *bc.seq, ne, single_backend_rt_mode, n_warmup, rt_skip, mc.with_st);
            row_vals.push_back(res);
            csv_rows.push_back({ "4_exec_modes", mc.label, res });
        }
        print_row(mc.label, 8, row_vals);
    }

    // =================================================================================================================
    // SECTION 5: Sequence Chain Length Scaling (2, 4, 8, 16, 32, 64, 128 tasks)
    // =================================================================================================================
    print_header("5. SEQUENCE CHAIN LENGTH SCALING (amortization of Sequence loop)", backends);
    for (size_t chain_len : { 2, 4, 8, 16, 32, 64, 128 })
    {
        const std::string label = "Chain length = " + std::to_string(chain_len) + " tasks";
        const double scale = 8.0 / (double)chain_len;
        std::vector<Seq_bench_result> row_vals;
        for (const auto& b : backends)
        {
            std::vector<std::string> chain_b(chain_len, b);
            auto bc = build_chain(chain_b, "SFWD", data_length);
            const size_t ne = choose_n_exec(n_exec, chain_b, scale);
            const auto res = measure_sequence_bench(label, b, *bc.seq, ne, single_backend_rt_mode, n_warmup, rt_skip);
            row_vals.push_back(res);
            csv_rows.push_back({ "5_chain_length", label, res });
        }
        print_row(label, chain_len, row_vals);
    }

    // =================================================================================================================
    // SECTION 6: Multi-Wave Execution (8 tasks, n_waves = 1, 2, 4, 8; n_frames = n_waves, n_frames_per_wave = 1)
    // =================================================================================================================
    print_header("6. MULTI-WAVE EXECUTION (8 tasks, n_frames = n_waves, n_frames_per_wave ou fpw = 1)", backends);
    for (size_t n_waves : { 1, 2, 4, 8 })
    {
        const std::string label =
          "n_waves = " + std::to_string(n_waves) + " (n_frames=" + std::to_string(n_waves) + ", fpw=1)";
        const double scale = 1.0 / (double)n_waves;
        std::vector<Seq_bench_result> row_vals;
        for (const auto& b : backends)
        {
            std::vector<std::string> chain_b(8, b);
            auto bc = build_chain(chain_b, "SFWD", data_length, 1, 1, true, true, false, n_waves, 1);
            const size_t ne = choose_n_exec(n_exec, chain_b, scale);
            const auto res = measure_sequence_bench(
              label, b, *bc.seq, ne, single_backend_rt_mode, n_warmup, rt_skip, false, 8 * n_waves);
            row_vals.push_back(res);
            csv_rows.push_back({ "6_multi_wave", label, res });
        }
        print_row(label, 8 * n_waves, row_vals);
    }

    // =================================================================================================================
    // SECTION 7: Multi-Thread, Heterogeneous & Control Flow
    // =================================================================================================================
    print_header("7. MULTI-THREAD EFFECTIVE LATENCY & CONTROL FLOW", backends);
    for (size_t n_threads : { 1, 2, 4 })
    {
        const std::string label = "Threads = " + std::to_string(n_threads) + " (homogeneous 8 tasks)";
        std::vector<Seq_bench_result> row_vals;
        for (const auto& b : backends)
        {
            std::vector<std::string> chain_b(8, b);
            auto bc = build_chain(chain_b, "SFWD", data_length, 1, n_threads);
            const size_t ne = choose_n_exec(n_exec, chain_b, (double)n_threads);
            const auto res =
              measure_sequence_bench(label, b, *bc.seq, ne, single_backend_rt_mode, n_warmup, rt_skip, false, 0, true);
            row_vals.push_back(res);
            csv_rows.push_back({ "7_multithread_and_ctrl", label, res });
        }
        print_row(label, 8, row_vals);
    }

#ifdef SPU_JULIA
    if (has_cpp && std::find(backends.begin(), backends.end(), "JL_UNSAFE") != backends.end())
    {
        const std::vector<std::string> het_pattern = { "CPP", "JL_UNSAFE", "CPP", "JL_UNSAFE",
                                                       "CPP", "JL_UNSAFE", "CPP", "JL_UNSAFE" };
        for (size_t n_threads : { 1, 4 })
        {
            auto bc = build_chain(het_pattern, "SFWD", data_length, 1, n_threads);
            const size_t ne = choose_n_exec(n_exec, het_pattern, (double)n_threads);
            const std::string label = "Heterogeneous (CPP+JL_UNSAFE, t=" + std::to_string(n_threads) + ")";
            const auto res =
              measure_sequence_bench(label, "MIXED", *bc.seq, ne, false, n_warmup, rt_skip, false, 0, true);
            std::cout << "#  " << std::left << std::setw(39) << label.substr(0, 39) << std::right
                      << " |     8 | ns/stm || " << std::setw(12) << fmt_ns(res.ns_per_frame)
                      << " (mixed CPP + JL_UNSAFE)" << std::endl;
            std::cout << "#  " << std::left << std::setw(39) << "" << std::right << " |     8 | ns/tsk || "
                      << std::setw(12) << fmt_ns(res.ns_per_task) << " (mixed CPP + JL_UNSAFE)" << std::endl;
            csv_rows.push_back({ "7_multithread_and_ctrl", label, res });
        }
    }
#endif

    // Switcher commute + select control flow (6 tasks executed along path 0: ctrl_gen, src, commute, branch0, select,
    // snk)
    for (bool no_copy : { true, false })
    {
        const std::string label =
          std::string("Switcher commute+select (") + (no_copy ? "no_copy=true)" : "no_copy=false)");
        std::vector<Seq_bench_result> row_vals;
        for (const auto& b : backends)
        {
            std::unique_ptr<module::Module> src = make_empty_module(b, { { Sck_dir::OUT, "data" } }, data_length);
            module::Stateless ctrl_gen;
            auto& t_ctrl = ctrl_gen.create_task("exec");
            auto s_ctrl = ctrl_gen.create_socket_out<int8_t>(t_ctrl, "ctrl", 1);
            ctrl_gen.create_codelet(t_ctrl,
                                    [s_ctrl](module::Module&, runtime::Task& tsk, const size_t fid) -> int
                                    {
                                        int8_t* ctrl = static_cast<int8_t*>(tsk[s_ctrl].get_dataptr());
                                        ctrl[0] = 0;
                                        empty_kernel_1(ctrl, fid);
                                        return runtime::status_t::SUCCESS;
                                    });

            module::Switcher switcher(2, data_length, typeid(uint8_t));
            std::unique_ptr<module::Module> branch0 =
              make_empty_module(b, { { Sck_dir::IN, "in" }, { Sck_dir::OUT, "out" } }, data_length);
            std::unique_ptr<module::Module> branch1 =
              make_empty_module(b, { { Sck_dir::IN, "in" }, { Sck_dir::OUT, "out" } }, data_length);
            std::unique_ptr<module::Module> snk = make_empty_module(b, { { Sck_dir::IN, "in" } }, data_length);

            (*src)("exec") = ctrl_gen("exec");
            switcher["commute::in_data"] = (*src)["exec::data"];
            switcher["commute::in_ctrl"] = ctrl_gen["exec::ctrl"];
            (*branch0)["exec::in"] = switcher["commute::out_data0"];
            (*branch1)["exec::in"] = switcher["commute::out_data1"];
            switcher["select::in_data0"] = (*branch0)["exec::out"];
            switcher["select::in_data1"] = (*branch1)["exec::out"];
            (*snk)["exec::in"] = switcher["select::out_data"];

            runtime::Sequence seq(ctrl_gen("exec"), 1);
            seq.set_no_copy_mode(no_copy);
            for (auto* m : seq.get_modules<module::Module>(false))
                for (auto& t : m->tasks)
                    t->set_fast(true);

            const size_t ne = choose_n_exec(n_exec, { b });
            const auto res =
              measure_sequence_bench(label, b, seq, ne, single_backend_rt_mode, n_warmup, rt_skip, false, 6);
            row_vals.push_back(res);
            csv_rows.push_back({ "7_multithread_and_ctrl", label, res });
        }
        print_row(label, 6, row_vals);
    }

    if (!csv_path.empty())
    {
        std::ofstream ofs(csv_path);
        ofs << "section,case,backend,n_tasks,ns_per_frame,ns_per_task,ticks_per_frame,ticks_per_task,"
               "std_ns_per_frame,p50_ns_per_frame,p99_ns_per_frame,p999_ns_per_frame,p9999_ns_per_frame,"
               "max_ns_per_frame\n";
        for (const auto& r : csv_rows)
        {
            ofs << r.section << ",\"" << r.case_name << "\"," << r.res.backend << "," << r.res.n_tasks << ","
                << r.res.ns_per_frame << "," << r.res.ns_per_task << "," << r.res.ticks_per_frame << ","
                << r.res.ticks_per_task << "," << r.res.std_ns_per_frame << "," << r.res.p50_ns_per_frame << ","
                << r.res.p99_ns_per_frame << "," << r.res.p999_ns_per_frame << "," << r.res.p9999_ns_per_frame << ","
                << r.res.max_ns_per_frame << "\n";
        }
        std::cout << "# Wrote CSV results to " << csv_path << std::endl;
    }

    std::cout << "#" << std::endl;
    std::cout << "# " << rang::style::bold << rang::fg::green << "Empty-task sequence overhead breakdown completed!"
              << rang::style::reset << std::endl;
    return 0;
}
