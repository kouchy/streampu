/*!
 * \file
 * \brief Class runtime::Sequence.
 */
#ifndef SEQUENCE_HPP_
#define SEQUENCE_HPP_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Module/Module.hpp"
#include "Runtime/Socket/Socket.hpp"
#include "Tools/Algo/Digraph/Digraph_node.hpp"
#include "Tools/Display/Statistics/Statistics.hpp"
#include "Tools/Interface/Interface_clone.hpp"
#include "Tools/Interface/Interface_get_set_n_frames.hpp"
#include "Tools/Interface/Interface_is_done.hpp"
#include "Tools/Thread/Thread_pool/Thread_pool.hpp"

namespace spu
{
namespace module
{
class Task;
class Module;
} // namespace module
namespace sched
{
class Scheduler;
}
namespace runtime
{
class Pipeline;

enum class subseq_t : size_t
{
    STD,
    COMMUTE,
    SELECT
};

// Identifies the post-execution zero-copy socket-rebinding action associated with a task in `no_copy_mode`.
// Storing a lightweight flat descriptor per task in `Sub_sequence_generic::custom_processes` eliminates
// `std::function<const int*()>` indirect calls in `Sequence::exec_subsequence*` even when a sub-sequence contains
// `Switcher` or `Adaptor_m_to_n` tasks (such as inside `Pipeline` stages or `no_copy_mode` control-flow loops).
enum class process_t : uint8_t
{
    STD = 0,
    SWITCHER_SELECT = 1,
    SWITCHER_COMMUTE = 2,
    ADAPTOR_PULL = 3,
    ADAPTOR_PUSH = 4
};

struct Custom_process_info
{
    process_t type;
    uint32_t rebind_id;
    void* module_ptr;
};

template<class VTA = std::vector<runtime::Task*>>
class Sub_sequence_generic
{
  public:
    subseq_t type;
    VTA tasks;
    std::vector<size_t> tasks_id;
    size_t id;

    // usefull in case of adaptor to make zero copy and restore original states at
    // the end of the chain execution
    std::vector<std::vector<std::vector<Socket*>>> rebind_sockets;
    std::vector<std::vector<std::vector<void*>>> rebind_dataptrs;
    // Per-task flat descriptors for zero-copy socket rebinding when `has_custom_processes == true`.
    std::vector<Custom_process_info> custom_processes;
    // True only when no_copy_mode injects custom socket-rebinding hooks (for Switcher or Adaptor_m_to_n).
    // When false (the vast majority of sub-sequences), Sequence::exec_subsequence* executes a tight loop of
    // inlined `task->exec()` calls without checking `custom_processes`.
    bool has_custom_processes;

    explicit Sub_sequence_generic()
      : type(subseq_t::STD)
      , id(0)
      , has_custom_processes(false)
    {
    }
    virtual ~Sub_sequence_generic() = default;
};

using Sub_sequence = Sub_sequence_generic<std::vector<runtime::Task*>>;
using Sub_sequence_const = Sub_sequence_generic<std::vector<const runtime::Task*>>;

class Sequence
  : public tools::Interface_clone
  , public tools::Interface_get_set_n_frames
  , public tools::Interface_is_done
{
    friend Pipeline;
    friend sched::Scheduler;
    friend tools::Statistics;

  protected:
    const size_t n_threads;
    std::shared_ptr<tools::Thread_pool> thread_pool;

    std::vector<tools::Digraph_node<Sub_sequence>*> sequences;
    std::vector<size_t> firsts_tasks_id;
    std::vector<size_t> lasts_tasks_id;
    std::vector<std::vector<runtime::Task*>> firsts_tasks;
    std::vector<std::vector<runtime::Task*>> lasts_tasks;
    std::vector<std::vector<std::shared_ptr<module::Module>>> modules;
    std::vector<std::vector<module::Module*>> all_modules;
    std::vector<std::vector<module::Module*>> jl_modules;
    std::shared_ptr<std::mutex> mtx_exception;
    std::vector<std::string> prev_exception_messages;
    std::vector<std::string> prev_exception_messages_to_display;
    std::shared_ptr<std::atomic<bool>> force_exit_loop;
    size_t n_tasks;
    bool tasks_inplace;
    bool thread_pinning;
    std::vector<size_t> puids;
    bool no_copy_mode;
    const std::vector<const runtime::Task*> saved_exclusions;
    std::vector<tools::Interface_is_done*> donners;
    std::vector<std::vector<tools::Interface_reset*>> switchers_reset;
    bool auto_stop;
    bool is_part_of_pipeline;

    // internal state for the `exec_step` method
    std::vector<bool> next_round_is_over;
    std::vector<size_t> cur_task_id;
    std::vector<tools::Digraph_node<Sub_sequence>*> cur_ss;

    // extra attribute for pinning v2
    std::vector<std::string> pin_objects_per_thread;

    // is memory allocation enabled?
    bool memory_allocation;

    // Real-Time (RT) latency and jitter statistics state (enabled via `set_rt_stats(true, n_skip_frames)`).
    bool rt_stats;                                    // true when RT frame-level profiling is active
    size_t rt_stats_skip_frames;                      // number of warmup executions to skip per thread
    bool is_first_pipeline_stage;                     // true when this Sequence is stage 0 of a Pipeline
    bool is_last_pipeline_stage;                      // true when this Sequence is the final stage of a Pipeline
    double rt_ns_per_tick;                            // calibrated nanoseconds per `tools::read_tsc()` tick
    std::vector<size_t> rt_stats_skipped_per_thread;  // count of skipped warmup executions per thread
    std::vector<tools::Rt_stats> rt_stats_per_thread; // per-thread sequence/stage processing latency (excl. wait)
    std::vector<tools::Rt_stats>
      pipeline_e2e_rt_stats_per_thread; // per-thread Pipeline end-to-end latency (incl. wait & FIFO)
    std::vector<tools::Rt_stats>
      pipeline_inter_rt_stats_per_thread;     // per-thread inter-frame wall-clock period at pipeline output
    std::vector<uint64_t> cur_frame_birth_ts; // per-thread birth timestamp of the current frame in stage 0
    std::vector<uint64_t> cur_stage_start_ts; // per-thread start timestamp of the current stage execution

  public:
    Sequence(const std::vector<const runtime::Task*>& firsts,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool memory_allocation = true);
    Sequence(const std::vector<const runtime::Task*>& firsts,
             const std::vector<const runtime::Task*>& lasts,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool memory_allocation = true);
    Sequence(const std::vector<const runtime::Task*>& firsts,
             const std::vector<const runtime::Task*>& lasts,
             const std::vector<const runtime::Task*>& exclusions,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool memory_allocation = true);
    Sequence(const runtime::Task& first,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool memory_allocation = true);
    Sequence(const runtime::Task& first,
             const runtime::Task& last,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool memory_allocation = true);
    Sequence(const std::vector<runtime::Task*>& firsts,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(const std::vector<runtime::Task*>& firsts,
             const std::vector<runtime::Task*>& lasts,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(const std::vector<runtime::Task*>& firsts,
             const std::vector<runtime::Task*>& lasts,
             const std::vector<runtime::Task*>& exclusions,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(runtime::Task& first,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(runtime::Task& first,
             runtime::Task& last,
             const size_t n_threads = 1,
             const bool thread_pinning = false,
             const std::vector<size_t>& puids = {},
             const bool tasks_inplace = true,
             const bool memory_allocation = true);

    // Constructors for pinning v2
    Sequence(const std::vector<const runtime::Task*>& firsts,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool memory_allocation = true);
    Sequence(const std::vector<const runtime::Task*>& firsts,
             const std::vector<const runtime::Task*>& lasts,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool memory_allocation = true);
    Sequence(const std::vector<const runtime::Task*>& firsts,
             const std::vector<const runtime::Task*>& lasts,
             const std::vector<const runtime::Task*>& exclusions,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool memory_allocation = true);
    Sequence(const runtime::Task& first,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool memory_allocation = true);
    Sequence(const runtime::Task& first,
             const runtime::Task& last,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool memory_allocation = true);
    Sequence(const std::vector<runtime::Task*>& firsts,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(const std::vector<runtime::Task*>& firsts,
             const std::vector<runtime::Task*>& lasts,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(const std::vector<runtime::Task*>& firsts,
             const std::vector<runtime::Task*>& lasts,
             const std::vector<runtime::Task*>& exclusions,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(runtime::Task& first,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool tasks_inplace = true,
             const bool memory_allocation = true);
    Sequence(runtime::Task& first,
             runtime::Task& last,
             const size_t n_threads,
             const bool thread_pinning,
             const std::string& sequence_pinning_policy,
             const bool tasks_inplace = true,
             const bool memory_allocation = true);

    virtual ~Sequence();
    virtual Sequence* clone() const;

    void set_thread_pinning(const bool thread_pinning);
    void set_thread_pinning(const bool thread_pinning, const std::vector<size_t>& puids = {});
    // Set pinning for the second version
    void set_thread_pinning(const bool thread_pinning, const std::string& sequence_pinning_policy);
    bool is_thread_pinning();

    // Allocate sequence memory
    void allocate_outbuffers();
    // Deallocate sequence memory
    void deallocate_outbuffers();

    void exec(std::function<bool(const std::vector<const int*>&)> stop_condition);
    void exec(std::function<bool()> stop_condition);
    void exec();
    void exec_seq(const size_t tid = 0, const int frame_id = -1);
    runtime::Task* exec_step(const size_t tid = 0, const int frame_id = -1);

    inline size_t get_n_threads() const;

    template<class C = module::Module>
    std::vector<C*> get_modules(const bool set_modules = true) const;
    template<class C = module::Module>
    std::vector<C*> get_cloned_modules(const C& module_ref) const;

    std::vector<std::vector<module::Module*>> get_modules_per_threads() const;
    std::vector<std::vector<module::Module*>> get_modules_per_types() const;

    std::vector<std::vector<runtime::Task*>> get_tasks_per_threads() const;
    std::vector<std::vector<runtime::Task*>> get_tasks_per_types() const;

    inline const std::vector<std::vector<runtime::Task*>>& get_firsts_tasks() const;
    inline const std::vector<std::vector<runtime::Task*>>& get_lasts_tasks() const;

    void export_dot(std::ostream& stream = std::cout) const;

    void set_no_copy_mode(const bool no_copy_mode);
    bool is_no_copy_mode() const;

    void set_auto_stop(const bool auto_stop);
    bool is_auto_stop() const;

    /*!
     * \brief Enables or disables low-overhead Real-Time (RT) frame latency and jitter profiling on this Sequence.
     * \param rt_stats: true to record per-execution hardware timestamps into lock-free `tools::Rt_stats` histograms.
     * \param n_skip_frames: number of initial executions per thread to discard before recording statistics
     *                       (useful to ignore cold-cache, page-fault, DVFS, or JIT warmup transients; default: 0).
     */
    void set_rt_stats(const bool rt_stats, const size_t n_skip_frames = 0);
    bool is_rt_stats() const;
    size_t get_rt_stats_skip_frames() const;
    void reset_rt_stats();
    const std::vector<tools::Rt_stats>& get_rt_stats_per_thread() const;
    const std::vector<tools::Rt_stats>& get_pipeline_e2e_rt_stats_per_thread() const;
    const std::vector<tools::Rt_stats>& get_pipeline_inter_rt_stats_per_thread() const;
    double get_rt_ns_per_tick() const;

    inline size_t get_n_frames() const;
    void set_n_frames(const size_t n_frames);

    virtual bool is_done() const;

    bool is_control_flow() const;

    bool has_julia() const;

  protected:
    template<class SS>
    void delete_tree(tools::Digraph_node<SS>* node, std::vector<tools::Digraph_node<SS>*>& already_deleted_nodes);

    template<class SS, class TA>
    tools::Digraph_node<SS>* init_recursive(tools::Digraph_node<SS>* cur_subseq,
                                            size_t& ssid,
                                            size_t& taid,
                                            std::vector<std::pair<TA*, tools::Digraph_node<SS>*>>& selectors,
                                            std::vector<TA*>& switchers,
                                            TA& first,
                                            TA& current_task,
                                            const std::vector<TA*>& lasts,
                                            const std::vector<TA*>& exclusions,
                                            std::vector<size_t>& real_lasts_id,
                                            std::vector<TA*>& real_lasts,
                                            std::map<TA*, unsigned>& in_sockets_feed,
                                            std::map<TA*, std::pair<tools::Digraph_node<SS>*, size_t>>& task_subseq);

    template<class VTA>
    void export_dot_subsequence(const VTA& subseq,
                                const std::vector<size_t>& tasks_id,
                                const subseq_t& subseq_type,
                                const std::string& subseq_name,
                                const std::string& tab,
                                std::ostream& stream = std::cout) const;

    template<class VTA>
    void export_dot_connections(const VTA& subseq, const std::string& tab, std::ostream& stream = std::cout) const;

    template<class SS>
    void export_dot(tools::Digraph_node<SS>* root, std::ostream& stream = std::cout) const;

    template<class SS, class MO>
    void replicate(const tools::Digraph_node<SS>* sequence);

    void _exec(const size_t tid,
               std::function<bool(const std::vector<const int*>&)>& stop_condition,
               tools::Digraph_node<Sub_sequence>* sequence);

    void _exec_without_statuses(const size_t tid,
                                std::function<bool()>& stop_condition,
                                tools::Digraph_node<Sub_sequence>* sequence);

    void gen_processes(const bool no_copy_mode = false);
    void reset_no_copy_mode();

    template<class SS>
    void check_ctrl_flow(tools::Digraph_node<SS>* root);
    Sub_sequence* get_last_subsequence(const size_t tid);
    void update_tasks_id(const size_t tid);

    std::vector<runtime::Task*> get_tasks_from_id(const size_t taid);

    void update_firsts_and_lasts_tasks();

    void _set_n_frames_unbind(std::vector<std::pair<runtime::Socket*, runtime::Socket*>>& unbind_sockets,
                              std::vector<std::pair<runtime::Task*, runtime::Socket*>>& unbind_tasks);
    void _set_n_frames(const size_t n_frames);
    void _set_n_frames_rebind(const std::vector<std::pair<runtime::Socket*, runtime::Socket*>>& unbind_sockets,
                              const std::vector<std::pair<runtime::Task*, runtime::Socket*>>& unbind_tasks);

  private:
    static void apply_custom_process_rebind(Sub_sequence* contents, Task* task, const Custom_process_info& info);
    static inline void exec_subsequence(tools::Digraph_node<Sub_sequence>* cur_ss, std::vector<const int*>& statuses);
    static inline void exec_subsequence_without_statuses(tools::Digraph_node<Sub_sequence>* cur_ss);
    template<class SS, class TA>
    void init(const std::vector<TA*>& firsts, const std::vector<TA*>& lasts, const std::vector<TA*>& exclusions);
    template<class SS>
    inline void _init(tools::Digraph_node<SS>* root);
    void init_jl_module(module::Module* m);
    void eval_jl_modules(const size_t tid);
};
} // namespace runtime
} // namespace spu

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include "Runtime/Sequence/Sequence.hxx"
#endif

#endif /* SEQUENCE_HPP_ */
