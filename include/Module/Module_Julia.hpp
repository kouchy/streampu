/*!
 * \file
 * \brief Class module::Module_Julia.
 */
#ifndef MODULE_JULIA_HPP_
#define MODULE_JULIA_HPP_

#ifdef SPU_JULIA

#include <atomic>
#include <cstdint>
#include <functional>
#include <julia.h>
#include <memory>
#include <string>
#include <vector>

#include "Module/Module.hpp"
#include "Runtime/Task/Task.hpp"
#include "Tools/Julia/Proxy.hpp"

namespace spu
{
namespace module
{
/*!
 * \class Module_Julia
 *
 * \brief Common Julia execution and state-management mixin shared by Stateless_Julia and Stateful_Julia.
 */
class Module_Julia
{
  protected:
    /*!
     * \brief Per-socket cache of GC-rooted 1D Julia `Array` (`Vector{T}`) views wrapping C++ socket buffers.
     *
     * Avoids calling `jl_apply_array_type()`, `jl_ptr_to_array_1d()`, and `Ref_keeper::mark()`/`unmark()` on every
     * codelet execution when socket buffer addresses (`dataptr`) remain unchanged or cycle through a fixed set of
     * addresses (such as multi-wave execution `w = 0..W-1` or Pipeline adaptor double-buffers).
     */
    struct Socket_jl_cache
    {
        jl_value_t* array_type = nullptr; //!< Cached `Vector{T}` Julia `DataType` resolved once in `eval()`.
        void* last_dataptr =
          nullptr;                 //!< Fast-path check: last C++ buffer pointer bound to `func_args` for this socket.
        size_t last_databytes = 0; //!< Fast-path check: last socket byte size.
        size_t last_n_frames_per_wave = 0; //!< Fast-path check: last `n_frames_per_wave` value.
        size_t last_hit_idx =
          0; //!< Index of the most recently matched entry in `entries` (for O(1) cyclic prediction).
        size_t next_evict_idx = 0; //!< Round-robin (FIFO) replacement cursor when `entries.size() == max_entries`.
        struct Entry
        {
            void* dataptr;        //!< Wrapped C++ buffer base pointer.
            size_t n_elmts;       //!< Number of elements in the 1D Julia array view.
            jl_value_t* jl_array; //!< Pointer to the 1D Julia `Array` object (`jl_array_t*`).
            size_t jl_id;         //!< Slot index in the owning thread's `Ref_keeper` keeping `jl_array` GC-rooted.
        };
        std::vector<Entry> entries; //!< Cached `(dataptr, n_elmts) -> jl_array` descriptors for this socket.
    };

    Module* self_module;
    Module_Julia& (*to_module_julia)(Module&);
    std::vector<jl_value_t*> jl_states_ptr;
    std::vector<size_t> jl_states_id;
    std::vector<std::vector<jl_value_t*>> jl_constants_ptr;
    std::vector<std::vector<size_t>> jl_constants_id;
    std::vector<std::vector<Socket_jl_cache>> jl_sockets_cache;
    std::vector<std::vector<void*>> jl_func_args;
    bool evaluated;
    // Address of the thread_local Ref_keeper singleton that evaluated this module instance (set in eval()).
    // In multi-threaded Sequences and Pipelines, cloned Stateful_Julia / Stateless_Julia modules are evaluated on their
    // respective worker threads (allocating state and constant slot IDs in each worker's thread_local Ref_keeper),
    // whereas ~Sequence() always runs on the main thread and destroys all clones there. Tracking the owning Ref_keeper
    // ensures that ~Module_Julia() only calls tools::Julia::release(jl_id) when destroyed on the same thread that
    // evaluated it, preventing worker-thread slot IDs from corrupting the main thread's Ref_keeper freelist and marks
    // array.
    void* jl_eval_thread_keeper;
    std::shared_ptr<size_t> n_clones;

    std::shared_ptr<std::vector<std::function<void()>>> jl_evaluate_states;
    std::shared_ptr<std::vector<std::function<void(Module_Julia& m)>>> jl_create_states;
    std::shared_ptr<std::vector<std::vector<std::function<void(Module_Julia& m)>>>> jl_create_constants;
    std::shared_ptr<std::vector<std::function<void(Module_Julia& m)>>> jl_evaluate;
    std::shared_ptr<std::vector<std::function<void(Module_Julia& m)>>> jl_create_codelet;

    bool jl_safe;

  public:
    Module_Julia(Module* self_module, Module_Julia& (*to_module_julia)(Module&), const bool jl_safe = true);
    virtual ~Module_Julia();

    bool is_eval() const;

    void set_jl_safety(const bool jl_safe);
    bool is_jl_safe();

    template<typename T>
    void create_constant(runtime::Task& task, const T& value);

    void create_cdl(runtime::Task& task,
                    const std::string& julia_code = "",
                    const std::string& julia_function_name = "");
    void create_codelet(runtime::Task& task,
                        const std::string& julia_code = "",
                        const std::string& julia_function_name = "");

    void create_cdl_file(runtime::Task& task,
                         const std::string& julia_filepath,
                         const std::string& julia_function_name = "");
    void create_codelet_file(runtime::Task& task,
                             const std::string& julia_filepath,
                             const std::string& julia_function_name = "");

    void create_cdl_function(runtime::Task& task, jl_function_t* f);
    void create_codelet_function(runtime::Task& task, jl_function_t* f);

    void eval();

  protected:
    void deep_copy(const Module_Julia& m, Module* self_module);

    runtime::Task& _create_task(const std::string& name);

    void _create_codelet(runtime::Task& task, const std::string& julia_function_name = "");

    void _create_state(const std::string& julia_constructor,
                       const bool shared = false,
                       const std::string& julia_code = "");
    void _create_state_file(const std::string& julia_constructor,
                            const bool shared = false,
                            const std::string& julia_filepath = "");
    void _register_state(const std::string& julia_constructor, const bool shared);
    const std::vector<jl_value_t*>& _get_states() const;

    static size_t get_task_id(runtime::Task& task);
    static jl_value_t* get_jl_datatype(const runtime::Socket* sck);
    static tools::Proxy jl_new_array_from_data(const runtime::Socket* sck, size_t n_elmts);
    jl_value_t* get_or_create_socket_array(const size_t tid, const size_t s, void* p, const size_t n_elmts);
};
}
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include "Module/Module_Julia.hxx"
#endif

#endif /* SPU_JULIA */

#endif /* MODULE_JULIA_HPP_ */
