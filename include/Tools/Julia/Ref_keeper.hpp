#ifndef REF_KEEPER_HPP_
#define REF_KEEPER_HPP_

#ifdef SPU_JULIA

#include <julia.h>
#include <string>
#include <vector>

#if JULIA_VERSION_MAJOR > 1 || (JULIA_VERSION_MAJOR == 1 && JULIA_VERSION_MINOR >= 13)
typedef jl_value_t jl_function_t;
#endif

namespace spu
{
namespace tools
{
/*!
 * \class Ref_keeper
 *
 * \brief Thread-local Garbage Collector (GC) root frame manager for Julia objects held by C++.
 *
 * Each C++ thread interacting with Julia owns a `thread_local` singleton `Ref_keeper` whose `marks` array
 * is pushed onto the thread's Julia GC stack (`jl_pgcstack`) as a `jl_gcframe_t`.
 * - `marks[0]` stores the encoded GC frame size (`JL_GC_ENCODE_PUSHARGS(size + 1)`).
 * - `marks[1]` stores the pointer to the previous `jl_gcframe_t` on the thread's GC stack.
 * - `marks[2]` permanently roots `last_result` (a `Ref{Any}` cell used to temporarily hold return values).
 * - `marks[3 .. size + 2]` are dynamically allocatable slots managed via a high-water mark (`cpt`) and a
 *   LIFO `freelist` of recycled slot indices.
 */
class Ref_keeper
{
  private:
    jl_value_t* last_result;

    size_t size;
    jl_value_t** marks;
    size_t cpt;

    std::vector<size_t> freelist;

    Ref_keeper()
      : last_result(NULL)
      , size(0)
      , marks(NULL)
      , cpt(0)
      , freelist(0)
    {
    }

  public:
    ~Ref_keeper();

    size_t get_size() const;
    jl_value_t* get_res() const;

    jl_value_t* call0(jl_function_t* f);
    jl_value_t* call1(jl_function_t* f, jl_value_t* a);
    jl_value_t* call(jl_function_t* f, jl_value_t** args, size_t n);
    jl_value_t* safe_call(jl_function_t* f, jl_value_t** args, size_t n);

    jl_value_t* eval(std::string code);
    jl_value_t* unsafe_eval(std::string code);

    /*!
     * \brief Roots a Julia value `v` in the calling thread's GC frame and returns its slot index.
     *        Automatically doubles the GC frame capacity (2x) in place if all existing slots are in use.
     */
    size_t mark(jl_value_t* v);

    /*!
     * \brief Releases slot `id` by replacing its pointer with `last_result` and pushing `id` onto `freelist`.
     */
    void unmark(size_t id);

    /*!
     * \brief Replaces the rooted Julia pointer stored at an already-allocated slot `id` in O(1) without
     *        touching `freelist`. Used by `Module_Julia::get_or_create_socket_array()` when updating an
     *        existing cache entry's `Array` descriptor in place.
     */
    void update_value(size_t id, jl_value_t* v);

    static void init(size_t size = 0);
    static Ref_keeper* get();
};
}
}

#endif /* SPU_JULIA */
#endif /* REF_KEEPER */
