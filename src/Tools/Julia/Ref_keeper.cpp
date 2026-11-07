#include "Tools/Exception/julia_error/julia_error.hpp"
#include "Tools/Exception/runtime_error/runtime_error.hpp"
#ifdef SPU_JULIA

#include <cstddef>
#include <iostream>
#include <julia.h>

#include "Tools/Julia/Julia.hpp"
#include "Tools/Julia/Ref_keeper.hpp"

namespace spu
{
namespace tools
{
Ref_keeper::~Ref_keeper()
{
    delete[] marks;
}

size_t
Ref_keeper::get_size() const
{
    return this->size;
}

jl_value_t*
Ref_keeper::get_res() const
{
    return Julia::unref(last_result);
}

jl_value_t*
Ref_keeper::call0(jl_function_t* f)
{
    return jl_call2(Julia::get_call(), last_result, f);
}

jl_value_t*
Ref_keeper::call1(jl_function_t* f, jl_value_t* a)
{
    return jl_call3(Julia::get_call(), last_result, f, a);
}

static inline jl_value_t*
_call(jl_function_t* call, jl_value_t* ref, jl_function_t* f, jl_value_t** args, size_t n)
{
    jl_value_t** args2;
    JL_GC_PUSHARGS(args2, n + 2);
    args2[0] = ref;
    args2[1] = f;
    for (size_t i = 0; i < n; i++)
    {
        args2[i + 2] = args[i];
    }
    jl_value_t* res = jl_call(call, args2, n + 2);
    JL_GC_POP();
    return res;
}

jl_value_t*
Ref_keeper::call(jl_function_t* f, jl_value_t** args, size_t n)
{
    jl_value_t** args2;
    JL_GC_PUSHARGS(args2, n + 2);
    args2[0] = last_result;
    args2[1] = f;
    for (size_t i = 0; i < n; i++)
    {
        args2[i + 2] = args[i];
    }
    jl_value_t* res = jl_call(Julia::get_call(), args2, n + 2);
    JL_GC_POP();
    return res;
}

jl_value_t*
Ref_keeper::safe_call(jl_function_t* f, jl_value_t** args, size_t n)
{
    jl_value_t* res = _call(Julia::get_safe_call(), last_result, f, args, n);
    if (jl_unbox_bool(jl_get_nth_field(res, 1)))
        throw julia_error(__FILE__, __LINE__, __func__, jl_string_ptr(jl_get_nth_field(res, 3)));
    return jl_get_nth_field(res, 0);
}

jl_value_t*
Ref_keeper::eval(std::string code)
{
    size_t len = code.length() + 1;
    char* s = (char*)malloc(sizeof(char) * len);
    size_t i = 0;
    for (char c : code)
    {
        s[i] = c;
        i++;
    }
    static jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_char_type, 1);
    jl_array_t* c = jl_ptr_to_array_1d(array_type, s, len, false);
    jl_value_t* res = this->call1(Julia::get_eval(), (jl_value_t*)c);
    free(s);
    return res;
}

jl_value_t*
Ref_keeper::unsafe_eval(std::string code)
{
    jl_value_t* res = jl_eval_string(code.c_str());
    jl_call2(Julia::get_assign(), last_result, res);
    return res;
}

size_t
Ref_keeper::mark(jl_value_t* v)
{
    size_t r;
    if (freelist.empty())
    {
        if (cpt < size + 3)
        {
            r = cpt;
            cpt++;
        }
        else if (jl_pgcstack == (jl_gcframe_t*)this->marks)
        {
            // All `size` slots are currently occupied and `this->marks` is at the top of the thread's
            // Julia GC frame stack (`jl_pgcstack`). Dynamically double the GC frame capacity (2x) in place:
            // allocate a larger `new_marks` array, update the encoded GC root count in `new_marks[0]`,
            // preserve the previous stack link in `new_marks[1]`, copy all active roots, initialize the
            // newly added slots with `last_result` (valid non-null Julia pointer), and atomically redirect
            // `jl_pgcstack` to `new_marks` before freeing the old array.
            size_t new_size = this->size * 2;
            jl_value_t** new_marks = new jl_value_t*[new_size + 3];
            new_marks[0] = (jl_value_t*)JL_GC_ENCODE_PUSHARGS(new_size + 1);
            new_marks[1] = this->marks[1];
            for (size_t i = 2; i < this->size + 3; i++)
                new_marks[i] = this->marks[i];
            for (size_t i = this->size + 3; i < new_size + 3; i++)
                new_marks[i] = this->last_result;
            jl_pgcstack = (jl_gcframe_t*)new_marks;
            delete[] this->marks;
            this->marks = new_marks;
            this->size = new_size;
            r = cpt++;
        }
        else
        {
            throw runtime_error(__FILE__, __LINE__, __func__, "No space available for marks in Ref_keeper.");
        }
    }
    else
    {
        r = freelist.back();
        freelist.pop_back();
    }
    marks[r] = v;
    return r;
}

void
Ref_keeper::update_value(size_t id, jl_value_t* v)
{
    // Slots 0 and 1 hold the `jl_gcframe_t` header (`nroots` and `prev`), and slot 2 holds `last_result`.
    // Only user slots (`id >= 3`) may be updated in place.
    if (id <= 2)
    {
        throw runtime_error(__FILE__, __LINE__, __func__, "Trying to change Ref_keeper's protected value.");
    }
    marks[id] = v;
}

void
Ref_keeper::unmark(size_t id)
{
    marks[id] = last_result;
    freelist.push_back(id);
}

void
Ref_keeper::init(size_t size)
{
    Ref_keeper* singleton = get();
    if (singleton->marks != nullptr) return;

    if (size == 0) size = Julia::get_n_proxies_init();

    singleton->last_result = jl_call0(Julia::get_mk_ref());
    singleton->size = size;
    singleton->marks = new jl_value_t*[size + 3];
    singleton->cpt = 3; // 0 and 1 for gcframe, 2 for last_result
    singleton->marks[0] = (jl_value_t*)JL_GC_ENCODE_PUSHARGS(size + 1);
    singleton->marks[1] = (jl_value_t*)jl_pgcstack;
    for (size_t i = 0; i < size + 1; i++)
        singleton->marks[i + 2] = singleton->last_result;
    jl_pgcstack = (jl_gcframe_t*)singleton->marks;
}

Ref_keeper*
Ref_keeper::get()
{
    static thread_local Ref_keeper singleton;
    return &singleton;
}

}
}

#endif /* SPU_JULIA */
