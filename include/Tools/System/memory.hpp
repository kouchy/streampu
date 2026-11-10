/*!
 * \file
 * \brief Memory allocation utilities.
 */
#ifndef SYSTEM_MEMORY_HPP__
#define SYSTEM_MEMORY_HPP__

#include <cstddef>
#include <new>

// Portable C++11 type/member alignment macro (`SPU_ALIGNAS(N)`) supporting all major compilers:
// - MSVC (`_MSC_VER`): uses `__declspec(align(N))` (which supports power-of-two alignments up to 8192 bytes even on
//   older MSVC toolsets where C++11 `alignas` had restrictions).
// - GCC, Clang, and Intel C++ (`__GNUC__`, `__clang__`, `__INTEL_COMPILER` classic `icpc`, `__INTEL_LLVM_COMPILER`
//   `icx`): uses `__attribute__((aligned(N)))`.
// - Standard C++11 fallback: `alignas(N)`.
// Note: In C++11 (prior to C++17 `P0035R4` over-aligned `operator new(size_t, std::align_val_t)`), the default global
// `::operator new` and `std::allocator<T>` only guarantee `alignof(std::max_align_t)` (8 or 16 bytes) on the heap.
// Consequently, any heap-allocated structure or `std::vector` relying on `SPU_ALIGNAS(128)` MUST be allocated via
// `tools::mem_alloc` / `tools::aligned_allocator<T>` (or overload `operator new` / `operator delete` to delegate to
// `tools::mem_alloc` / `tools::mem_free`).
#if !defined(SPU_ALIGNAS)
#if defined(_MSC_VER)
#define SPU_ALIGNAS(N) __declspec(align(N))
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define SPU_ALIGNAS(N) __attribute__((aligned(N)))
#elif __cplusplus >= 201103L
#define SPU_ALIGNAS(N) alignas(N)
#else
#define SPU_ALIGNAS(N)
#endif
#endif

namespace spu
{
namespace tools
{
void*
mem_alloc(std::size_t size);
void
mem_free(void* ptr);

template<typename T>
T*
buffer_alloc(std::size_t nData)
{
    if (nData == 0) return nullptr;
    T* const ptr = reinterpret_cast<T*>(mem_alloc(nData * sizeof(T)));
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

template<typename T>
void
buffer_free(T* ptr)
{
    mem_free(ptr);
}

template<class T>
struct aligned_allocator
{
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<class U>
    struct rebind
    {
        typedef aligned_allocator<U> other;
    };

    aligned_allocator() noexcept {}
    template<class C>
    aligned_allocator(const aligned_allocator<C>& /*other*/) noexcept
    {
    }
    T* allocate(std::size_t n) { return buffer_alloc<T>(n); }
    void deallocate(T* p, std::size_t /*n*/) noexcept { buffer_free<T>(p); }
};

// Returns true if and only if storage allocated from ma1 can be
// deallocated from ma2, and vice versa. Always returns true for
// stateless allocators.
template<class C1, class C2>
bool
operator==(const aligned_allocator<C1>& /*ma1*/, const aligned_allocator<C2>& /*ma2*/) noexcept
{
    return true;
}

template<class C1, class C2>
bool
operator!=(const aligned_allocator<C1>& ma1, const aligned_allocator<C2>& ma2) noexcept
{
    return !(ma1 == ma2);
}

}
}

#endif // SYSTEM_MEMORY_HPP__
