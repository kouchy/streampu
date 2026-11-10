#include <cstdint>
#include <cstdlib>
#if defined(_MSC_VER) || defined(_WIN32)
#include <malloc.h>
#endif

#include "Tools/System/memory.hpp"

namespace spu
{
namespace tools
{

// Cross-platform aligned heap allocation helper supporting:
// 1. Windows / MSVC (`_MSC_VER` / `_WIN32`): `_aligned_malloc(size, alignment)` paired with `_aligned_free(ptr)`
//    (since MSVC's Universal CRT does not implement C11/C++17 `std::aligned_alloc` because `free()` cannot release
//    aligned blocks on Windows).
// 2. POSIX (`Linux`, `macOS`, `BSD`, `Unix`): `posix_memalign(&ptr, alignment, size)` paired with `std::free(ptr)`
//    (available in C++11 without requiring `size` to be rounded up to a multiple of `alignment`).
// 3. C++17 standard fallback: `std::aligned_alloc(alignment, aligned_size)` with `size` rounded up to a multiple of
//    `alignment` as required by the C11/C++17 specification.
// 4. Generic portable fallback: over-allocates `size + alignment` bytes via `std::malloc` and stores the original
//    pointer right before the aligned return address (`addr - sizeof(void*)`).
static void*
mem_alloc_aligned(std::size_t alignment, std::size_t size)
{
    if (size == 0) return nullptr;
    void* ptr = nullptr;
#if defined(_MSC_VER) || defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#elif (defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)) || defined(__APPLE__) || defined(__linux__) ||        \
  defined(__unix__)
    if (posix_memalign(&ptr, alignment, size) != 0) ptr = nullptr;
#elif __cplusplus >= 201703L
    const std::size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    ptr = std::aligned_alloc(alignment, aligned_size);
#else
    {
        if (alignment < sizeof(void*))
        {
            alignment = sizeof(void*);
        }
        void* raw_ptr = std::malloc(size + alignment);
        if (raw_ptr != nullptr)
        {
            uintptr_t addr = (uintptr_t)raw_ptr;
            addr = (addr & ~(alignment - 1)) + alignment;
            *(void**)(addr - sizeof(void*)) = raw_ptr;
            ptr = (void*)addr;
        }
    }
#endif
    return ptr;
}

void*
mem_alloc(std::size_t size)
{
    // 128-byte alignment satisfies both 512-bit SIMD vectors (64B) and 128-byte cache-line / coherence-granule
    // isolation on x86-64 (spatial adjacent-line prefetcher pair) and ARM64 / Apple Silicon (128B L2/SLC lines).
    std::size_t alignment = 128;
    return mem_alloc_aligned(alignment, size);
}

void
mem_free(void* ptr)
{
    if (ptr == nullptr) return;
#if defined(_MSC_VER) || defined(_WIN32)
    _aligned_free(ptr);
#elif (defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)) || defined(__APPLE__) || defined(__linux__) ||        \
  defined(__unix__) || (__cplusplus >= 201703L)
    std::free(ptr);
#else
    {
        uintptr_t addr = (uintptr_t)ptr;
        void* raw_ptr = *(void**)(addr - sizeof(void*));
        std::free(raw_ptr);
    }
#endif
}

}
}
