#ifndef NEMO_BITOPS_H
#define NEMO_BITOPS_H

#include <limits.h>
#include <config.h>

/* Compute leading zeros for type T which should have B bits.
 *
 * This could be done faster using one of the other bit-twiddling hacks from
 * http://graphics.stanford.edu/~seander/bithacks.html */
template<typename T, int B>
int
clzN(T v)
{
	uint r = 0;
	while (v >>= 1) {
		r++;
	}
	return (B - 1) - r;
}


/* Count leading zeros in 64-bit word. Unfortunately the gcc builtin to deal
 * with this is not explicitly 64 bit. Instead it is defined for long long. In
 * C99 this is required to be /at least/ 64 bits. However, we require it to be
 * /exactly/ 64 bits. */
#if LLONG_MAX == 9223372036854775807 && defined(HAVE_BUILTIN_CLZLL)
inline int clz64(uint64_t val) { return __builtin_clzll(val); }
#else
#warning "__builtint_clzll not defined or long long is not 64 bit. Using slow clzll instead"
inline int clz64(uint64_t val) { return clzN<uint64_t, 64>(val); }
#endif


/* Ditto for 32 bits */
#if UINT_MAX == 4294967295U && defined(HAVE_BUILTIN_CLZ)
inline int clz32(uint32_t val) { return __builtin_clz(val); }
#else
#warning "__builtin_clz not defined or long int is not 32 bit. Using slow clzl"
inline int clz32(uint32_t val) { return clzN<uint32_t, 32>(val); }
#endif // LONG_MAX


/* Count trailing zeros. This should work even if long long is greater than
 * 64-bit. The uint64_t will be safely extended to the appropriate length */
inline int ctz64(uint64_t val) { return __builtin_ctzll(val); }


/* compute the next highest power of 2 of 32-bit v. From "bit-twiddling hacks".  */
inline
uint32_t
ceilPowerOfTwo(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}



#endif // BITOPS_H
