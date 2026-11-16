#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <math.h>

static uint64_t rng_state = 88172645463325252ULL;

static uint64_t rng_next(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ULL;
}

static float rand_uniform(void) { return ((rng_next() >> 11) + 0.5f) * (1.0f / 9007199254740992.0f); }

static float rand_normal(void) {
    float u1 = rand_uniform(), u2 = rand_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

#endif
