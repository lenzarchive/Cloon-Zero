#ifndef SAMPLER_H
#define SAMPLER_H

#include "config.h"
#include "rng.h"

typedef struct {
    float v;
    int i;
} Cand;

static int cand_cmp(const void *a, const void *b) {
    float x = ((const Cand *)a)->v, y = ((const Cand *)b)->v;
    return (x < y) - (x > y);
}

// Order: repetition penalty, n-gram ban, temperature, top-k, min-p (relative to the best token), top-p.
static int sample_token(float *logits, int V, const int *recent, int n_recent, const unsigned char *banned,
                        float temp, int top_k, float top_p, float min_p, float rep) {
    if (rep > 1.0f && n_recent > 0) {
        unsigned char *mark = XALLOC(unsigned char, V);
        for (int r = 0; r < n_recent; r++) {
            int id = recent[r];
            if (mark[id]) continue;
            mark[id] = 1;
            logits[id] = logits[id] > 0.0f ? logits[id] / rep : logits[id] * rep;
        }
        free(mark);
    }
    if (banned) {
        int nb = 0;
        for (int i = 0; i < V; i++) nb += banned[i] != 0;
        if (nb < V)
            for (int i = 0; i < V; i++)
                if (banned[i]) logits[i] = -1e30f;
    }
    Cand *c = XALLOC(Cand, V);
    for (int i = 0; i < V; i++) {
        c[i].v = logits[i];
        c[i].i = i;
    }
    qsort(c, (size_t)V, sizeof(Cand), cand_cmp);
    if (temp < 0.01f) {
        int best = c[0].i;
        free(c);
        return best;
    }
    int k = (top_k > 0 && top_k < V) ? top_k : V;
    float sum = 0.0f;

    // c[0].v is overwritten in the loop below, so keep the best logit separately.
    const float best_logit = c[0].v;
    for (int i = 0; i < k; i++) {
        c[i].v = expf((c[i].v - best_logit) / temp);
        sum += c[i].v;
    }
    for (int i = 0; i < k; i++) c[i].v /= sum;
    if (min_p > 0.0f) {
        float floor_p = min_p * c[0].v;
        int keep = 1;
        while (keep < k && c[keep].v >= floor_p) keep++;
        k = keep;
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        int keep = k;
        for (int i = 0; i < k; i++) {
            cum += c[i].v;
            if (cum >= top_p) {
                keep = i + 1;
                break;
            }
        }
        k = keep;
    }
    sum = 0.0f;
    for (int i = 0; i < k; i++) sum += c[i].v;
    for (int i = 0; i < k; i++) c[i].v /= sum;
    float r = rand_uniform(), acc = 0.0f;
    int pick = c[k - 1].i;
    for (int i = 0; i < k; i++) {
        acc += c[i].v;
        if (r < acc) {
            pick = c[i].i;
            break;
        }
    }
    free(c);
    return pick;
}

// Flags every token that would complete an n-gram already present in seq.
static void ban_ngrams(const int *seq, size_t len, int N, unsigned char *banned, int V) {
    memset(banned, 0, (size_t)V);
    if (N < 2 || len < (size_t)N) return;
    const int *ctx = seq + len - (size_t)(N - 1);
    for (size_t i = 0; i + (size_t)N <= len; i++) {
        int match = 1;
        for (int k = 0; k < N - 1; k++) {
            if (seq[i + (size_t)k] != ctx[k]) {
                match = 0;
                break;
            }
        }
        if (match) banned[seq[i + (size_t)(N - 1)]] = 1;
    }
}

static int count_repeated_ngrams(const int *seq, int len, int N) {
    int rep_count = 0;
    for (int i = 0; i + N <= len; i++)
        for (int j = 0; j < i; j++) {
            int same = 1;
            for (int k = 0; k < N; k++)
                if (seq[i + k] != seq[j + k]) {
                    same = 0;
                    break;
                }
            if (same) {
                rep_count++;
                break;
            }
        }
    return rep_count;
}

static int run_selftest(void) {
    const int V = 64, STEPS = 300;
    int fails = 0;
    int *seq = XALLOC(int, STEPS + 8);
    float *lg = XALLOC(float, V);
    unsigned char *banned = XALLOC(unsigned char, V);
    for (int mode = 0; mode < 3; mode++) {
        const int N = mode == 0 ? 0 : 4;
        const float rep_pen = mode == 2 ? 1.3f : 1.0f;
        int len = 1;
        seq[0] = 0;
        for (int i = 0; i < STEPS; i++) {
            for (int v = 0; v < V; v++) lg[v] = -2.0f + 0.01f * (float)(v % 7);
            lg[(seq[len - 1] + 1) % 5] = 12.0f;
            ban_ngrams(seq, (size_t)len, N, banned, V);
            int nr = len < REP_WINDOW ? len : REP_WINDOW;
            seq[len] = sample_token(lg, V, seq + len - nr, nr, banned, 0.0f, 40, 0.9f, 0.05f, rep_pen);
            len++;
        }
        int repeats = count_repeated_ngrams(seq, len, 4);
        int distinct = 0;
        unsigned char seen[64] = {0};
        for (int i = 0; i < len; i++) if (!seen[seq[i]]) { seen[seq[i]] = 1; distinct++; }
        int ok = mode == 0 ? repeats > 100 : repeats == 0;
        printf("%s  greedy loop test, %s: repeated 4-grams %d of %d, distinct tokens %d\n", ok ? "PASS" : "FAIL",
               mode == 0 ? "guards off (must loop)" : (mode == 1 ? "no_repeat_ngram=4" : "ngram=4 + rep 1.3"), repeats,
               len - 3, distinct);
        fails += !ok;
    }

    int tail_hits = 0;
    for (int i = 0; i < 2000; i++) {
        for (int v = 0; v < V; v++) lg[v] = 0.0f;
        lg[3] = 6.0f;
        int t = sample_token(lg, V, NULL, 0, NULL, 1.0f, 0, 1.0f, 0.05f, 1.0f);
        tail_hits += t != 3;
    }
    printf("%s  min_p=0.05 removes the flat tail: %d/2000 off-peak samples\n", tail_hits == 0 ? "PASS" : "FAIL", tail_hits);
    fails += tail_hits != 0;
    int tail_hits0 = 0;
    for (int i = 0; i < 2000; i++) {
        for (int v = 0; v < V; v++) lg[v] = 0.0f;
        lg[3] = 6.0f;
        int t = sample_token(lg, V, NULL, 0, NULL, 1.0f, 0, 1.0f, 0.0f, 1.0f);
        tail_hits0 += t != 3;
    }
    printf("%s  without min_p the tail does get sampled (sanity): %d/2000\n", tail_hits0 > 0 ? "PASS" : "FAIL", tail_hits0);
    fails += tail_hits0 == 0;

    int top_hits = 0;
    for (int i = 0; i < 2000; i++) {
        for (int v = 0; v < V; v++) lg[v] = -3.0f;
        lg[5] = 10.0f;
        lg[9] = 8.0f;
        top_hits += sample_token(lg, V, NULL, 0, NULL, 0.8f, 40, 1.0f, 0.0f, 1.0f) == 5;
    }
    printf("%s  sampler keeps the ranking: top token chosen %d/2000 (expected ~1840)\n",
           top_hits > 1700 && top_hits < 1950 ? "PASS" : "FAIL", top_hits);
    fails += !(top_hits > 1700 && top_hits < 1950);

    int s2[] = {5, 6, 7, 8, 5, 6, 7};
    ban_ngrams(s2, 7, 4, banned, V);
    int ok2 = banned[8] == 1;
    int others = 0;
    for (int v = 0; v < V; v++) if (v != 8 && banned[v]) others++;
    printf("%s  ban_ngrams flags only the repeated continuation (flagged 8: %d, others: %d)\n",
           ok2 && others == 0 ? "PASS" : "FAIL", banned[8], others);
    fails += !(ok2 && others == 0);
    free(seq);
    free(lg);
    free(banned);
    printf(fails ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return fails ? 2 : 0;
}

#endif
