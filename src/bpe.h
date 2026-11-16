#ifndef BPE_H
#define BPE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BPE_MAX_VOCAB 8192
#define BPE_MAX_PIECE 48

typedef struct {
    int V;
    int n_base;
    int stoi[256];
    uint8_t itos[256];
    uint16_t left[BPE_MAX_VOCAB];
    uint16_t right[BPE_MAX_VOCAB];
    int32_t *mtab;
    uint32_t mcap;
    uint8_t *dpool;
    int32_t doff[BPE_MAX_VOCAB];
    int32_t dlen[BPE_MAX_VOCAB];
} Tokenizer;

typedef struct {
    uint8_t *pool;
    size_t plen, pcap;
    int32_t *off, *len;
    int64_t *cnt;
    uint32_t *hash;
    int n, cap;
    int32_t *tab;
    uint32_t tcap;
} PieceMap;

typedef struct {
    int32_t cnt, pair;
} HeapItem;

static void *bpe_alloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return p;
}

static void *bpe_realloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return q;
}

static int bpe_class(uint8_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128) return 1;
    if (c >= '0' && c <= '9') return 2;
    if (c == ' ') return 3;
    return 4;
}

static size_t bpe_next_piece(const uint8_t *s, size_t n, size_t i) {
    size_t j = i;
    int c = bpe_class(s[j]);
    if (c == 3) {
        if (i + 1 < n && (bpe_class(s[i + 1]) == 1 || bpe_class(s[i + 1]) == 2)) {
            j = i + 1;
            c = bpe_class(s[j]);
        } else {
            return 1;
        }
    }
    size_t k = j + 1;
    if (c == 1 || c == 2) {
        while (k < n && bpe_class(s[k]) == c) k++;
    } else {
        while (k < n && s[k] == s[j]) k++;
    }
    size_t len = k - i;
    return len > BPE_MAX_PIECE ? BPE_MAX_PIECE : len;
}

static uint32_t bpe_hash_bytes(const uint8_t *p, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

static void pm_init(PieceMap *pm) {
    memset(pm, 0, sizeof(*pm));
    pm->pcap = 1 << 16;
    pm->pool = (uint8_t *)bpe_alloc(pm->pcap);
    pm->cap = 1 << 12;
    pm->off = (int32_t *)bpe_alloc(sizeof(int32_t) * pm->cap);
    pm->len = (int32_t *)bpe_alloc(sizeof(int32_t) * pm->cap);
    pm->cnt = (int64_t *)bpe_alloc(sizeof(int64_t) * pm->cap);
    pm->hash = (uint32_t *)bpe_alloc(sizeof(uint32_t) * pm->cap);
    pm->tcap = 1 << 13;
    pm->tab = (int32_t *)bpe_alloc(sizeof(int32_t) * pm->tcap);
    memset(pm->tab, 0xFF, sizeof(int32_t) * pm->tcap);
}

static void pm_free(PieceMap *pm) {
    free(pm->pool);
    free(pm->off);
    free(pm->len);
    free(pm->cnt);
    free(pm->hash);
    free(pm->tab);
}

static void pm_rehash(PieceMap *pm) {
    free(pm->tab);
    pm->tcap *= 2;
    pm->tab = (int32_t *)bpe_alloc(sizeof(int32_t) * pm->tcap);
    memset(pm->tab, 0xFF, sizeof(int32_t) * pm->tcap);
    for (int i = 0; i < pm->n; i++) {
        uint32_t h = pm->hash[i] & (pm->tcap - 1);
        while (pm->tab[h] >= 0) h = (h + 1) & (pm->tcap - 1);
        pm->tab[h] = i;
    }
}

static int pm_get(PieceMap *pm, const uint8_t *p, int len, int *is_new) {
    uint32_t hv = bpe_hash_bytes(p, len);
    uint32_t h = hv & (pm->tcap - 1);
    while (pm->tab[h] >= 0) {
        int idx = pm->tab[h];
        if (pm->hash[idx] == hv && pm->len[idx] == len && memcmp(pm->pool + pm->off[idx], p, (size_t)len) == 0) {
            *is_new = 0;
            return idx;
        }
        h = (h + 1) & (pm->tcap - 1);
    }
    if (pm->n == pm->cap) {
        pm->cap *= 2;
        pm->off = (int32_t *)bpe_realloc(pm->off, sizeof(int32_t) * pm->cap);
        pm->len = (int32_t *)bpe_realloc(pm->len, sizeof(int32_t) * pm->cap);
        pm->cnt = (int64_t *)bpe_realloc(pm->cnt, sizeof(int64_t) * pm->cap);
        pm->hash = (uint32_t *)bpe_realloc(pm->hash, sizeof(uint32_t) * pm->cap);
    }
    while (pm->plen + (size_t)len > pm->pcap) {
        pm->pcap *= 2;
        pm->pool = (uint8_t *)bpe_realloc(pm->pool, pm->pcap);
    }
    int idx = pm->n++;
    memcpy(pm->pool + pm->plen, p, (size_t)len);
    pm->off[idx] = (int32_t)pm->plen;
    pm->len[idx] = len;
    pm->cnt[idx] = 0;
    pm->hash[idx] = hv;
    pm->plen += (size_t)len;
    pm->tab[h] = idx;
    if ((uint32_t)pm->n * 2 > pm->tcap) pm_rehash(pm);
    *is_new = 1;
    return idx;
}

static void heap_push(HeapItem **h, size_t *n, size_t *cap, HeapItem it) {
    if (*n == *cap) {
        *cap *= 2;
        *h = (HeapItem *)bpe_realloc(*h, sizeof(HeapItem) * *cap);
    }
    HeapItem *a = *h;
    size_t i = (*n)++;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        int better = it.cnt > a[p].cnt || (it.cnt == a[p].cnt && it.pair < a[p].pair);
        if (!better) break;
        a[i] = a[p];
        i = p;
    }
    a[i] = it;
}

static HeapItem heap_pop(HeapItem *a, size_t *n) {
    HeapItem top = a[0];
    HeapItem last = a[--(*n)];
    size_t i = 0;
    for (;;) {
        size_t c = 2 * i + 1;
        if (c >= *n) break;
        if (c + 1 < *n) {
            int right_better = a[c + 1].cnt > a[c].cnt || (a[c + 1].cnt == a[c].cnt && a[c + 1].pair < a[c].pair);
            if (right_better) c++;
        }
        int better = a[c].cnt > last.cnt || (a[c].cnt == last.cnt && a[c].pair < last.pair);
        if (!better) break;
        a[i] = a[c];
        i = c;
    }
    if (*n > 0) a[i] = last;
    return top;
}

static int tok_pair_id(const Tokenizer *T, int a, int b) {
    uint32_t h = ((uint32_t)a * 2654435761u ^ (uint32_t)b * 40503u) & (T->mcap - 1);
    for (;;) {
        int id = T->mtab[h];
        if (id < 0) return -1;
        if (T->left[id] == a && T->right[id] == b) return id;
        h = (h + 1) & (T->mcap - 1);
    }
}

static void tok_finalize(Tokenizer *T) {
    int n_merge = T->V - T->n_base;
    uint32_t cap = 16;
    while (cap < (uint32_t)(4 * n_merge + 16)) cap *= 2;
    free(T->mtab);
    T->mcap = cap;
    T->mtab = (int32_t *)bpe_alloc(sizeof(int32_t) * cap);
    memset(T->mtab, 0xFF, sizeof(int32_t) * cap);
    for (int id = T->n_base; id < T->V; id++) {
        uint32_t h = ((uint32_t)T->left[id] * 2654435761u ^ (uint32_t)T->right[id] * 40503u) & (cap - 1);
        while (T->mtab[h] >= 0) h = (h + 1) & (cap - 1);
        T->mtab[h] = id;
    }
    size_t total = 0;
    for (int id = 0; id < T->V; id++) {
        if (id < T->n_base) {
            T->dlen[id] = 1;
        } else {
            T->dlen[id] = T->dlen[T->left[id]] + T->dlen[T->right[id]];
        }
        total += (size_t)T->dlen[id];
    }
    free(T->dpool);
    T->dpool = (uint8_t *)bpe_alloc(total);
    size_t off = 0;
    for (int id = 0; id < T->V; id++) {
        T->doff[id] = (int32_t)off;
        if (id < T->n_base) {
            T->dpool[off] = T->itos[id];
        } else {
            memcpy(T->dpool + off, T->dpool + T->doff[T->left[id]], (size_t)T->dlen[T->left[id]]);
            memcpy(T->dpool + off + T->dlen[T->left[id]], T->dpool + T->doff[T->right[id]],
                   (size_t)T->dlen[T->right[id]]);
        }
        off += (size_t)T->dlen[id];
    }
}

static void tok_free(Tokenizer *T) {
    free(T->mtab);
    free(T->dpool);
    T->mtab = NULL;
    T->dpool = NULL;
}

static int tok_train(Tokenizer *T, const uint8_t *text, size_t len, int target, int verbose) {
    memset(T, 0, sizeof(*T));
    int seen[256] = {0};
    for (size_t i = 0; i < len; i++) seen[text[i]] = 1;
    for (int c = 0; c < 256; c++) {
        T->stoi[c] = -1;
        if (seen[c]) {
            T->stoi[c] = T->n_base;
            T->itos[T->n_base++] = (uint8_t)c;
        }
    }
    if (target > BPE_MAX_VOCAB) target = BPE_MAX_VOCAB;
    if (target < T->n_base) target = T->n_base;
    const int Vm = target;

    PieceMap pm;
    pm_init(&pm);
    for (size_t i = 0; i < len;) {
        size_t l = bpe_next_piece(text, len, i);
        int is_new;
        int idx = pm_get(&pm, text + i, (int)l, &is_new);
        pm.cnt[idx]++;
        i += l;
    }
    const int W = pm.n;
    size_t total = 0;
    for (int w = 0; w < W; w++) total += (size_t)pm.len[w];
    uint16_t *sym = (uint16_t *)bpe_alloc(sizeof(uint16_t) * total);
    int32_t *woff = (int32_t *)bpe_alloc(sizeof(int32_t) * (size_t)W);
    int32_t *wlen = (int32_t *)bpe_alloc(sizeof(int32_t) * (size_t)W);
    int32_t *wcnt = (int32_t *)bpe_alloc(sizeof(int32_t) * (size_t)W);
    size_t pos = 0;
    for (int w = 0; w < W; w++) {
        woff[w] = (int32_t)pos;
        wlen[w] = pm.len[w];
        wcnt[w] = pm.cnt[w] > INT32_MAX ? INT32_MAX : (int32_t)pm.cnt[w];
        for (int j = 0; j < pm.len[w]; j++) sym[pos++] = (uint16_t)T->stoi[pm.pool[pm.off[w] + j]];
    }
    if (verbose) printf("BPE: %d unique pieces, %d base symbols\n", W, T->n_base);
    pm_free(&pm);

    size_t pairs = (size_t)Vm * (size_t)Vm;
    int32_t *pc = (int32_t *)calloc(pairs, sizeof(int32_t));
    int32_t *head = (int32_t *)bpe_alloc(sizeof(int32_t) * pairs);
    if (!pc) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    memset(head, 0xFF, sizeof(int32_t) * pairs);
    size_t ix_cap = 1 << 20, ix_n = 0;
    int32_t *ix_w = (int32_t *)bpe_alloc(sizeof(int32_t) * ix_cap);
    int32_t *ix_next = (int32_t *)bpe_alloc(sizeof(int32_t) * ix_cap);

#define IX_ADD(p, w)                                                                  \
    do {                                                                              \
        if (head[p] < 0 || ix_w[head[p]] != (w)) {                                    \
            if (ix_n == ix_cap) {                                                     \
                ix_cap *= 2;                                                          \
                ix_w = (int32_t *)bpe_realloc(ix_w, sizeof(int32_t) * ix_cap);        \
                ix_next = (int32_t *)bpe_realloc(ix_next, sizeof(int32_t) * ix_cap);  \
            }                                                                         \
            ix_w[ix_n] = (w);                                                         \
            ix_next[ix_n] = head[p];                                                  \
            head[p] = (int32_t)ix_n++;                                                \
        }                                                                             \
    } while (0)

    for (int w = 0; w < W; w++) {
        const uint16_t *s = sym + woff[w];
        for (int i = 0; i + 1 < wlen[w]; i++) {
            int p = s[i] * Vm + s[i + 1];
            pc[p] += wcnt[w];
            IX_ADD(p, w);
        }
    }

    size_t h_cap = 1 << 16, h_n = 0;
    HeapItem *heap = (HeapItem *)bpe_alloc(sizeof(HeapItem) * h_cap);
    for (int a = 0; a < T->n_base; a++) {
        for (int b = 0; b < T->n_base; b++) {
            int p = a * Vm + b;
            if (pc[p] > 0) {
                HeapItem it = {pc[p], p};
                heap_push(&heap, &h_n, &h_cap, it);
            }
        }
    }

    size_t t_cap = 1 << 12, t_n = 0;
    int32_t *touched = (int32_t *)bpe_alloc(sizeof(int32_t) * t_cap);
    int m = T->n_base;
    while (m < target && h_n > 0) {
        HeapItem top = heap_pop(heap, &h_n);
        int cur = pc[top.pair];
        if (cur <= 0) continue;
        if (cur != top.cnt) {
            if (cur < top.cnt) {
                HeapItem it = {cur, top.pair};
                heap_push(&heap, &h_n, &h_cap, it);
            }
            continue;
        }
        if (cur < 2) break;
        const int a = top.pair / Vm, b = top.pair % Vm;
        T->left[m] = (uint16_t)a;
        T->right[m] = (uint16_t)b;
        t_n = 0;
        for (int32_t e = head[top.pair]; e >= 0; e = ix_next[e]) {
            const int w = ix_w[e];
            uint16_t *s = sym + woff[w];
            int L = wlen[w];
            int found = 0;
            for (int i = 0; i + 1 < L; i++) {
                if (s[i] == a && s[i + 1] == b) {
                    found = 1;
                    break;
                }
            }
            if (!found) continue;
            const int c = wcnt[w];
            for (int i = 0; i + 1 < L; i++) pc[s[i] * Vm + s[i + 1]] -= c;
            int ww = 0;
            for (int i = 0; i < L;) {
                if (i + 1 < L && s[i] == a && s[i + 1] == b) {
                    s[ww++] = (uint16_t)m;
                    i += 2;
                } else {
                    s[ww++] = s[i++];
                }
            }
            L = ww;
            wlen[w] = L;
            for (int i = 0; i + 1 < L; i++) {
                int p = s[i] * Vm + s[i + 1];
                if (s[i] == m || s[i + 1] == m) {
                    if (pc[p] == 0) {
                        if (t_n == t_cap) {
                            t_cap *= 2;
                            touched = (int32_t *)bpe_realloc(touched, sizeof(int32_t) * t_cap);
                        }
                        touched[t_n++] = p;
                    }
                    IX_ADD(p, w);
                }
                pc[p] += c;
            }
        }
        for (size_t i = 0; i < t_n; i++) {
            if (pc[touched[i]] > 0) {
                HeapItem it = {pc[touched[i]], touched[i]};
                heap_push(&heap, &h_n, &h_cap, it);
            }
        }
        m++;
        if (verbose && (m % 500 == 0 || m == target)) printf("BPE: %d / %d symbols\n", m, target);
    }
#undef IX_ADD
    T->V = m;
    free(sym);
    free(woff);
    free(wlen);
    free(wcnt);
    free(pc);
    free(head);
    free(ix_w);
    free(ix_next);
    free(heap);
    free(touched);
    tok_finalize(T);
    return 0;
}

static int tok_encode_piece(const Tokenizer *T, const uint8_t *p, int n, uint16_t *out) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int id = T->stoi[p[i]];
        if (id >= 0) out[m++] = (uint16_t)id;
    }
    for (;;) {
        int best = -1;
        for (int i = 0; i + 1 < m; i++) {
            int id = tok_pair_id(T, out[i], out[i + 1]);
            if (id >= 0 && (best < 0 || id < best)) best = id;
        }
        if (best < 0) break;
        int w = 0;
        for (int i = 0; i < m;) {
            if (i + 1 < m && out[i] == T->left[best] && out[i + 1] == T->right[best]) {
                out[w++] = (uint16_t)best;
                i += 2;
            } else {
                out[w++] = out[i++];
            }
        }
        m = w;
    }
    return m;
}

static uint16_t *tok_encode(const Tokenizer *T, const uint8_t *text, size_t len, size_t *n_out) {
    size_t cap = len / 2 + 16, n = 0;
    uint16_t *out = (uint16_t *)bpe_alloc(sizeof(uint16_t) * cap);
    PieceMap pm;
    pm_init(&pm);
    size_t c_cap = 1 << 16, c_len = 0;
    uint16_t *cache = (uint16_t *)bpe_alloc(sizeof(uint16_t) * c_cap);
    int32_t *c_off = NULL, *c_cnt = NULL;
    int c_slots = 0;
    uint16_t tmp[BPE_MAX_PIECE + 1];
    for (size_t i = 0; i < len;) {
        size_t l = bpe_next_piece(text, len, i);
        int is_new;
        int idx = pm_get(&pm, text + i, (int)l, &is_new);
        if (idx >= c_slots) {
            int ns = c_slots ? c_slots * 2 : 4096;
            while (ns <= idx) ns *= 2;
            c_off = (int32_t *)bpe_realloc(c_off, sizeof(int32_t) * (size_t)ns);
            c_cnt = (int32_t *)bpe_realloc(c_cnt, sizeof(int32_t) * (size_t)ns);
            c_slots = ns;
        }
        if (is_new) {
            int k = tok_encode_piece(T, text + i, (int)l, tmp);
            while (c_len + (size_t)k > c_cap) {
                c_cap *= 2;
                cache = (uint16_t *)bpe_realloc(cache, sizeof(uint16_t) * c_cap);
            }
            memcpy(cache + c_len, tmp, sizeof(uint16_t) * (size_t)k);
            c_off[idx] = (int32_t)c_len;
            c_cnt[idx] = k;
            c_len += (size_t)k;
        }
        while (n + (size_t)c_cnt[idx] > cap) {
            cap *= 2;
            out = (uint16_t *)bpe_realloc(out, sizeof(uint16_t) * cap);
        }
        memcpy(out + n, cache + c_off[idx], sizeof(uint16_t) * (size_t)c_cnt[idx]);
        n += (size_t)c_cnt[idx];
        i += l;
    }
    pm_free(&pm);
    free(cache);
    free(c_off);
    free(c_cnt);
    *n_out = n;
    return out;
}

static size_t tok_decode_token(const Tokenizer *T, int id, uint8_t *out) {
    memcpy(out, T->dpool + T->doff[id], (size_t)T->dlen[id]);
    return (size_t)T->dlen[id];
}

static int tok_write(FILE *f, const Tokenizer *T) {
    uint32_t hdr[2] = {(uint32_t)T->V, (uint32_t)T->n_base};
    if (fwrite(hdr, sizeof(uint32_t), 2, f) != 2) return -1;
    if (fwrite(T->itos, 1, 256, f) != 256) return -1;
    int n_merge = T->V - T->n_base;
    if (n_merge > 0) {
        if (fwrite(T->left + T->n_base, sizeof(uint16_t), (size_t)n_merge, f) != (size_t)n_merge) return -1;
        if (fwrite(T->right + T->n_base, sizeof(uint16_t), (size_t)n_merge, f) != (size_t)n_merge) return -1;
    }
    return 0;
}

static int tok_read(FILE *f, Tokenizer *T) {
    memset(T, 0, sizeof(*T));
    uint32_t hdr[2];
    if (fread(hdr, sizeof(uint32_t), 2, f) != 2) return -1;
    if (hdr[0] > BPE_MAX_VOCAB || hdr[1] > 256 || hdr[1] == 0 || hdr[0] < hdr[1]) return -1;
    T->V = (int)hdr[0];
    T->n_base = (int)hdr[1];
    if (fread(T->itos, 1, 256, f) != 256) return -1;
    for (int c = 0; c < 256; c++) T->stoi[c] = -1;
    for (int i = 0; i < T->n_base; i++) T->stoi[T->itos[i]] = i;
    int n_merge = T->V - T->n_base;
    if (n_merge > 0) {
        if (fread(T->left + T->n_base, sizeof(uint16_t), (size_t)n_merge, f) != (size_t)n_merge) return -1;
        if (fread(T->right + T->n_base, sizeof(uint16_t), (size_t)n_merge, f) != (size_t)n_merge) return -1;
    }
    for (int id = T->n_base; id < T->V; id++) {
        if (T->left[id] >= id || T->right[id] >= id) return -1;
    }
    tok_finalize(T);
    return 0;
}

#endif
