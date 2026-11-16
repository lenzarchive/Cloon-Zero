#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t hash_line(const unsigned char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= s[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1;
}

typedef struct {
    uint64_t *slot;
    size_t cap, used;
} HashSet;

static void set_init(HashSet *s, size_t cap) {
    s->slot = (uint64_t *)calloc(cap, sizeof(uint64_t));
    if (!s->slot) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    s->cap = cap;
    s->used = 0;
}

static void set_grow(HashSet *s) {
    HashSet n;
    set_init(&n, s->cap * 2);
    for (size_t i = 0; i < s->cap; i++) {
        if (!s->slot[i]) continue;
        size_t j = (size_t)(s->slot[i] & (n.cap - 1));
        while (n.slot[j]) j = (j + 1) & (n.cap - 1);
        n.slot[j] = s->slot[i];
        n.used++;
    }
    free(s->slot);
    *s = n;
}

// Returns 1 if h was already present, otherwise inserts it and returns 0. The table doubles at 50% load,
// so deduplication stays exact however many lines the corpus has.
static int set_seen_or_add(HashSet *s, uint64_t h) {
    size_t j = (size_t)(h & (s->cap - 1));
    while (s->slot[j]) {
        if (s->slot[j] == h) return 1;
        j = (j + 1) & (s->cap - 1);
    }
    if ((s->used + 1) * 2 > s->cap) {
        set_grow(s);
        j = (size_t)(h & (s->cap - 1));
        while (s->slot[j]) j = (j + 1) & (s->cap - 1);
    }
    s->slot[j] = h;
    s->used++;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <in.txt> <out.txt> [min_dedupe_len=40] [min_alpha_ratio=0.5]\n", argv[0]);
        return 1;
    }
    const size_t min_len = argc > 3 ? (size_t)atoi(argv[3]) : 40;
    const double min_alpha = argc > 4 ? atof(argv[4]) : 0.5;
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }
    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "Cannot write %s\n", argv[2]);
        fclose(in);
        return 1;
    }
    setvbuf(in, NULL, _IOFBF, 1 << 20);
    setvbuf(out, NULL, _IOFBF, 1 << 20);

    HashSet seen;
    set_init(&seen, (size_t)1 << 22);
    size_t lcap = 1 << 16;
    unsigned char *line = (unsigned char *)malloc(lcap);
    long long total = 0, kept = 0, dup = 0, junk = 0, blank_skipped = 0;
    long long bytes_in = 0, bytes_out = 0;
    int prev_blank = 1;
    size_t n = 0;
    int eof = 0;
    while (!eof) {
        const int c = getc_unlocked(in);
        if (c == EOF) {
            eof = 1;
            if (n == 0) break;
        } else if (c != '\n') {
            if (n + 2 >= lcap) {
                lcap *= 2;
                line = (unsigned char *)realloc(line, lcap);
            }
            line[n++] = (unsigned char)c;
            continue;
        }
        total++;
        bytes_in += (long long)n + 1;
        size_t b = 0, e = n;
        while (b < e && line[b] == '\r') b++;
        while (e > b && (line[e - 1] == ' ' || line[e - 1] == '\t' || line[e - 1] == '\r')) e--;
        const size_t len = e - b;
        n = 0;
        if (len == 0) {
            if (!prev_blank) {
                fputc('\n', out);
                bytes_out++;
                prev_blank = 1;
            } else {
                blank_skipped++;
            }
            continue;
        }
        size_t alpha = 0;
        for (size_t i = b; i < e; i++) {
            const unsigned char ch = line[i];
            if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == ' ') alpha++;
        }
        if (len >= 8 && (double)alpha / (double)len < min_alpha) {
            junk++;
            continue;
        }
        if (len >= min_len && set_seen_or_add(&seen, hash_line(line + b, len))) {
            dup++;
            continue;
        }
        fwrite(line + b, 1, len, out);
        fputc('\n', out);
        bytes_out += (long long)len + 1;
        kept++;
        prev_blank = 0;
    }
    fclose(in);
    fclose(out);
    printf("lines: %lld | kept: %lld | duplicates removed: %lld | low-alpha removed: %lld | blank collapsed: %lld\n",
           total, kept, dup, junk, blank_skipped);
    printf("bytes: %lld -> %lld (%.1f%%)\n", bytes_in, bytes_out,
           bytes_in ? 100.0 * (double)bytes_out / (double)bytes_in : 0.0);
    free(seen.slot);
    free(line);
    return 0;
}
