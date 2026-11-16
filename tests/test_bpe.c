#include <time.h>
#include "../src/bpe.h"

static uint8_t *load(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)n + 1);
    *len = fread(b, 1, (size_t)n, f);
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s corpus.txt vocab [max_bytes]\n", argv[0]);
        return 1;
    }
    size_t len;
    uint8_t *text = load(argv[1], &len);
    if (!text) return 1;
    if (argc > 3 && (size_t)atol(argv[3]) < len) len = (size_t)atol(argv[3]);
    Tokenizer T;
    clock_t t0 = clock();
    tok_train(&T, text, len, atoi(argv[2]), 1);
    printf("train: %.1fs, vocab %d\n", (double)(clock() - t0) / CLOCKS_PER_SEC, T.V);
    t0 = clock();
    size_t n;
    uint16_t *ids = tok_encode(&T, text, len, &n);
    printf("encode: %.1fs, %zu tokens, %.3f bytes/token\n", (double)(clock() - t0) / CLOCKS_PER_SEC, n, (double)len / (double)n);
    uint8_t *back = (uint8_t *)malloc(len + 64);
    size_t pos = 0;
    for (size_t i = 0; i < n; i++) pos += tok_decode_token(&T, ids[i], back + pos);
    printf("roundtrip: %s (%zu vs %zu bytes)\n", (pos == len && memcmp(back, text, len) == 0) ? "OK" : "FAIL", pos, len);
    FILE *f = fopen("/tmp/tok.bin", "wb");
    tok_write(f, &T);
    fclose(f);
    Tokenizer T2;
    f = fopen("/tmp/tok.bin", "rb");
    int rc = tok_read(f, &T2);
    fclose(f);
    size_t n2;
    uint16_t *ids2 = tok_encode(&T2, text, len < 2000000 ? len : 2000000, &n2);
    int same = rc == 0;
    for (size_t i = 0; i < n2 && same; i++) same = ids2[i] == ids[i];
    printf("serialize roundtrip: %s\n", same ? "OK" : "FAIL");
    printf("sample tokens:");
    for (int id = T.n_base; id < T.n_base + 8; id++) {
        printf(" [");
        fwrite(T.dpool + T.doff[id], 1, (size_t)T.dlen[id], stdout);
        printf("]");
    }
    for (int id = T.V - 6; id < T.V; id++) {
        printf(" [");
        fwrite(T.dpool + T.doff[id], 1, (size_t)T.dlen[id], stdout);
        printf("]");
    }
    printf("\n");
    return 0;
}
