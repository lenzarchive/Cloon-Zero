NVCC ?= nvcc
ARCH ?= sm_75
CXX ?= g++
CC ?= gcc

SRC := src/gpt_cuda.cu src/kernels.cuh src/config.h src/rng.h src/sampler.h src/bpe.h src/cuda_emu.h
EMU_FLAGS := -O2 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -DCUDA_EMULATION -x c++
EMU_SMALL := -DN_EMBD=32 -DN_HEAD=2 -DN_LAYER=2 -DBLOCK_SIZE=16 -DBATCH=2 \
             -DEW_THREADS=32 -DLN_THREADS=32 -DATT_THREADS=32 -DXENT_THREADS=32
EMU_GRAD := $(subst BLOCK_SIZE=16,BLOCK_SIZE=80,$(EMU_SMALL))

.PHONY: all gpu emu prep bench test test-bpe test-grad test-nodropout test-sampler test-multi test-opt clean

all: gpu prep

gpu: gpt_cuda

gpt_cuda: $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -std=c++17 -lineinfo -o $@ src/gpt_cuda.cu

emu: $(SRC)
	$(CXX) $(EMU_FLAGS) -o gpt_emu src/gpt_cuda.cu -lm

prep:
	$(CC) -O2 -std=gnu99 -Wall -Wextra -o prep_corpus tools/prep_corpus.c

bench: gpt_cuda
	./gpt_cuda bench

test: test-bpe test-grad test-nodropout test-sampler test-multi test-opt

test-bpe:
	$(CC) -O2 -std=gnu99 -o test_bpe tests/test_bpe.c
	./test_bpe data/id_en_corpus.txt 4096 3000000

test-grad: $(SRC)
	$(CXX) $(EMU_FLAGS) $(EMU_GRAD) -o gpt_emu_check src/gpt_cuda.cu -lm
	./gpt_emu_check check 80

test-nodropout: $(SRC)
	$(CXX) $(EMU_FLAGS) $(filter-out -DATT_THREADS=32,$(EMU_GRAD)) -DDROPOUT=0 -DATT_THREADS=16 -o gpt_emu_check_nodrop src/gpt_cuda.cu -lm
	./gpt_emu_check_nodrop check 80

test-sampler: test-grad
	./gpt_emu_check selftest

test-multi: $(SRC)
	$(CXX) $(EMU_FLAGS) $(EMU_SMALL) -DLOG_EVERY=1 -DEVAL_EVERY=5 -DEVAL_BATCHES=2 -o gpt_emu_multi src/gpt_cuda.cu -lm
	sh tests/test_multi.sh ./gpt_emu_multi data/id_en_corpus.txt

OPT_FLAGS := -DLOG_EVERY=10 -DEVAL_EVERY=1000 -DSAVE_EVERY=30

test-opt: $(SRC)
	$(CXX) $(EMU_FLAGS) $(EMU_GRAD) -DQK_NORM=1 -DOPTIMIZER_MUON=1 -o gpt_emu_check_opt src/gpt_cuda.cu -lm
	./gpt_emu_check_opt check 60
	$(CXX) $(EMU_FLAGS) $(EMU_SMALL) $(OPT_FLAGS) -o gpt_emu_o_base src/gpt_cuda.cu -lm
	$(CXX) $(EMU_FLAGS) $(EMU_SMALL) $(OPT_FLAGS) -DQK_NORM=1 -o gpt_emu_o_qk src/gpt_cuda.cu -lm
	$(CXX) $(EMU_FLAGS) $(EMU_SMALL) $(OPT_FLAGS) -DQK_NORM=1 -DOPTIMIZER_MUON=1 -o gpt_emu_o_muon src/gpt_cuda.cu -lm
	sh tests/test_opt.sh ./gpt_emu_o_base ./gpt_emu_o_qk ./gpt_emu_o_muon data/id_en_corpus.txt

clean:
	rm -f gpt_cuda gpt_emu gpt_emu_check gpt_emu_check_nodrop gpt_emu_check_opt gpt_emu_multi gpt_emu_o_base gpt_emu_o_qk gpt_emu_o_muon test_bpe prep_corpus
