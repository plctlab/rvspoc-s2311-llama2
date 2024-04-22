# choose your compiler, e.g. gcc/clang
# example override to clang: make run COMPILER=clang
ifndef COMPILER
#COMPILER = clang
COMPILER = gcc
endif

ifeq ($(COMPILER),clang)
CC = ~/milkv/Xuantie-900-llvm-linux-5.10.4-glibc-x86_64-V1.0.0-beta/bin/clang -mcpu=c906fdv -mrvv-vector-bits=128 -fopenmp-simd -static
CFLAGS_EXTRA = -fno-vectorize
else ifeq ($(COMPILER),gcc)
CC = ~/milkv/Xuantie-900-gcc-linux-5.10.4-musl64-x86_64-V2.8.1/bin/riscv64-unknown-linux-musl-gcc -mcpu=c906fdv -mrvv-vector-bits=128 -fopenmp-simd -static
else
$(error "unknown compiler!")
endif

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c runq.c
	$(CC) -O3 $(CFLAGS_EXTRA) -o run run.c -lm
	$(CC) -O3 $(CFLAGS_EXTRA) -o runq runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c runq.c
	$(CC) -g -o run run.c -lm
	$(CC) -g -o runq runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c runq.c
	$(CC) -Ofast $(CFLAGS_EXTRA) -o run-fast run.c -lm
	$(CC) -Ofast $(CFLAGS_EXTRA) -o runq-fast runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c runq.c
	$(CC) -Ofast -fopenmp -march=native $(CFLAGS_EXTRA) run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native $(CFLAGS_EXTRA) runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq
