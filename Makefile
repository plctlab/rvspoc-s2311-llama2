# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
# CC = gcc
CC = riscv64-unknown-linux-musl-gcc 

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -static -Ofast -o run-gcc-musl-Ofast-rv64gcv0p7_zfh_xthead run.c
	$(CC) -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c906 -static -Ofast -o runq-gcc-musl-Ofast-rv64gcv0p7_zfh_xthead runq.c

.PHONY: clean
clean:
	rm -f run
	rm -f runq
