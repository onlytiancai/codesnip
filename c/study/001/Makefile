CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -pedantic
LDLIBS  ?= -lm

.PHONY: all test clean

all: stats

stats: stats.c
	$(CC) $(CFLAGS) -o $@ stats.c $(LDLIBS)

# 单元测试直接 #include "../stats.c" 以复用 static 函数，
# 因此只编译测试文件本身，stats.c 仅作为重建依赖。
tests/test_stats: tests/test_stats.c stats.c
	$(CC) $(CFLAGS) -o $@ tests/test_stats.c $(LDLIBS)

test: stats tests/test_stats
	./tests/run_tests.sh

clean:
	rm -f stats tests/test_stats
