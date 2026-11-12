# Makefile for branchless_filter.c
#
# Reproduces the Rust NEON × branchless × nozero × stnp benchmark in C,
# side-by-side with src/main.rs in the parent project.
#
# Targets:
#   make            — default build, clang on aarch64-darwin with -O3 + NEON
#   make run        — build and run
#   make clean      — remove the binary
#
# Override CC, CFLAGS, LDFLAGS as needed.

CC      ?= clang
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-function -Wno-unused-parameter
LDFLAGS ?=

# On aarch64-darwin, NEON is the baseline — no extra flags needed.
# On x86_64 / other archs the NEON intrinsics won't compile; the fallback
# versions in branchless_filter.c will kick in via #if !HAVE_NEON.

SRC    := branchless_filter.c
BIN    := branchless_filter

.PHONY: all run clean

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)