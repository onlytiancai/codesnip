# wav2aac — WAV → AAC-LC (.m4a) converter using Apple AudioToolbox
# Target: macOS 11+ on Apple Silicon (M1/M2/M3/M4)

CC      = /usr/bin/clang
CFLAGS  = -O3 -Wall -Wextra -Wno-unused-parameter -std=c11 \
          -march=armv8.5-a -mtune=apple-m4 \
          -Iinclude
LDFLAGS = -framework AudioToolbox -framework CoreFoundation \
          -framework CoreAudio -framework Foundation

BUILD   = build
SRC     = src
TEST    = test

# Main converter
TARGET  = $(BUILD)/wav2aac

# Object files
OBJS    = $(BUILD)/main.o \
          $(BUILD)/util.o \
          $(BUILD)/wav_parser.o \
          $(BUILD)/aac_encoder.o \
          $(BUILD)/m4a_muxer.o \
          $(BUILD)/converter.o \
          $(BUILD)/benchmark.o

.PHONY: all clean test bench format

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD)/%.o: $(SRC)/%.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

# Unit tests
$(BUILD)/test_wav_parser: $(TEST)/test_wav_parser.c $(BUILD)/wav_parser.o $(BUILD)/util.o
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD)/test_m4a_muxer: $(TEST)/test_m4a_muxer.c $(BUILD)/m4a_muxer.o $(BUILD)/util.o
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@

test: $(BUILD)/test_wav_parser $(BUILD)/test_m4a_muxer
	@echo "=== Running unit tests ==="
	./$(BUILD)/test_wav_parser
	./$(BUILD)/test_m4a_muxer
	@echo "=== All tests passed ==="

bench:
	@echo "Generating test WAVs..."
	./bench/generate_test_wavs.sh
	@echo "Running benchmark..."
	./bench/run_benchmark.sh

clean:
	rm -rf $(BUILD)/*.o $(BUILD)/wav2aac $(BUILD)/test_*

format:
	@which clang-format >/dev/null && clang-format -i -style=LLVM $(SRC)/*.c $(SRC)/*.h $(TEST)/*.c include/*.h || echo "clang-format not installed, skipping"