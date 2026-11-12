# MiniLang —— 拆成多文件之后的构建脚本
#
#   make            编译出 ./minilang
#   make run        编译并运行 demo.ml
#   make test       编译并跑完整测试
#   make clean      删掉编译产物
#
# 注意链接顺序：-lm（数学库）必须放在源文件/目标文件后面。

CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -O2
LDLIBS  ?= -lm

TARGET  := minilang
SRCDIR  := src
SRCS    := $(SRCDIR)/lexer.c $(SRCDIR)/ast.c $(SRCDIR)/parser.c \
           $(SRCDIR)/interpreter.c $(SRCDIR)/main.c
OBJS    := $(SRCS:.c=.o)
DEPS    := $(SRCDIR)/minilang.h

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET) demo.ml

test: $(TARGET)
	./run.sh

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run test clean
