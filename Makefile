CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -O2
TARGET  = regression
SRCS    = main.c regression_core.c regression_io.c
OBJS    = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c regression_utils.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
