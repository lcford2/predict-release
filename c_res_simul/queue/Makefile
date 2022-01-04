
CC=gcc
CFLAGS=-fPIC -Wall -Werror -Wextra -O2 -g
LDFLAGS=-shared
RM=rm -f
TARGET_LIB=libqueue.so

SRCS=queue.c
OBJS=$(SRCS:.c=.o)

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^
	cp $(TARGET_LIB) ../lib/$(TARGET_LIB)
	cp queue.h ../include/queue.h

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

clean:
	$(RM) $(TARGET_LIB) $(OBJS) $(SRCS:.c=.d)
