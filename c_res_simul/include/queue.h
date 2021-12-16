#ifndef QUEUE_H_
#define QUEUE_H_

#include <sys/queue.h>

struct entry
{
    double data;
    STAILQ_ENTRY(entry) entries;
};

STAILQ_HEAD(myqueue, entry);

// calculate the average of values in the queue
double average(struct myqueue q);

// append an item to the right side of the queue (tail)
void append(struct myqueue *q, double value);

// append an item to the left side of the queue (head)
void appendleft(struct myqueue *q, double value);

// remove and return an item from the right side of the queue (tail)
double pop(struct myqueue *q);

// remove and return an item from the left side of the queue (head)
double popleft(struct myqueue *q);

// print all items of the queue on new lines
void print_queue(struct myqueue q);

// free the queue when done using it
void free_queue(struct myqueue *q);

#endif // QUEUE_H_
