#include <stdio.h>
#include <sys/queue.h>
#include <stddef.h>
#include <stdlib.h>

struct entry
{
    double data;
    STAILQ_ENTRY(entry) entries;
};

STAILQ_HEAD(myqueue, entry);

double average(struct myqueue q)
{
    struct entry *np;
    double out = 0.0;
    double n = 0.0;

    STAILQ_FOREACH(np, &q, entries)
    {
        out += np->data;
        n += 1;
    }
    return out / n;
}

void append(struct myqueue *q, double value)
{
    struct entry *p;
    p = malloc(sizeof(struct entry));
    STAILQ_INSERT_TAIL(q, p, entries);
    p->data = value;
}

void appendleft(struct myqueue *q, double value)
{
    struct entry *p;
    p = malloc(sizeof(struct entry));
    STAILQ_INSERT_HEAD(q, p, entries);
    p->data = value;
}

double popleft(struct myqueue *q)
{
    struct entry *p;
    double out;
    p = STAILQ_FIRST(q);
    STAILQ_REMOVE_HEAD(q, entries);
    out = p->data;
    free(p);
    return out;
}

double pop(struct myqueue *q)
{
    struct entry *p1, *p2;
    double out=0.0;
    p1 = STAILQ_FIRST(q);
    while (p1 != NULL)
    {
        p2 = STAILQ_NEXT(p1, entries);
        if (p2 == NULL)
        {
            STAILQ_REMOVE(q, p1, entry, entries);
            out = p1->data;
            free(p1);
        }
        p1 = p2;
    }
    return out;
}

void print_queue(struct myqueue q)
{
    struct entry *p;
    STAILQ_FOREACH(p, &q, entries)
        printf("%f\n", p->data);
}

void free_queue(struct myqueue *q)
{
    struct entry *p1, *p2;
    p1 = STAILQ_FIRST(q);
    while (p1 != NULL)
    {
        p2 = STAILQ_NEXT(p1, entries);
        free(p1);
        p1 = p2;
    }
    STAILQ_INIT(q);
}
