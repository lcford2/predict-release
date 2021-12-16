#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include "queue.h"

int main(void)
{
    struct entry *n1, *n2;
    struct myqueue release;
    struct myqueue storage;
    struct myqueue inflow;
    double array_average;
    double popped;

    STAILQ_INIT(&release);
    STAILQ_INIT(&storage);
    STAILQ_INIT(&inflow);

    for (int i=0; i<7; i++)
    {
        n1 = malloc(sizeof(struct entry));
        STAILQ_INSERT_HEAD(&release, n1, entries);
        n1->data = (double) (rand() % 10) + 1; // random number between 1 and 10
    }

    /* print_queue(release); */

    for (int i=0; i<5; i++)
    {
        append(&release, (double) (rand() % 10) + 1);

        popped = popleft(&release);
        printf("\nPopped = %.2f\n", popped);
        printf("\n");
        print_queue(release);
        array_average = average(release);
        printf("Average: %f\n", array_average);
    }

    popped = pop(&release);
    printf("\nPopped = %.2f\n\n", popped);
    print_queue(release);

    n1 = STAILQ_FIRST(&release);
    while (n1 != NULL)
    {
        n2 = STAILQ_NEXT(n1, entries);
        free(n1);
        n1 = n2;
    }

    STAILQ_INIT(&release);

    exit(EXIT_SUCCESS);
}
