#include <stdio.h>
#include <stdlib.h>
#include "queue.h"


double standardize(double value, double mean, double std)
{
    return (value - mean) / std;
}

double unstandardize(double value, double mean, double std)
{
    return value * std + mean;
}

double normalize(double value, double minval, double maxval)
{
    return (value - minval)/(maxval - minval);
}

double unnormalize(double value, double minval, double maxval)
{
    return value * (maxval - minval) + minval;
}


void res_simul(double *intercepts, double *re_coefs,                               // params
               double *prev_rel, double *prev_sto, double *inflow,                 // independent variables
               double rel_mean, double sto_mean, double inf_mean, double sxi_mean, // var means
               double rel_std, double sto_std, double inf_std, double sxi_std,     // var stds
               int ts_len,
               double rel_max, double rel_min,                         // time series length, rel bounds
               double sto_max, double sto_min,
               double inf_max, double inf_min,
               double sxi_max, double sxi_min,
               int std_or_norm,                                    // sto bounds
               double *rel_out, double *sto_out)                                   // output arrays
{
    int i, j;
    double sto_tmin1, rel_tmin1;
    double n_sto_tmin1, n_rel_tmin1, n_inf_t, n_sxi_t;
    double rweek_mean, sweek_mean, iweek_mean;
    struct myqueue rel_week, sto_week, inf_week;

    // init queues for weekly release, storage, and inflow
    STAILQ_INIT(&rel_week);
    STAILQ_INIT(&sto_week);
    STAILQ_INIT(&inf_week);

    // add initial values to the queues
    for (i=0; i<7; i++)
    {
        append(&rel_week, prev_rel[i]);
        append(&sto_week, prev_sto[i]);
        append(&inf_week, inflow[i]);
    }


    for (i=0; i<ts_len; i++)
    {
        j = i + 6;
        if (i == 0) {
            sto_tmin1 = prev_sto[j];
            rel_tmin1 = prev_rel[j];
        } else {
            sto_tmin1 = sto_out[i-1];
            rel_tmin1 = rel_out[i-1];
        }


        if (std_or_norm == 0) {
            n_rel_tmin1  = standardize(rel_tmin1, rel_mean, rel_std);
            n_sto_tmin1  = standardize(sto_tmin1, sto_mean, sto_std);
            n_inf_t      = standardize(inflow[j], inf_mean, inf_std);
            n_sxi_t      = standardize(sto_tmin1 * inflow[j], sxi_mean, sxi_std);
            rweek_mean = standardize(average(rel_week), rel_mean, rel_std);
            sweek_mean = standardize(average(sto_week), sto_mean, sto_std);
            iweek_mean = standardize(average(inf_week), inf_mean, inf_std);
        } else {
            n_rel_tmin1  = normalize(rel_tmin1, rel_min, rel_max);
            n_sto_tmin1  = normalize(sto_tmin1, sto_min, sto_max);
            n_inf_t      = normalize(inflow[j], inf_min, inf_max);
            n_sxi_t      = normalize(sto_tmin1 * inflow[j], sxi_min, sxi_max);
            rweek_mean = normalize(average(rel_week), rel_min, rel_max);
            sweek_mean = normalize(average(sto_week), sto_min, sto_max);
            iweek_mean = normalize(average(inf_week), inf_min, inf_max);
        }

        rel_out[i] = intercepts[j] +
            re_coefs[0] * n_rel_tmin1 +
            re_coefs[1] * n_sto_tmin1 +
            re_coefs[2] * n_inf_t +
            re_coefs[3] * n_sxi_t +
            re_coefs[4] * rweek_mean +
            re_coefs[5] * sweek_mean +
            re_coefs[6] * iweek_mean;

        if (std_or_norm == 0) {
            rel_out[i] = unstandardize(rel_out[i], rel_mean, rel_std);
        } else {
            rel_out[i] = unnormalize(rel_out[i], rel_min, rel_std);
        }
        if (rel_out[i] < rel_min)
            rel_out[i] = rel_min;
        if (rel_out[i] > rel_max)
            rel_out[i] = rel_max;

        sto_out[i] = sto_tmin1 + inflow[j] - rel_out[i];

        if (sto_out[i] < sto_min)
            sto_out[i] = sto_min;
        if (sto_out[i] > sto_max)
            sto_out[i] = sto_max;

        // remove old values and add new ones to weekly containers
        append(&rel_week, rel_out[i]);
        append(&sto_week, sto_out[i]);
        append(&inf_week, inflow[j]);

        popleft(&rel_week);
        popleft(&sto_week);
        popleft(&inf_week);
    }

    free_queue(&rel_week);
    free_queue(&sto_week);
    free_queue(&inf_week);
}
