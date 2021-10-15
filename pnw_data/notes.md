# Notes on Data Collection

## How is data being collected?

* Initially, data was collected from all reservoirs in the [Hydromet for PNW](https://www.usbr.gov/pn/hydromet/arcread.html)
  * This was done with the `get_pnw_data.py` script.
  * In this script, if any of the three time series was not available (Storage, Inflow, Release) then the data was not stored.
    * This is probably not reasonable for ROR reservoirs as their release does not depend on storage. This is something to visit in the future.
    * For example, The Dalles operates as a run of river reservoir but generates massive amounts of hydropower.
  * This process only generated data for several reservoirs but most are incomplete (i.e. missing one of the three major variables)
  * This data is stored in `./dam_data/hydromet_data`
* To collect more information, I began using the Army Corps of Engineers [Dataquery 2.0 web service](https://www.nwd-wc.usace.army.mil/dd/common/dataquery/www/)
  * This process was a bit more complicated but I used pandas to query information for reservoirs that have Storage, Inflow, and Release time series available
  * Release and Inflow is available as daily averages, Storage is retrieved as instantaneous hourly values and then the end of day storage is saved as the storage for each reservoir (i.e. when the hour is 2300)
  * To do this I split the url into three parts:
    * The preamble: `https://www.nwd-wc.usace.army.mil/dd/common/web_service/webexec/ecsv?`
    * The data part: `id={}.Flow-In.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Flow-Out.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Stor.Inst.1Hour.0.CBT-REV%3Aunits%3Dkaf`
      * In this part, the brackets are replaced with dam ids, like BON for Bonneville Dam
    * The format part: `&headers=true&filename=&timezone=PST&lookback=1658w1d9h&lookforward=-40w5d9h&startdate=01%2F01%2F1990+04%3A00&enddate=12%2F31%2F2020+04%3A00`
  * Here the missing rows are dropped, and the data is saved in CFS for Release and Inflow and Acre-feet for Storage
  * To know which URL to use, I have to look up the reservoir on the dataquery page and see what fields they have available. 
  * This could be automated by iterating through them, but I have not done that yet.
  * This data is stored in `./dam_data/acoe_data`

## Reservoir Notes

* The Dalles Dam has no storage time series but is a ROR
* Palisades Dam only has a time series for Average Daily storage, not end of day storage. 