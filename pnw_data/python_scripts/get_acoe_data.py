import pandas as pd
import sys

pre_part = "https://www.nwd-wc.usace.army.mil/dd/common/web_service/webexec/ecsv?id="
# data_part_hour = "id={}.Flow-In.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Flow-Out.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Stor.Inst.1Hour.0.CBT-REV%3Aunits%3Dkaf"
# data_part_ave = "id={}.Flow-In.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Flow-Out.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Stor.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkaf"
# data_part_day = "id={}.Flow-In.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Flow-Out.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Stor.Inst.~1Day.0.CBT-REV%3Aunits%3Dkaf"
# data_part_hour_computed ="id={}.Flow-In.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Flow-Out.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs%7C{}.Stor.Inst.1Hour.0.CBT-COMPUTED-REV%3Aunits%3Dkaf"
post_part = "&headers=true&filename=&timezone=PST&lookback=1658w1d9h&lookforward=-40w5d9h&startdate=01%2F01%2F1990+04%3A00&enddate=12%2F31%2F2020+04%3A00"

# cbt = dict( 
#     inf_rev = "{}.Flow-In.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs",
#     rel_rev = "{}.Flow-Out.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkcfs",
#     sto_ave = "{}.Stor.Ave.~1Day.1Day.CBT-REV%3Aunits%3Dkaf",
#     sto_hour = "{}.Stor.Inst.1Hour.0.CBT-REV%3Aunits%3Dkaf",
#     sto_day = "{}.Stor.Inst.~1Day.0.CBT-REV%3Aunits%3Dkaf",
#     sto_hour_comp = "{}.Stor.Inst.1Hour.0.CBT-COMPUTED-REV%3Aunits%3Dkaf"
# )
# usbr = dict(
#     inf_rev = "{}.Flow-In.Ave.~1Day.1Day.USBR-REV%3Aunits%3Dkcfs",
#     rel_rev = "{}.Flow-Out.Ave.~1Day.1Day.USBR-COMPUTED-REV%3Aunits%3Dkcfs",
#     sto_ave = "{}.Stor.Ave.~1Day.1Day.USBR-REV%3Aunits%3Dkaf",
#     sto_hour = "{}.Stor.Inst.1Hour.0.USBR-REV%3Aunits%3Dkaf",
#     sto_day = "{}.Stor.Inst.~1Day.0.USBR-REV%3Aunits%3Dkaf",
#     sto_hour_comp = "{}.Stor.Inst.1Hour.0.USBR-COMPUTED-REV%3Aunits%3Dkaf"
# )


def get_data(data_part):
    print("".join([pre_part, data_part, post_part]))
    df = pd.read_csv("".join([pre_part, data_part, post_part]))
    df = df.dropna().reset_index().drop("index", axis=1)
    df.columns = ["DateTime", "Inflow_cfs", "Release_cfs", "Storage_acft"]
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.loc[:, ["Inflow_cfs", "Release_cfs", "Storage_acft"]] *= 1000
    return df


def main(args):
    dam = args[0]

    inf_start = "{}.Flow-In".format(dam)
    rel_start = "{}.Flow-Out".format(dam)
    sto_start = "{}.Stor-Total".format(dam)
    
    inf_middle = "Inst.~1Day.0"
    rel_middle = "Inst.~1Day.0"
    sto_middle = "Inst.~1Day.0"

    inf_end = "IDP-REV%3Aunits%3Dkcfs"
    rel_end = "IDP-COMPUTED-REV%3Aunits%3Dkcfs"
    sto_end = "IDP-COMPUTED-REV%3Aunits%3Dkaf"

    inf_part = ".".join([inf_start, inf_middle, inf_end])
    rel_part = ".".join([rel_start, rel_middle, rel_end])
    sto_part = ".".join([sto_start, sto_middle, sto_end])

    df = get_data("%7C".join([inf_part, rel_part, sto_part]))
    print(df)
    response = input("Do you want to save this data? [y,N] ")
    if response.lower() == "y":
        print(f"Saving to ./dam_data/{dam.upper()}.json")
        df.to_json(f"./dam_data/{dam.upper()}.json")
        
if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)