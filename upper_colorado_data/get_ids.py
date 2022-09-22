import pandas as pd
from tqdm import tqdm
from IPython import embed as II

def load_get_ids():
    return pd.read_csv("./get_ids.csv", header=None, names=["Res", "SID", "Desc"])

def make_sid_query(ids):
    return ",".join(ids["SID"].astype(str))

def make_url(sid_query):
    fmturl = f"https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=lchdb&sdi={sid_query}& \
               tstp=DY&t1=1950-01-01T19:03&t2=2022-01-24T00:00&table=R&mrid=0&format=table"
    return fmturl.replace(" ", "")

def get_url_1by1(ids):
    output = []
    for sid in tqdm(ids["SID"]):
        try:
            url = make_url(str(sid))
            df = pd.read_csv(url)
            output.append(df)
        except pd.errors.ParserError:
            print(f"ERROR: {url}")
    return output

def main():
    ids = load_get_ids()
    sid_query = make_sid_query(ids)
    url = make_url(sid_query)
    tables = pd.read_html(url)
    df = tables[0]
    II()

if __name__ == "__main__":
    main()
