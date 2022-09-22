import json
from IPython import embed as II


def load_ids():
    with open("./usbr_ids.json", "r") as f:
        ids = json.load(f)
    return ids

def get_user_selections(ids):
    get_ids = []
    
    for res, values in ids.items():
        print(f"\n{res} has:")
        for sid, desc in values:
            print(desc)
        resp = input(f"\nGet all reservoir information for {res}? [Y/n] ")
        if resp.lower() != "n":
            for sid, desc in values:
                get_ids.append((res, sid, desc))
    return get_ids

def write_selections(selections):
    with open("get_ids.csv", "w") as f:
        f.writelines([",".join(i) + "\n" for i in selections])

def main():
    ids = load_ids()
    selections = get_user_selections(ids)
    write_selections(selections)
    II()

if __name__ == "__main__":
    main()
