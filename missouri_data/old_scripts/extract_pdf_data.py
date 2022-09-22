import PyPDF2
import re
import pandas as pd
from locale import atof, setlocale, LC_NUMERIC

# setup string to numeric conversion
setlocale(LC_NUMERIC, "")

# setup parsing patterns and columns
comma_pat = r"((\d{1,3},)+\d{3})"
sdec_pat = r"\-?\d+\.\d"
ddec_pat = r"\-?\d+\.\d{2}"
PATTERNS = [sdec_pat, sdec_pat, comma_pat, comma_pat, 
            ddec_pat, ddec_pat, comma_pat, comma_pat, comma_pat, 
            sdec_pat, f"0|{comma_pat}", sdec_pat]

COLUMNS = ["mp_sto", "fc_sto", "mp_cum_sto", "fc_cum_sto",
           "elev", "delta_elev", "storage", "inflow", "release",
           "mp_pct_occ", "fc_occ", "fc_pct_occ"]

def get_pdf_text(file_name: str) -> str:
    with open(file_name, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        text = reader.getPage(0).extractText()
    return text

def extract_preliminary_text(text: str) -> list:
    dams = ["Fort Peck Dam", "Garrison Dam", "Oahe Dam", "Big Bend Dam", "Fort Randall Dam", "Gavins Point Dam"]
    match_res = "|".join(dams)
    dam_pattern = f"({match_res})([0-9\-\.\,]+)(?=\w)"
    return re.findall(dam_pattern, text)

def parse_result_numbers(string: str, patterns: list) -> list:
    output = []
    for pattern in patterns:
        match = re.search(pattern, string)
        if not match:
            output.append(-99999999.0)
        else:
            match_text = match.group(0)
            output.append(atof(match_text))
            string = string[:match.start()] + string[match.end():]
    return output

def create_output_data(results: list, patterns: list=None, columns: list=None) -> pd.DataFrame:
    if not patterns:
        patterns = PATTERNS
    if not columns:
        columns = COLUMNS
    output = []
    for dam, string in results:
        parsed = parse_result_numbers(string, patterns)
        output.append(pd.DataFrame(parsed, index=columns, columns=[dam]))
    return pd.concat(output, axis=1, ignore_index=False)

if __name__ == "__main__":
    file_name = "./acoe_data/pdfs/MRBWM_Reservoir_01012021.pdf"
    text = get_pdf_text(file_name)
    results = extract_preliminary_text(text)
    output = create_output_data(results)
    print(output)
