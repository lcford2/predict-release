import glob
from IPython.core.interactiveshell import get_default_colors
import pandas as pd
from extract_pdf_data import *
from IPython import embed as II

files = glob.glob("./acoe_data/pdfs/*.pdf")
files.sort()

def get_res_df(file):
    text = get_pdf_text(file)
    results = extract_preliminary_text(text)
    II()
    return create_output_data(results)

for file in files:
    try:
        text = get_pdf_text(file)
    except PyPDF2.utils.PdfReadError as e:
        continue
    results = extract_preliminary_text(text)    
    if results:
        print(file)
        break



II()
