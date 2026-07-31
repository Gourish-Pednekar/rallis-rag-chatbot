from dotenv import load_dotenv
import pdfplumber

load_dotenv()

with pdfplumber.open("catalogue.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    
    for i in range(2, 5): 
        page = pdf.pages[i]
        text = page.extract_text(layout=True)
        print(f"\n---- Page {i+1} ----")
        print(text)