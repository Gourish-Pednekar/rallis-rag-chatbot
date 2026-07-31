from pypdf import PdfReader

reader = PdfReader("catalogue.pdf")
print(f"Total pages: {len(reader.pages)}")

for i in range(2, 5):
    text = reader.pages[i].extract_text()
    print(f"\n---- Page {i+1} ----")
    print(text)