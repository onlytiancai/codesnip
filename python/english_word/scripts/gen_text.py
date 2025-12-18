import pdfplumber

pdf_path = "8-1.pdf"
txt_path = "8-1.txt"

with pdfplumber.open(pdf_path) as pdf:
    with open(txt_path, "w", encoding="utf-8") as f:
        for page in pdf.pages:
            print(page.page_number)
            text = page.extract_text()
            print(text)
            if text:
                f.write(text + "\n")

print("转换完成")
