from readability import Document
text = open('1.html', encoding='utf-8').read()
doc = Document(text)
print(doc.summary())