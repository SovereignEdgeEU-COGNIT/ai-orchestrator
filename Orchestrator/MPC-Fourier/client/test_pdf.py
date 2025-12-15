from PyPDF2 import PdfReader, PdfWriter

reader = PdfReader("../../../../chanh/Downloads/output.pdf")
writer = PdfWriter()

for page in reader.pages:
    writer.add_page(page)

writer.add_metadata({
    '/Author': 'Klarna',
    '/Producer': 'Klarna',
'/Creator': 'Klarna',
    '/CreationDate': "D:20250904223015Z",  # PDF datetime format
})

with open("../../../../chanh/Downloads/output.pdf", "wb") as f:
    writer.write(f)
