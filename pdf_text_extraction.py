import fitz 



def pdf_text(uploaded_file):
    """Extract text from a PDF file using PyMuPDF and return it as a list."""
    text_descriptions = []  # Store text descriptions for all pages
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text = page.get_text()
            text_descriptions.append(text)  # Collect text for each page
    return ', '.join(text_descriptions)