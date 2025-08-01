# summarizer.py

import fitz  # PyMuPDF for reading PDF
from transformers import pipeline

# Load summarization pipeline with pretrained model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_pdf(pdf_file):
    """
    Extracts text from a PDF file and summarizes it using BART.
    """
    text = ""

    # Read PDF content
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    # Limit text to 1024 tokens (BART's max input size)
    if len(text) > 1024:
        text = text[:1024]

    # Summarize the text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    return summary[0]['summary_text']
