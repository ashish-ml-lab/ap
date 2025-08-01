# pdf_chatbot.py

import fitz  # PDF parser
from transformers import pipeline

# Load question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def ask_question_about_pdf(pdf_file, question):
    """
    Extracts text from PDF and uses it as context for answering a user question.
    """
    text = ""

    # Read PDF content
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    # Truncate long text for performance
    if len(text) > 512:
        text = text[:512]

    # Answer the question
    result = qa_pipeline(question=question, context=text)

    return result['answer']
