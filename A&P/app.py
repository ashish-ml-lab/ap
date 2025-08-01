# Import required Flask modules
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Force Transformers to use PyTorch only

from flask import Flask, render_template, request

# Import our custom AI logic from ai_tools folder
from ai_tools.summarizer import summarize_pdf
from ai_tools.pdf_chatbot import ask_question_about_pdf
from ai_tools.handwriting import convert_handwriting_image

# Initialize the Flask app
app = Flask(__name__)

# -------------------------------
# Route: Homepage (navigation UI)
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')  # renders main menu HTML


# ---------------------------------------
# Route: PDF Summarizer tool (/summarizer)
# Handles GET (open form) & POST (submit PDF)
# ---------------------------------------
@app.route('/summarizer', methods=['GET', 'POST'])
def summarizer():
    summary = ""
    if request.method == 'POST':
        pdf = request.files['pdf']  # get uploaded PDF
        summary = summarize_pdf(pdf)  # call AI logic to summarize
    return render_template('summarizer.html', summary=summary)


# ------------------------------------------
# Route: PDF Chatbot (Question Answering on PDF)
# ------------------------------------------
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    answer = ""
    if request.method == 'POST':
        pdf = request.files['pdf']
        question = request.form['question']  # get user question
        answer = ask_question_about_pdf(pdf, question)  # call AI logic
    return render_template('chatbot.html', answer=answer)


# -------------------------------------------------
# Route: Handwriting to Text Converter (/handwriting)
# -------------------------------------------------
@app.route('/handwriting', methods=['GET', 'POST'])
def handwriting():
    result = ""
    if request.method == 'POST':
        image = request.files['image']  # uploaded handwritten image
        result = convert_handwriting_image(image)  # call AI OCR logic
    return render_template('handwriting.html', result=result)


# ---------------------
# Start the Flask app
# ---------------------
if __name__ == '__main__':
    app.run(debug=True)
