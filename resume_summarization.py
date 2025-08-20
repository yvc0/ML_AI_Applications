# resume_summarizer_app.py
#pip install streamlit python-docx PyMuPDF transformers torch
import streamlit as st
import docx
import fitz  # PyMuPDF for PDF
from io import BytesIO
from transformers import pipeline

# Load AI Summarizer (HuggingFace)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.title("📄 AI Resume Summarizer")
st.write("Upload your resume (PDF or DOCX) and get a smart summary with key highlights.")


# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text += page.get_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


# File uploader
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        resume_text = ""

    if resume_text:
        st.subheader("📃 Extracted Resume Text")
        st.text_area("Resume Content", resume_text[:1500] + "...", height=200)

        # Summarize
        if st.button("🔍 Summarize Resume"):
            with st.spinner("Summarizing..."):
                # HuggingFace summarizer works best with smaller chunks
                chunks = [resume_text[i:i + 1000] for i in range(0, len(resume_text), 1000)]
                summary = ""
                for chunk in chunks:
                    try:
                        result = summarizer(chunk, max_length=130, min_length=50, do_sample=False)
                        summary += result[0]['summary_text'] + " "
                    except Exception as e:
                        summary += ""

                st.subheader("✨ AI Summary")
                st.write(summary)

                # Highlight key insights (basic keyword extraction)
                st.subheader("🎯 Key Insights")
                if "Python" in resume_text or "Machine Learning" in resume_text:
                    st.write("- ✅ Candidate has technical background in **AI/ML**.")
                if "Project" in resume_text or "Management" in resume_text:
                    st.write("- ✅ Candidate has **Project Management** experience.")
                if "Bachelor" in resume_text or "Master" in resume_text:
                    st.write("- 🎓 Candidate has higher education.")
