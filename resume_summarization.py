# resume_summarizer_app.py
# pip install streamlit python-docx PyMuPDF transformers torch

import streamlit as st
import docx
import fitz  # PyMuPDF for PDF
from transformers import pipeline

st.set_page_config(page_title="AI Resume Summarizer", page_icon="📄")

st.title("📄 AI Resume Summarizer")
st.write("Upload your resume (PDF or DOCX) and get a smart summary with key highlights.")


# ✅ Cache model so it's loaded only once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        text += pdf_document[page_num].get_text()
    return text


# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


# Upload file
uploaded_file = st.file_uploader("📂 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    if resume_text.strip():
        st.subheader("📃 Extracted Resume Text")
        st.text_area("Resume Content", resume_text[:1500] + "...", height=200)

        if st.button("🔍 Summarize Resume"):
            with st.spinner("Summarizing... Please wait"):
                summarizer = load_summarizer()

                # Split text into chunks of ~1500 characters
                chunks = [resume_text[i:i + 1500] for i in range(0, len(resume_text), 1500)]

                summary = ""
                for chunk in chunks:
                    try:
                        result = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
                        summary += result[0]['summary_text'] + " "
                    except Exception:
                        continue

                st.subheader("✨ AI Summary")
                st.write(summary.strip())
                
                # Simple key insights extraction
                st.subheader("🎯 Key Insights")
                insights = []
                if "Python" in resume_text or "Machine Learning" in resume_text:
                    insights.append("✅ Candidate has technical background in **AI/ML**.")
                if "Project" in resume_text or "Management" in resume_text:
                    insights.append("✅ Candidate has **Project Management** experience.")
                if "Bachelor" in resume_text or "Master" in resume_text:
                    insights.append("🎓 Candidate has higher education.")
                
                if insights:
                    for item in insights:
                        st.write(item)
                else:
                    st.write("⚡ No major insights detected.")

