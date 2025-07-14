# app.py
# Contains the frontend of the website
import streamlit as st

st.set_page_config(page_title="Resume Evaluator", layout="centered")

st.title("📄 Resume Evaluator")
st.subheader("Check how well your resume matches a job description")

resume_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

job_description = st.text_area("Paste the Job Description here")

if st.button("Evaluate"):
    if resume_file is None and job_description.strip() == "":
        st.error("Please upload a resume and enter a job description.")
    elif resume_file is None:
        st.error("Please upload a resume")
    elif job_description.strip() == "":
        st.error("Please enter a job description.")
    else:
        # Placeholder score (you'll replace this with real logic)
        st.success("✅ Match Score: 78.5%")
        st.info("Missing Skills: ['SQL', 'AWS']")
