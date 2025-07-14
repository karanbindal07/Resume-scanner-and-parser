import streamlit as st
import time
from model import clean_resume_and_job_description as cleaner


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
        with st.spinner("Evaluating..."):
            time.sleep(1.5)  

            # Currently put placeholder data
            # Will be connected to model that will be in the model.py file
            score = 78.5
            missing_skills = ['SQL', 'AWS', 'Docker']

            st.success(f"✅ Match Score: {score:.1f}%")
            st.info(f"Missing Skills: {', '.join(missing_skills)}")

            # Currently put everything except common words into the variables and print 
            cleaned_resume,cleaned_jd = (cleaner(resume_file,job_description))
            st.success(cleaned_resume)
            st.success(cleaned_jd)




# Change to only take PDF's
