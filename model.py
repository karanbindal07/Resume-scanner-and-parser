import PyPDF2
import re


def extract_text_from_file(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\n', ' ', text)  # remove newlines
    text = re.sub(r'[^a-z\s]', '', text)  # remove everything except letters and spaces
    words = text.split()
    # remove very common junk words
    common_words = ['the', 'and', 'or', 'to', 'a', 'an', 'of', 'in', 'on', 'with', 'for', 'is', 'at']
    words = [word for word in words if word not in common_words]
    return " ".join(words)

def clean_resume_and_job_description(resume_file, job_description_text):
    resume_text = extract_text_from_file(resume_file)
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description_text)
    return cleaned_resume, cleaned_jd




