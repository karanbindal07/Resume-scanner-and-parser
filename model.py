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

# Next start with taking above outputs as inputs and then making the actual ANN model and then training the model
# The clean function should happen automatically before the ANN starts for a resume and JD
# Main things to do - compare the cleaned resume and JD - find what works and what is missing 
# - this should not be done through what is in both but through a model which can predict what all would be needed for the job 
# #- well can be both - but focus on how the model can predict that yes what part of the resume works and which part does not
# Predict from the trained jd what skills should be needed

# For this last maybe also train a list of thousands of skills???? maybe change based current progress based on that?
# - Otherwise how will it predict which skill would be needed? 

# For this look at ChatGPT's most resent output and then continue from there -- changes to its step 1. 


# have test jd get test resume data and test skills data



