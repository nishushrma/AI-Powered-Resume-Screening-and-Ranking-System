import streamlit as st 
from PyPDF2 import PdfReader
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)  

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    vectors = tfidf_matrix.toarray()

    # calculate cosine similarity
    job_description_vector=vectors[0]
    resume_vectors=vectors[1:]
    cosine_similarities=cosine_similarity([job_description_vector],resume_vectors).flatten()

    return cosine_similarities

# streamlit app
st.title("AI Resume Screening & Candidate Ranking System")
# Job Description
st.header("Job Description")
job_description=st.text_area("Enter the job description")

# file uploader
st.header("Upload Resumes")
uploaded_files=st.file_uploader("Upload PDF files",type=["pdf"],accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    for file in uploaded_files:  # Fixed incorrect variable name
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)  # Fixed variable name

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})  
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)

