import pandas as pd
import torch
import re
import numpy as np
from tqdm import tqdm
from getCleanData import get_cleaned_datasets
from sentence_transformers import SentenceTransformer, util

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Encode all resume job roles globally
def precompute_resume_role_embeddings(cleaned_resume):
    resume_texts = cleaned_resume['resume_text'].tolist()
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
    return resume_embeddings

def precompute_job_role_embeddings(cleaned_job):
    job_texts = cleaned_job['job_text'].tolist()
    job_embeddings = model.encode(job_texts, convert_to_tensor=True)
    return job_embeddings

# SBERT-based semantic matching of resumes to job title
# This function finds resumes that match the job title semantically
def find_matching_resumes(job_embedding, cleaned_resume, resume_embeddings, top_n=50):
    cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
    top_scores, top_indices = torch.topk(cosine_scores, k=top_n)
    return cleaned_resume.iloc[top_indices.tolist()]

def extract_years(text):
    if pd.isna(text): return 0
    match = re.search(r'(\d+)', str(text))
    return int(match.group(1)) if match else 0

# Main pairing function
# This function generates pairs of resumes and jobs based on semantic matching and other criteria
# If not enough matches are found, you randomly choose resumes
def generate_pairs(cleaned_job, cleaned_resume, resumes_per_job=5):
    paired_data = []

    print('\n\nEncoding full resume texts and job descriptions...')
    resume_embeddings = precompute_resume_role_embeddings(cleaned_resume)
    job_embeddings = precompute_job_role_embeddings(cleaned_job)

    for jidx, job in tqdm(cleaned_job.iterrows(), total=len(cleaned_job), desc="Pairing"):
        job_id = job['job_id']
        job_text = job['job_text']
        job_embedding = job_embeddings[jidx]

        # Compute similarity to all resumes
        cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
        top_scores, top_indices = torch.topk(cosine_scores, k=resumes_per_job)

        selected_resumes = cleaned_resume.iloc[top_indices.tolist()]

        for i, (_, resume) in enumerate(selected_resumes.iterrows()):
            resume_years = extract_years(resume.get('experiencere_requirement', ''))
            
            paired_data.append({
                'job_id': job_id,
                'resume_id': resume.name,
                'job_text': job_text,
                'resume_text': resume['resume_text'],
                'similarity_score': float(top_scores[i]),
                'experience_years': resume_years,
                'label': resume['label']
            })

    return pd.DataFrame(paired_data)

# Entry point to generate paired dataset
def main():
    cleaned_resume, cleaned_job = get_cleaned_datasets()
    cleaned_job = cleaned_job.head(10)
    
    paired_df = generate_pairs(cleaned_job, cleaned_resume, resumes_per_job=3)
    
    print(f'\nGenerated {len(paired_df)} resume-job pairs.')
    print('\nSample:')
    print(paired_df.head())
    
    paired_df.to_csv('paired_data.csv', index=False)
    print('Paired data saved to paired_data.csv')

if __name__ == '__main__':
    main()