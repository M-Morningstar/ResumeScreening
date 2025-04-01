import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from getCleanData import get_cleaned_datasets
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all resume job roles globally
def precompute_resume_role_embeddings(cleaned_resume):
    resume_roles = cleaned_resume['job_role'].tolist()
    resume_embeddings = model.encode(resume_roles, convert_to_tensor=True)
    return resume_embeddings

def precompute_job_role_embeddings(cleaned_job):
    job_titles = cleaned_job['title'].tolist()
    job_embeddings = model.encode(job_titles, convert_to_tensor=True)
    return job_embeddings

# SBERT-based semantic matching of resumes to job title
# This function finds resumes that match the job title semantically
def find_matching_resumes(job_embedding, cleaned_resume, resume_embeddings, top_n=50):
    cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
    top_scores, top_indices = torch.topk(cosine_scores, k=top_n)
    return cleaned_resume.iloc[top_indices.tolist()]

# Main pairing function
# This function generates pairs of resumes and jobs based on semantic matching and other criteria
# If not enough matches are found, you randomly choose resumes
def generate_pairs(cleaned_job, cleaned_resume, resumes_per_job=5, score_threshold=70, match_threshold=0.6):
    paired_data = []

    print('Encoding resume job roles and job titles...')
    resume_embeddings = precompute_resume_role_embeddings(cleaned_resume)
    job_embeddings = precompute_job_role_embeddings(cleaned_job)

    for jidx, job in tqdm(cleaned_job.iterrows(), total=len(cleaned_job), desc="Pairing"):
        job_id = job['job_id']
        job_text = job['job_text']
        job_embedding = job_embeddings[jidx]

        # Find resumes with semantically similar job roles
        matching_resumes = find_matching_resumes(job_embedding, cleaned_resume, resume_embeddings, top_n=50)

        # If not enough matching resumes, fall back to random
        if len(matching_resumes) >= resumes_per_job:
            selected_resumes = matching_resumes.iloc[np.random.choice(len(matching_resumes), resumes_per_job, replace=False)]
        else:
            selected_resumes = cleaned_resume.sample(resumes_per_job, random_state=jidx)

        for _, resume in selected_resumes.iterrows():
            ai_score = resume['ai_score']
            decision = resume['recruiter_decision']
            label = 1 if ai_score >= score_threshold or decision == 'hire' else 0

            paired_data.append({
                'job_id': job_id,
                'resume_id': resume['resume_id'],
                'job_text': job_text,
                'resume_text': resume['resume_text'],
                'experience': resume['experience'],
                'projects': resume['projects'],
                'ai_score': ai_score,
                'label': label
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