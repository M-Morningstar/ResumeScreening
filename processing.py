import pandas as pd
from getCleanData import get_cleaned_datasets
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all resume job roles globally
def precompute_resume_role_embeddings(cleaned_resume):
    resume_roles = cleaned_resume['job_role'].tolist()
    resume_embeddings = model.encode(resume_roles, convert_to_tensor=True)
    return resume_embeddings

# SBERT-based semantic matching of resumes to job title
# This function finds resumes that match the job title semantically
def find_matching_resumes(job_title, cleaned_resume, resume_embeddings, threshold=0.6):
    job_embedding = model.encode(job_title, convert_to_tensor=True)
    cosine_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
    matched_indices = [i for i, score in enumerate(cosine_scores) if score >= threshold]
    return cleaned_resume.iloc[matched_indices]

# Main pairing function
# This function generates pairs of resumes and jobs based on semantic matching and other criteria
# If not enough matches are found, you randomly choose resumes
def generate_pairs(cleaned_job, cleaned_resume, resumes_per_job=5, score_threshold=70, match_threshold=0.6):
    paired_data = []

    print('Encoding resume job roles...')
    resume_embeddings = precompute_resume_role_embeddings(cleaned_resume)

    for jidx, job in cleaned_job.iterrows():
        job_id = job['job_id']
        job_text = job['job_text']
        job_title = job['title']

        # Find resumes with semantically similar job roles
        matching_resumes = find_matching_resumes(job_title, cleaned_resume, resume_embeddings, threshold=match_threshold)

        # If not enough matching resumes, fall back to random
        if len(matching_resumes) >= resumes_per_job:
            selected_resumes = matching_resumes.sample(resumes_per_job, random_state=jidx)
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
    paired_df = generate_pairs(cleaned_job, cleaned_resume)
    print(f'\nGenerated {len(paired_df)} resume-job pairs.')
    print('\nSample:')
    print(paired_df.head())

if __name__ == '__main__':
    main()