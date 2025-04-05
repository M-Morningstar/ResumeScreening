import pandas as pd
import numpy as np
import torch
import faiss
import re
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from getCleanData import get_cleaned_datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# === Helper Functions ===

def extract_years(text):
    if pd.isna(text): return 0
    match = re.search(r'(\d+)', str(text))
    return int(match.group(1)) if match else 0

def compute_title_match_score(job_title, resume_title):
    # Force CPU here so GPU isn't blocked
    embeddings = model.encode([job_title, resume_title], convert_to_tensor=True, device='cpu')
    sim = cosine_similarity(
        embeddings[0].cpu().numpy().reshape(1, -1),
        embeddings[1].cpu().numpy().reshape(1, -1)
    )[0][0]
    return sim

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def faiss_top_k_search(index, job_embeddings, top_k=5):
    faiss.normalize_L2(job_embeddings)
    scores, indices = index.search(job_embeddings, top_k)
    return scores, indices

# === Precompute Embeddings ===

def encode_resumes(cleaned_resume, cache_path='resume_embeddings.pt'):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    texts = cleaned_resume['resume_text'].tolist()
    emb = model.encode(texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True)
    torch.save(emb, cache_path)
    return emb

def encode_jobs(texts):
    return model.encode(texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True).cpu()

# === Pairing with FAISS + Chunks ===

def generate_pairs_faiss(
    cleaned_job,
    cleaned_resume,
    resume_embeddings,
    chunk_size=1000,
    resumes_per_job=3,
    similarity_threshold=0.65
):
    all_pairs = []

    print("\nBuilding FAISS index...")
    index = build_faiss_index(resume_embeddings.numpy())

    for i in range(0, len(cleaned_job), chunk_size):
        job_chunk = cleaned_job.iloc[i:i + chunk_size]
        job_texts = job_chunk['job_text'].tolist()
        job_emb = encode_jobs(job_texts)

        print(f"\nProcessing job chunk {i} to {i + len(job_chunk)}...")

        scores, indices = faiss_top_k_search(index, job_emb.numpy(), top_k=50)  # search more, filter later

        for j, (_, job) in enumerate(job_chunk.iterrows()):
            job_id = job['job_id']
            job_title = job['title']
            job_text = job['job_text']
            job_len = len(job_text.split())

            job_scores = scores[j]
            job_indices = indices[j]

            matches = []
            seen_positions = set()

            def compute_features(resume_idx):
                resume = cleaned_resume.iloc[resume_idx]
                similarity = job_scores[list(job_indices).index(resume_idx)]

                position = resume.get('job_position_name', '')
                if position in seen_positions:
                    return None  # Skip duplicates

                title_score = compute_title_match_score(job_title, position)
                resume_years = extract_years(resume.get('experiencere_requirement', ''))

                return {
                    'job_id': job_id,
                    'resume_id': resume['resume_id'],
                    'job_title': job_title,
                    'resume_position': position,
                    'job_text': job_text,
                    'resume_text': resume['resume_text'],
                    'similarity_score': similarity,
                    'experience_years': resume_years,
                    'resume_length': len(resume['resume_text'].split()),
                    'job_length': job_len,
                    'title_match_score': title_score,
                    'label': resume['label']
                }

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(compute_features, idx): idx for idx in job_indices}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        if result['similarity_score'] >= similarity_threshold or len(matches) < resumes_per_job:
                            matches.append(result)
                            seen_positions.add(result['resume_position'])
                    if len(matches) >= resumes_per_job:
                        break

            all_pairs.extend(matches)

    return pd.DataFrame(all_pairs)

# === Main ===

def main():
    # cleaned_resume, cleaned_job = get_cleaned_datasets()
    cleaned_resume = pd.read_csv('cleaned_resume.csv')
    cleaned_job = pd.read_csv('cleaned_job.csv')
    resume_embeddings = encode_resumes(cleaned_resume)

    cleaned_job = cleaned_job.head(1000)
    
    paired_df = generate_pairs_faiss(cleaned_job, cleaned_resume, resume_embeddings, chunk_size=1000, resumes_per_job=3)

    print(f'\nGenerated {len(paired_df)} resume-job pairs.')
    print('\nSample:')
    print(paired_df.head())

    paired_df.to_csv('paired_data.csv', index=False)
    print('Saved to paired_data.csv')

if __name__ == '__main__':
    main()