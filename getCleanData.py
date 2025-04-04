import kagglehub
import pandas as pd
import re, os, glob, ast

# Constants
DOWNLOAD_PATH = os.path.join(os.getcwd(), 'downloaded_datasets')
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

DATASETS = {
    "linkedin-job-postings": "arshkon/linkedin-job-postings",
    "new_resume": "saugataroyarghya/resume-dataset"
}

# Function to download datasets if not already present
def download_if_needed(name: str, slug: str) -> str:
    dataset_dir = os.path.join(DOWNLOAD_PATH, name)
    if os.path.exists(dataset_dir):
        print(f"Dataset '{name}' already exists at: {dataset_dir}")
    else:
        print(f"Downloading '{name}' from KaggleHub...")
        dataset_obj = kagglehub.dataset_download(slug)
        os.rename(dataset_obj, dataset_dir)
        print(f"Downloaded and moved to: {dataset_dir}")
    return dataset_dir

# Function to load CSV files from the dataset directory
def load_csv(file_path: str) -> pd.DataFrame:
    csv_file = glob.glob(os.path.join(file_path, '**', '*.csv'), recursive=True)
    if not csv_file:
        raise FileNotFoundError(f"No CSV files found in {file_path}")
    print(f'Found: {csv_file[0]}')
    return pd.read_csv(csv_file[0], encoding='utf-8')

# Function to clean text data
def clean_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Parse stringified lists like "['python', 'sql']"
def parse_list_string(val):
    if pd.isna(val): return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [clean_text(item) for item in parsed]
    except:
        pass
    return [clean_text(val)]

# Function to clean resume data
def clean_resume_data(resume_df: pd.DataFrame) -> pd.DataFrame:
    # Fix hidden BOM or weird characters in column names
    if '﻿job_position_name' in resume_df.columns:
        resume_df = resume_df.rename(columns={'﻿job_position_name': 'job_position_name'})

    cols_to_keep = [
        'job_position_name',
        'career_objective',
        'skills',
        'positions',
        'responsibilities',
        'degree_names',
        'major_field_of_studies',
        'certification_skills',
        'experiencere_requirement',
        'matched_score'
    ]
    resume_df = resume_df[[col for col in cols_to_keep if col in resume_df.columns]].copy()

    # Clean list-like columns and flatten them into space-separated strings
    for col in ['skills', 'certification_skills']:
        if col in resume_df.columns:
            resume_df[col] = resume_df[col].apply(parse_list_string)
            resume_df[col] = resume_df[col].apply(lambda x: ' '.join(x))

    # Apply basic text cleaning to all remaining object columns
    for col in resume_df.columns:
        if col not in ['matched_score'] and resume_df[col].dtype == object:
            resume_df[col] = resume_df[col].apply(clean_text)

    # Combine relevant fields into resume_text
    text_fields = [
        'job_position_name',
        'career_objective',
        'skills',
        'positions',
        'responsibilities',
        'degree_names',
        'major_field_of_studies',
        'certification_skills'
    ]
    text_fields = [col for col in text_fields if col in resume_df.columns]
    resume_df['resume_text'] = resume_df[text_fields].apply(
        lambda row: ' '.join(str(val).strip() for val in row if pd.notna(val)), axis=1
    )

    # Create weakly-supervised label
    resume_df['label'] = resume_df['matched_score'].apply(lambda x: 1 if x >= 0.7 else 0)

    return resume_df

# Function to clean job data
def clean_job_data(job_df: pd.DataFrame) -> pd.DataFrame:
    job_keep = ['job_id', 'title', 'description', 'skills_desc']
    job_df = job_df[job_keep]

    text_cols = ['title', 'description', 'skills_desc']
    cleaned = pd.DataFrame()
    cleaned['job_id'] = job_df['job_id']

    # Clean individual text fields
    for col in text_cols:
        cleaned[col.lower().replace(' ', '_')] = job_df[col].apply(clean_text)

    # Properly format job_text by stripping each field and joining with single spaces
    cleaned['job_text'] = cleaned[
        [col.lower().replace(' ', '_') for col in text_cols]
    ].apply(lambda row: ' '.join(str(val).strip() for val in row if pd.notna(val)), axis=1)

    return cleaned

# The main function to get cleaned datasets
def get_cleaned_datasets():
    print("Downloading datasets...")
    job_path = download_if_needed("linkedin-job-postings", DATASETS["linkedin-job-postings"])
    resume_path = download_if_needed("new_resume", DATASETS["new_resume"])

    job_df = load_csv(job_path)
    resume_df = load_csv(resume_path)

    print("Cleaning datasets...")
    cleaned_resume = clean_resume_data(resume_df)
    cleaned_job = clean_job_data(job_df)

    print("Cleaning complete.")
    return cleaned_resume, cleaned_job
