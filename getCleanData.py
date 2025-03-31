import kagglehub
import pandas as pd
import re, os, glob

# Constants
DOWNLOAD_PATH = os.path.join(os.getcwd(), 'downloaded_datasets')
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

DATASETS = {
    "linkedin-job-postings": "arshkon/linkedin-job-postings",
    "resume-screening": "mdtalhask/ai-powered-resume-screening-dataset-2025"
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
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# Function to clean resume data
def clean_resume_data(resume_df: pd.DataFrame) -> pd.DataFrame:
    resume_df = resume_df.drop(['Name', 'Salary Expectation ($)'], axis=1)
    
    # Define columns
    text_cols = ['Skills', 'Education', 'Certifications', 'Job Role']
    numeric_map = {
        'Experience (Years)': 'experience',
        'Projects Count': 'projects',
        'AI Score (0-100)': 'ai_score'
    }

    # Build cleaned DataFrame
    cleaned = pd.DataFrame()
    cleaned['resume_id'] = resume_df['Resume_ID']
    
    # Clean text columns
    for col in text_cols:
        cleaned[col.lower().replace(' ', '_')] = resume_df[col].apply(clean_text)

    # Combine into full text
    cleaned['resume_text'] = cleaned[[col.lower().replace(' ', '_') for col in text_cols]].agg(' '.join, axis=1)

    # Numeric columns
    for raw_col, clean_col in numeric_map.items():
        cleaned[clean_col] = pd.to_numeric(resume_df[raw_col], errors='coerce').fillna(0)

    # Recruiter decision
    cleaned['recruiter_decision'] = resume_df['Recruiter Decision'].str.lower().fillna('')

    return cleaned

# Function to clean job data
def clean_job_data(job_df: pd.DataFrame) -> pd.DataFrame:
    job_keep = ['job_id', 'title', 'description', 'skills_desc', 'formatted_experience_level']
    job_df = job_df[job_keep]

    text_cols = ['title', 'description', 'skills_desc', 'formatted_experience_level']
    cleaned = pd.DataFrame()
    cleaned['job_id'] = job_df['job_id']

    for col in text_cols:
        cleaned[col.lower().replace(' ', '_')] = job_df[col].apply(clean_text)

    cleaned['job_text'] = cleaned[[col.lower().replace(' ', '_') for col in text_cols]].agg(' '.join, axis=1)

    return cleaned

# The main function to get cleaned datasets
def get_cleaned_datasets():
    job_path = download_if_needed("linkedin-job-postings", DATASETS["linkedin-job-postings"])
    resume_path = download_if_needed("resume-screening", DATASETS["resume-screening"])

    job_df = load_csv(job_path)
    resume_df = load_csv(resume_path)

    cleaned_resume = clean_resume_data(resume_df)
    cleaned_job = clean_job_data(job_df)

    return cleaned_resume, cleaned_job
