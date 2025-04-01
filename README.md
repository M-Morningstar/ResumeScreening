# Automated Resume Screener
This project is an AI-powered system designed to automate the resume screening process using Natural Language Processing (NLP) and Machine Learning. The goal is to improve efficiency in candidate selection by analyzing resume content and matching it to relevant job postings.

## Features
- Preprocessing and cleaning of resume and job posting datasets
- Semantic matching between job titles and resume roles using Sentence-BERT
- Generation of job–resume pairs with weak supervision
- Embedding-based similarity matching
- Support for feature-based ML model training to predict candidate fit

## Project Structure
```
.
├── getData.py           # Downloads and cleans resume and job datasets
├── processing.py        # Pairs resumes with jobs using SBERT-based role matching
├── downloaded_datasets/ # Automatically ignored dataset cache (see .gitignore)
├── requirements.txt     # Project dependencies
├── README.md            # Project overview and setup
```

## Dataset Sources
- [AI-Powered Resume Screening Dataset (Kaggle)](https://www.kaggle.com/datasets/mdtalhask/ai-powered-resume-screening-dataset-2025)
- [LinkedIn Job Postings (Kaggle)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/automated-resume-screener.git
cd automated-resume-screener
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To download datasets, clean them, and generate paired data:
```bash
python processing.py
```
This will:
- Download and clean both datasets
- Match resumes to semantically similar job postings
- Output a labeled dataset for model training

## Notes
- Dataset files are automatically stored in `downloaded_datasets/` and excluded from version control via `.gitignore`.
- Labels are weakly supervised based on resume AI score and recruiter decisions.

## Future Improvements
- Train and evaluate machine learning models on the paired dataset
- Integrate skill and experience-level comparison into feature engineering
- Add bias detection and resume feedback generation

## License
This project is for academic and educational purposes.
