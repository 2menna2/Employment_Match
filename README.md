Skill Extraction Project
A Python project to extract technical and soft skills from job descriptions and standardize them against the ESCO taxonomy (v1.2.0) using the Google Gemini API and sentence transformers.
Overview
This project processes job descriptions to identify required skills, producing both raw skills (as listed in the description) and standardized skills (mapped to the ESCO taxonomy). It leverages the Google Gemini API (gemini-1.5-flash) for skill summarization and sentence-transformers/all-MiniLM-L6-v2 for embedding-based skill matching, with fuzzy matching as a fallback. The ESCO v1.2.0 skills taxonomy is sourced from skills_en.csv (converted to data/esco_skills.json), containing approximately 13,939 skills. Embeddings are precomputed using generate_embeddings.py and saved to data/esco_embeddings.npy for efficiency. The project is hosted at https://github.com/MahmoudSalama7/Employment_Match.
Features

Extracts technical and soft skills from job descriptions.
Standardizes skills against the ESCO v1.2.0 taxonomy using cosine similarity and fuzzy matching.
Supports batch processing for efficient handling of large ESCO datasets (~13,939 skills).
Precomputes and reuses ESCO skill embeddings for faster execution.
Configurable similarity threshold and batch size for performance optimization.
Comprehensive logging to troubleshoot skill matching issues.
Converts ESCO CSV data (skills_en.csv) to JSON format for compatibility.

Prerequisites

Python: 3.10
Git: For cloning the repository
Google Gemini API Key: Obtain from https://makersuite.google.com/
ESCO v1.2.0 Skills Taxonomy: Download skills_en.csv from https://esco.ec.europa.eu/en/download (select English, CSV, Skills, Classification)
Optional: Git LFS for handling large JSON files (esco_skills.json)

Setup

Clone the Repository:
git clone https://github.com/MahmoudSalama7/Employment_Match.git
cd Employment_Match


Create and Activate a Virtual Environment:
python -m venv employ
.\employ\Scripts\activate  # On Windows
source employ/bin/activate  # On Linux/Mac


Install Dependencies:
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.1 -i https://download.pytorch.org/whl/cpu


The -i https://pypi.tuna.tsinghua.edu.cn/simple uses a mirror for faster downloads in some regions.
torch is installed from a CPU-specific wheel to avoid GPU dependencies.


Set Up the Gemini API Key:

Create a .env file in the project root:echo GEMINI_API_KEY=your-gemini-api-key-here > .env


Replace your-gemini-api-key-here with your key from https://makersuite.google.com/.


Prepare the ESCO Skills Taxonomy:

Download the ESCO v1.2.0 skills dataset (skills_en.csv) from https://esco.ec.europa.eu/en/download.
Move skills_en.csv to the data/ folder:mkdir data
copy C:\Users\LENOVO\Downloads\skills_en.csv\skills_en.csv data\skills_en.csv  # On Windows
cp ~/Downloads/skills_en.csv/skills_en.csv data/skills_en.csv  # On Linux/Mac


Convert skills_en.csv to data/esco_skills.json:python convert_esco_to_json.py




Generate Precomputed Embeddings:

Run the embedding generation script (only needed once or when esco_skills.json changes):python generate_embeddings.py


This creates data/esco_embeddings.npy. If it fails with a name 'torch' is not defined error, ensure the script includes import torch and reinstall dependencies.


Run the Script:
python extract_skills.py


This processes a sample job description and outputs extracted and standardized skills in JSON format.



Usage

Modify the Job Description:

Edit the job_description variable in extract_skills.py:job_description = """Your custom job description here."""


Run the script to extract skills:python extract_skills.py




Expected Output:
{
  "standardized": [
    "Python (computer programming)",
    "Java (computer programming)",
    "SQL",
    "Agile project management",
    "communication",
    "solve problems"
  ],
  "raw": [
    "Python",
    "Java",
    "SQL",
    "problem-solving skills",
    "Agile methodologies",
    "communication skills"
  ]
}


Configuration Options:

SIMILARITY_THRESHOLD (default: 0.4): Controls embedding-based skill matching sensitivity.
FUZZY_THRESHOLD (default: 90): Controls fuzzy matching sensitivity for low-similarity skills.
BATCH_SIZE (default: 100): Adjusts memory usage for embedding large ESCO datasets.
EMBEDDER_MODEL: Switch to sentence-transformers/all-mpnet-base-v2 for improved matching if needed.



Folder Structure

data/:
skills_en.csv: ESCO v1.2.0 skills taxonomy (CSV)
esco_skills.json: Converted ESCO skills in JSON format
esco_embeddings.npy: Precomputed ESCO skill embeddings


docs/: Documentation (currently empty)
tests/: Unit tests (currently empty)
extract_skills.py: Main skill extraction script
convert_esco_to_json.py: Script to convert ESCO CSV to JSON
generate_embeddings.py: Script to precompute ESCO embeddings
requirements.txt: Python dependencies
.env: Environment variables (not tracked by Git)
.gitignore: Excludes .env, virtual environment, and large data files
README.md: This file

Dependencies
Listed in requirements.txt:
transformers==4.44.2
torch==2.0.1
sentence-transformers==2.2.2
numpy==1.25.2
scikit-learn==1.3.0
accelerate==0.21.0
setuptools==70.0.0
python-dotenv==1.0.1
google-generativeai==0.7.2
rapidfuzz==3.9.7

Troubleshooting

Empty or Incorrect standardized Skills:

Check logs for Top-3 matches and Fuzzy match entries in the terminal output.
Lower SIMILARITY_THRESHOLD to 0.3 or FUZZY_THRESHOLD to 80 in extract_skills.py:SIMILARITY_THRESHOLD = 0.3
FUZZY_THRESHOLD = 80


Try a stronger embedder:EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"


Update requirements.txt:echo "sentence-transformers==2.7.0" >> requirements.txt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


Regenerate embeddings:python generate_embeddings.py






Memory Issues:

Reduce BATCH_SIZE to 50 in extract_skills.py and generate_embeddings.py if processing ~13,939 skills causes errors:BATCH_SIZE = 50




Gemini API Errors:

Verify your API key in .env.
Check usage limits at https://makersuite.google.com/.
Review error messages in logs and consult https://ai.google.dev/docs.


Embedding Generation Errors:

If generate_embeddings.py fails, ensure import torch is included and torch==2.0.1 is installed.
Verify data/esco_skills.json exists and is valid.


Slow Downloads:

Install hf_xet for faster Hugging Face model downloads:pip install hf_xet -i https://pypi.tuna.tsinghua.edu.cn/simple




Large File Handling:

If data/esco_skills.json or data/esco_embeddings.npy exceeds 100MB, use Git LFS:git lfs install
git lfs track "data/esco_skills.json"
git lfs track "data/esco_embeddings.npy"
git add .gitattributes data/esco_skills.json data/esco_embeddings.npy
git commit -m "Track large files with Git LFS"
git push origin main





Additional ESCO Files
The ESCO v1.2.0 dataset in C:\Users\LENOVO\Downloads\skills_en.csv\ includes additional files:

occupations_en.csv: Occupation taxonomy.
occupationSkillRelations_en.csv: Maps skills to occupations.
skillGroups_en.csv: Hierarchical skill groups.
Others: Specialized collections (e.g., digitalSkillsCollection_en.csv, greenSkillsCollection_en.csv).

Only skills_en.csv is used currently. To incorporate files like occupationSkillRelations_en.csv for job-skill matching, extend the script as needed.
Contributing

Fork the repository: https://github.com/MahmoudSalama7/Employment_Match.
Create a feature branch:git checkout -b feature/your-feature


Commit changes:git commit -m "Add your feature"


Push to the branch:git push origin feature/your-feature


Open a pull request.

License
MIT License
Contact
For issues or feature requests, open an issue at https://github.com/MahmoudSalama7/Employment_Match/issues.
