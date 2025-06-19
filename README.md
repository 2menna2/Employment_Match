Skill Extraction Project
A Python project to extract technical and soft skills from job descriptions and standardize them against the ESCO taxonomy (v1.2.0) using the Google Gemini API and sentence transformers.
Overview
This project processes job descriptions to identify required skills, producing both raw skills (as listed in the description) and standardized skills (mapped to the ESCO taxonomy). It leverages the Google Gemini API (gemini-1.5-flash) for skill summarization and sentence-transformers/all-MiniLM-L6-v2 for embedding-based skill matching, with fuzzy matching as a fallback. The ESCO v1.2.0 skills taxonomy is sourced from skills_en.csv (converted to data/esco_skills.json), containing approximately 13,939 skills. Embeddings are precomputed using generate_embeddings.py and saved to data/esco_embeddings.npy for efficiency. The project is hosted at https://github.com/MahmoudSalama7/Employment_Match.
Features

Extracts technical and soft skills from job descriptions.
Standardizes skills against the ESCO v1.2.0 taxonomy using cosine similarity and fuzzy matching.
Supports batch processing for efficient handling of large ESCO datasets (~13,939 skills).
Precomputes and reuses ESCO skill embeddings for faster execution.
Configurable similarity and fuzzy matching thresholds for performance optimization.
Comprehensive logging with top-3 matches for troubleshooting skill standardization.
Converts ESCO CSV data (skills_en.csv) to JSON format for compatibility.

Prerequisites

Python: 3.10
Git: For cloning the repository
Google Gemini API Key: Obtain from https://makersuite.google.com/
ESCO v1.2.0 Skills Taxonomy: Download skills_en.csv from https://esco.ec.europa.eu/en/download (select English, CSV, Skills, Classification)
Git LFS: For handling large files (esco_skills.json, esco_embeddings.npy)

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


Replace your-gemini-api-key-here with your



