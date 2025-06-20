# 🧠 Skill Extraction 
**Extract technical and soft skills from job descriptions and CVs and standardize them using the ESCO taxonomy (v1.2.0)**

[![GitHub Stars](https://img.shields.io/github/stars/MahmoudSalama7/Employment_Match)](https://github.com/MahmoudSalama7/Employment_Match)  
[![MIT License](https://img.shields.io/github/license/MahmoudSalama7/Employment_Match)](LICENSE)

---

## 🔍 Overview  
This Python-based tool extracts **technical** and **soft skills** from job descriptions or CVs (PDF or text) and maps them to the standardized [ESCO skills taxonomy](https://esco.ec.europa.eu/en/download) using:

- **Google Gemini API** (`gemini-1.5-flash`)
- **Sentence Transformers** (`all-MiniLM-L6-v2`)
- **Fuzzy Matching** as fallback

---

## 🚀 Features

✅ Extracts skills from **text or PDF** (via PyPDF2)  
✅ Maps to ESCO v1.2.0 (~13,939 skills)  
✅ Uses **sentence-transformers** for semantic matching  
✅ Supports **batch processing** for fast lookups  
✅ Precomputes and caches embeddings for speed  
✅ Top-3 match logging for debugging  
✅ Configurable thresholds for matching  
✅ Converts ESCO CSV to JSON

---

## 📦 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MahmoudSalama7/Employment_Match.git
cd Employment_Match
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv employ
.\employ\Scripts\activate         # Windows
source employ/bin/activate        # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.1 -i https://download.pytorch.org/whl/cpu
```

---

## 🔑 Configure Gemini API

Create a `.env` file in the root:

```bash
echo GEMINI_API_KEY=your-gemini-api-key-here > .env
```

Replace with your actual key from [Google MakerSuite](https://makersuite.google.com/).

---

## 📚 Prepare ESCO Dataset

1. Download `skills_en.csv` from [ESCO v1.2.0 Download Page](https://esco.ec.europa.eu/en/download)  
2. Move it to the `data/` folder:

```bash
mkdir data
# Windows
copy C:\Users\LENOVO\Downloads\skills_en.csv\skills_en.csv data\skills_en.csv
# Linux/Mac
cp ~/Downloads/skills_en.csv/skills_en.csv data/skills_en.csv
```

3. Convert CSV to JSON:

```bash
python convert_esco_to_json.py
```

---

## 🧠 Generate ESCO Embeddings

```bash
python generate_embeddings.py
```

This creates `data/esco_embeddings.npy`.

---

## 🛠️ Usage

1. Modify `job_description` in `extract_skills.py`:

```python
job_description = """Your job description text here."""
```

2. Run the script:

```bash
python extract_skills.py
```

### ✅ Sample Output

```json
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
```

---

## ⚙️ Configuration

| Parameter              | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `SIMILARITY_THRESHOLD` | 0.4     | Cosine similarity threshold                    |
| `FUZZY_THRESHOLD`      | 90      | Fallback fuzzy match threshold                 |
| `BATCH_SIZE`           | 100     | Controls memory use for batch embedding        |
| `EMBEDDER_MODEL`       | `all-MiniLM-L6-v2` | Can switch to `all-mpnet-base-v2` |

---

## 📁 Project Structure

```
Employment_Match/
│
├── data/
│   ├── skills_en.csv
│   ├── esco_skills.json
│   └── esco_embeddings.npy
│
├── extract_skills.py
├── convert_esco_to_json.py
├── generate_embeddings.py
├── requirements.txt
├── .env
└── README.md
```

---

## 🧪 Troubleshooting

### ❌ Empty Standardized Skills?

- Lower similarity thresholds:

```python
SIMILARITY_THRESHOLD = 0.3
FUZZY_THRESHOLD = 80
```

- Switch embedder:

```python
EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

Then:

```bash
pip install sentence-transformers==2.7.0
python generate_embeddings.py
```

### 🧠 Memory Issues?

Reduce batch size:

```python
BATCH_SIZE = 50
```

### 🔐 Gemini API Errors?

- Check key in `.env`
- Check usage limits on MakerSuite
- Review logs

---

## 🧬 Extend the Dataset

ESCO also provides:
- `occupationSkillRelations_en.csv`: Link occupations to skills
- `skillGroups_en.csv`: Grouped skill hierarchies

_Not yet integrated, but useful for future development._

---

# Employment Match FastAPI

A comprehensive FastAPI application that provides AI-powered skill extraction and matching capabilities for job candidates and requirements.

## Features

- **Skill Extraction**: Extract skills from job descriptions and CVs (text or PDF)
- **Skill Standardization**: Map skills to ESCO taxonomy using AI
- **Skill Matching**: Match CV skills against job requirements
- **Complete Workflow**: End-to-end matching process
- **Web Interface**: Modern HTML frontend for easy testing
- **API Documentation**: Auto-generated OpenAPI docs

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file with your Gemini API key:

```bash
echo "GEMINI_API_KEY=your-gemini-api-key-here" > .env
```

### 3. Prepare Data (if not already done)

```bash
# Convert ESCO CSV to JSON
python convert_esco_to_json.py

# Generate embeddings
python generate_embeddings.py
```

### 4. Start the Server

```bash
python start_server.py
```

Or directly with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/ui
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Core Endpoints

#### 1. Extract Job Skills

```http
POST /extract-job-skills
Content-Type: application/json

{
  "job_description": "We are seeking a Software Engineer...",
  "similarity_threshold": 0.6,
  "fuzzy_threshold": 90
}
```

#### 2. Extract CV Skills (Text)

```http
POST /extract-cv-skills-text
Content-Type: application/json

{
  "cv_text": "EXPERIENCE\nSoftware Engineer...",
  "similarity_threshold": 0.4,
  "fuzzy_threshold": 90
}
```

#### 3. Extract CV Skills (PDF)

```http
POST /extract-cv-skills-pdf
Content-Type: multipart/form-data

file: [PDF file]
similarity_threshold: 0.4
fuzzy_threshold: 90
```

#### 4. Match Skills

```http
POST /match-skills
Content-Type: application/json

{
  "cv_skills": ["Python", "JavaScript", "SQL"],
  "job_skills": ["Python", "Java", "SQL", "Agile"],
  "similarity_threshold": 0.3,
  "fuzzy_threshold": 80
}
```

#### 5. Complete Matching Workflow

```http
POST /complete-matching
Content-Type: multipart/form-data

job_description: "We are seeking a Software Engineer..."
cv_text: "EXPERIENCE\nSoftware Engineer..."
# OR cv_file: [PDF file]
job_similarity_threshold: 0.6
cv_similarity_threshold: 0.4
match_similarity_threshold: 0.3
```

### Utility Endpoints

#### Health Check

```http
GET /health
```

#### ESCO Skills Info

```http
GET /esco-skills
```

#### Setup Data

```http
POST /setup-data
```

## Configuration

### Thresholds

- **Similarity Threshold** (0.0-1.0): Controls embedding-based matching sensitivity

  - Higher values = more strict matching
  - Lower values = more lenient matching

- **Fuzzy Threshold** (0-100): Controls fuzzy string matching sensitivity
  - Higher values = more strict matching
  - Lower values = more lenient matching

### Default Values

- Job Skills Extraction: similarity=0.6, fuzzy=90
- CV Skills Extraction: similarity=0.4, fuzzy=90
- Skill Matching: similarity=0.3, fuzzy=80

## Response Formats

### Skills Response

```json
{
  "standardized": ["Python (computer programming)", "SQL"],
  "raw": ["Python", "SQL"]
}
```

### Match Response

```json
{
  "match_score": 75.0,
  "matched_skills": [
    {
      "cv_skill": "Python",
      "job_skill": "Python",
      "similarity": 0.95
    }
  ],
  "missing_skills": ["Java"],
  "extra_skills": ["JavaScript"]
}
```

## Testing

### Using the Web Interface

1. Open http://localhost:8000/ui
2. Choose the appropriate tab for your use case
3. Fill in the required information
4. Click the submit button
5. View results in real-time

### Using the API Documentation

1. Open http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"

### Using the Test Script

```bash
python test_api.py
```

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not found"**

   - Create a `.env` file with your API key
   - Restart the server

2. **"Models not properly loaded"**

   - Check if data files exist in `data/` directory
   - Run the setup scripts if needed

3. **"ESCO skills not loaded"**

   - Run `python convert_esco_to_json.py`
   - Ensure `data/esco_skills.json` exists

4. **"Embeddings not found"**

   - Run `python generate_embeddings.py`
   - Ensure `data/esco_embeddings.npy` exists

5. **"Module object is not callable"**
   - This has been fixed in the latest version
   - Restart the server if you see this error

### Performance Tips

- Use appropriate similarity thresholds for your use case
- For large-scale processing, consider batch operations
- Monitor memory usage when processing large PDFs

## Development

### Project Structure

```
Employment_Match/
├── main.py                 # FastAPI application
├── start_server.py         # Server startup script
├── test_api.py            # API testing script
├── static/
│   └── index.html         # Web interface
├── data/                  # Data files
├── extract_skills.py      # Job skills extraction
├── extract_cv_skills.py   # CV skills extraction
├── match_skills.py        # Skill matching
├── generate_embeddings.py # Embedding generation
└── convert_esco_to_json.py # Data conversion
```

### Adding New Endpoints

1. Define Pydantic models for request/response
2. Create the endpoint function
3. Add proper error handling
4. Update this documentation

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

## License

This project is licensed under the same terms as the original Employment Match project.




---

## 📌 Contributing

1. Fork the repo  
2. Create a feature branch:

```bash
git checkout -b feature/your-feature
```

3. Commit and push:

```bash
git commit -m "Your feature"
git push origin feature/your-feature
```

4. Open a Pull Request

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for more.

---

## 📬 Contact

Got questions or feature requests?  
👉 [Open an issue](https://github.com/MahmoudSalama7/Employment_Match/issues)
