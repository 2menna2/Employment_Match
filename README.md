# 🧠 Skill Extraction with ESCO + Gemini  
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