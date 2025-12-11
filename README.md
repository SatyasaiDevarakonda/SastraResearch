# SASTRA Research Finder - Author ID Version

Research discovery system using **Author ID as single source of truth** with **abstract-based matching** and **Gemini 1.5 Free Tier RAG**.

## 🎯 Core Principles

1. **Author ID = Single Source of Truth**
   - Every author maps to a unique Author ID from `Author(s) ID` column
   - Multiple name variants (Brindha, Brindha G.R., Brindha GR) → ONE Author ID
   - ALL searches aggregate by Author ID

2. **Abstract-Based Matching**
   - ALL searches use abstracts as primary field
   - Keywords extracted from abstracts boost matching
   - Bigram and trigram extraction for better relevance

3. **Dynamic Skill Extraction (NO Hardcoding)**
   - Skills extracted from project title
   - Enhanced with abstract keywords
   - NO predefined skills like "NLP", "CV", "ML"

4. **One-Click Expansion**
   - Click Author ID → Full profile
   - All name variants
   - All papers with full abstracts

5. **Gemini 1.5 Free Tier RAG**
   - Temperature: 0.1-0.2 for accuracy
   - Author ID references in output

## 📊 Database Stats

- **5,159** Publications
- **9,561** Unique Author IDs
- **27,169** Name Variants
- **195,420** Indexed Keywords/Phrases

## 📋 Features

### Phase 1: Keyword → Abstract Matching
```
Input: "machine learning, deep learning, classification"

Output Table:
| Author ID    | Author Name Variants        | Matching Papers |
|--------------|----------------------------|-----------------|
| 37061393600  | Ramachandran, R.            | 95              |
| 54888993500  | Vairavasundaram, V.         | 97              |
```

### Phase 2: Skill-Based Search
```
Input: "Deep learning based segmentation models for MRI analysis"

Extracted Skills (Dynamic):
☑ deep learning
☑ segmentation  
☑ mri
☑ neural networks
☑ medical imaging

Output: Same table format as Phase 1
```

### Author/ID Lookup
- **Search by Author ID**: Direct lookup
- **Search by Author Name**: Returns all matching Author IDs with scoring

### One-Click Expansion (Both Phases)
Clicking any Author ID shows:
- All name variants
- Total papers & citations
- All publications with **FULL ABSTRACTS**
- Research keywords

### RAG Analysis (Gemini 1.5)
```
Output:
## 1. KEY METHODS & TECHNIQUES
- Transfer learning, U-Net architecture...

## 2. REPRESENTATIVE PAPERS
- "Title" (AUTHOR_ID: 57192051462)

## 3. REQUIRED TECHNOLOGIES
- PyTorch, TensorFlow, MONAI...

## 4. RECOMMENDED RESEARCHERS
- Author ID: XXXXX - Brief expertise description

## 5. NEXT STEPS FOR DEVELOPERS
- Start with pre-trained models...
```

## 🚀 Quick Start

### Windows
```batch
run.bat
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing (REQUIRED first time)
python src/preprocess.py

# Start the app
streamlit run app.py
```

### If dependencies fail
```bash
python -m pip install --upgrade pip
pip install -U sentence-transformers
pip install faiss-cpu
```

## 📁 Structure

```
sastra-research-finder/
├── app.py                 # Streamlit UI
├── requirements.txt
├── run.bat
├── .env.example
├── README.md
├── data/
│   ├── SASTRA_Publications_2024-25.xlsx
│   ├── publications.pkl      (generated)
│   ├── author_profiles.pkl   (generated)
│   ├── mappings.pkl          (generated)
│   ├── keyword_index.pkl     (generated)
│   └── abstract_keywords.pkl (generated)
└── src/
    ├── __init__.py
    ├── preprocess.py      # Data preprocessing
    ├── search_engine.py   # Search logic
    └── gemini_rag.py      # RAG analysis
```

## 🔧 Enable Gemini (Optional)

1. Get free key: https://aistudio.google.com/app/apikey
2. Create `.env`:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
3. Restart app

## 📊 Data Flow

```
User Keywords → Lowercase → Multi-layer Search
                                    ↓
                    1. Exact keyword index match
                    2. Partial keyword match
                    3. Full-text abstract search
                                    ↓
                    Aggregate Results per Author ID
                                    ↓
    | Author ID | Name Variants | Matching Papers | Score |
                                    ↓
            One-Click → Full Profile + All Abstracts
```

## 🔍 Search Accuracy

The search engine uses multi-layer matching:
1. **Author Keywords** (weight: 3.0) - Highest priority
2. **Index Keywords** (weight: 2.0) - Second priority
3. **Title Keywords** (weight: 1.5) - Moderate priority
4. **Abstract Keywords** (weight: 1.0) - Base matching
5. **N-grams** (weight: 0.8) - Phrase matching

Author name search uses intelligent scoring:
- Exact match: 200 points
- Full name substring: 80 points
- Partial match: 25-40 points
- Minimum threshold: 15 points (filters false positives)

---
Built for SASTRA University | Gemini 1.5 Free Tier Compatible
