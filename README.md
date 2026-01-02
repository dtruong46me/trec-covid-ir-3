# TREC COVID IR

## Data Directory Structure

```
data/
├── raw/                                         # Original, unprocessed datasets from TREC-COVID
│   └── CORD_19/                                 # CORD-19 corpus and TREC-COVID evaluation resources
│       ├── cord_19_embeddings_2020-05-19.csv    # Pre-computed document embeddings for semantic search
│       ├── metadata.csv                         # Document metadata (titles, authors, abstracts, DOIs, etc.)
│       ├── topics-rnd3.csv                      # Query topics for Round 3 (50 COVID-19 information needs)
│       ├── qrels.csv                            # Relevance judgments (ground truth for evaluation metrics)
│       ├── docids-rnd3.txt                      # Document IDs eligible for Round 3 submission
│       ├── submission.csv                       # Template file showing required submission format
│       ├── pdf_json/                            # Full-text papers extracted from PDFs (hash-named files)
│       │   ├── 0001418189999fea7f7cbe3e82703d71c85a6fe5.json
│       │   ├── 0003793cf9e709bc2b9d0c8111186f78fb73fc04.json
│       │   └── ...
│       └── pmc_json/                            # Full-text papers from PubMed Central (PMC ID format)
│           ├── PMC1054884.xml.json
│           ├── PMC1065028.xml.json
│           └── ...
└── processed/                                   # Preprocessed datasets (cleaned text, custom embeddings, etc.)
                                                 # Currently empty - populated during pipeline execution
```

---
