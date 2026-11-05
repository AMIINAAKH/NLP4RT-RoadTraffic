# NLP4RT-RoadTraffic: Urban Traffic Congestion Detection from Social Media

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset: Mendeley Data](https://img.shields.io/badge/DOI-10.17632%2Fc3xvj5snvv.1-orange.svg)](https://data.mendeley.com/datasets/c3xvj5snvv/1)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![Framework: PyTorch / HuggingFace](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-red.svg)]()

Official data and code repository for the research paper:  
**"Urban Traffic Congestion Detection from Social Media Using Transformer-Based Models and Retrieval-Augmented Generation"** (Springer Nature Applied Sciences / Springer Journal).

---

## 📌 Overview

This repository provides the official curated benchmark dataset and deterministic canonical splits for **NLP4RT-RoadTraffic**, a framework evaluating six transformer architectures (**BERT, ModernBERT, DeBERTa-v3, DistilBERT, GPT-2, and RoBERTa**) enhanced with a **Graph-Based Retrieval-Augmented Classification (Graph-RAC)** mechanism querying a spatial-semantic **Neo4j Knowledge Graph**.

```
                           ┌───────────────────────────┐
                           │   Raw Social Media Post   │
                           │   ("Accident on I-495...")│
                           └─────────────┬─────────────┘
                                         ▼
                           ┌───────────────────────────┐
                           │ SentenceTransformer Embed │
                           │     (all-MiniLM-L6-v2)    │
                           └─────────────┬─────────────┘
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Neo4j Knowledge Graph Query           │
                     │  - Dense Cosine Similarity (>= 0.9)   │
                     │  - Spatial/Location Co-occurrence     │
                     └───────────────────┬───────────────────┘
                                         ▼
                           ┌───────────────────────────┐
                           │ Context Prefix Synthesis  │
                           │ [CONTEXT] c1 [SEP] c2...  │
                           └─────────────┬─────────────┘
                                         ▼
                           ┌───────────────────────────┐
                           │  Transformer Classifier   │
                           │ (BERT / ModernBERT / GPT2)│
                           └─────────────┬─────────────┘
                                         ▼
                           ┌───────────────────────────┐
                           │ P(Traffic) vs Non-Traffic │
                           └───────────────────────────┘
```

---

## 📊 Dataset Description & Provenance

The benchmark corpus consists of **38,115 annotated social media posts** categorized into binary labels:
- **Class 0 (Non-Traffic)**: 19,190 instances (50.34%)
- **Class 1 (Traffic Incident / Conditions)**: 18,925 instances (49.66%)

### Constituent Open Data Sources
1. **Primary Corpus (97.5%)**: Curated from the open benchmark by **Dabiri & Heaslip (2018)**:  
   *Title*: *"Tweets with traffic-related labels for developing a Twitter-based traffic information system"*  
   *Mendeley Data DOI*: [`10.17632/c3xvj5snvv.1`](https://data.mendeley.com/datasets/c3xvj5snvv/1)  
   *Reference Paper*: Dabiri, S., & Heaslip, K. (2019). *"Developing a Twitter-based traffic event detection model using deep learning architectures"*, *Expert Systems with Applications*, 118, 425–439.
2. **Secondary Corpus (2.5%)**: Curated from the open NYC traffic benchmark by **Akshayaram et al. (2018)**:  
   *GitHub*: [`akshayaram95/Near-Real-Time-Road-Traffic-Event-Detection-using-Twitter`](https://github.com/akshayaram95/Near-Real-Time-Road-Traffic-Event-Detection-using-Twitter).

### Data Filtering Funnel ($52,425 \to 48,615 \to 38,115$)
1. **Raw Aggregation**: $51,093$ (Mendeley) + $1,332$ (NYC) = $52,425$ posts.
2. **Pass 1 (Cross-Source Deduplication)**: Exact string and retweet template removal eliminated $3,810$ duplicate posts $\to$ **$48,615$ unique posts** (reported in Section 4.1.1).
3. **Pass 2 (Quality Cleaning)**: Elimination of nulls, degenerate texts ($\le 4$ characters / $< 3$ semantic tokens after URL/mention stripping), automated bot feeds, and stratified balance adjustment $\to$ **$38,115$ curated posts**.
4. **Label Consistency**: Validated via a 1,000-tweet double-blind audit achieving **Cohen's $\kappa = 0.892$** (almost perfect agreement).

---

## 📁 Repository Structure

```
NLP4RT-RoadTraffic/
├── data/
│   ├── dataset.csv                  # Complete 38,115 annotated posts
│   └── dataset.xlsx                 # Excel version of curated benchmark
├── splits/
│   ├── train.csv                    # Canonical Training split (24,393 rows, 64.0%)
│   ├── val.csv                      # Canonical Validation split (6,099 rows, 16.0%)
│   ├── test.csv                     # Canonical Held-Out Test split (7,623 rows, 20.0%)
│   └── test_augmented.csv           # Paired Graph-Augmented Test split (7,623 rows)
├── LICENSE                          # MIT License
└── README.md                        # Documentation & Benchmark specifications
```

---

## ⚙️ Deterministic Canonical Splits

To guarantee complete reproducibility and eliminate test contamination (addressing Reviewer Remarks 1 & 2), all splits were generated using **Stratified Random Splitting (seed = 42)** with persistent integer identifiers (`id` from 0 to 7,622):

| Split | Number of Posts | Percentage | Class 0 (Non-Traffic) | Class 1 (Traffic) | Purpose |
|:---|:---:|:---:|:---:|:---:|:---|
| **`train.csv`** | **24,393** | 64.0% | 12,281 | 12,112 | Baseline & RAG fine-tuning |
| **`val.csv`** | **6,099** | 16.0% | 3,071 | 3,028 | Early stopping & hyperparameter tuning |
| **`test.csv`** | **7,623** | 20.0% | 3,838 | 3,785 | Canonical baseline held-out test set |
| **`test_augmented.csv`** | **7,623** | 20.0% | 3,838 | 3,785 | Strictly paired Graph-RAC test set |
| **Total** | **38,115** | **100.0%** | **19,190** | **18,925** | Full benchmark |

*Note: RandomOverSampler is applied exclusively to the training set. Neither the validation nor the test set contains synthetic or oversampled instances.*

---

## 🚀 Quick Start: Loading the Data

```python
import pandas as pd

# Load the full benchmark
df = pd.read_csv("data/dataset.csv")
print(f"Dataset Shape: {df.shape}")
print(df['Label'].value_counts())

# Load canonical paired test sets
test_base = pd.read_csv("splits/test.csv")
test_rag  = pd.read_csv("splits/test_augmented.csv")

# Verify pairing
assert len(test_base) == len(test_rag) == 7623
assert (test_base['id'] == test_rag['id']).all()
assert (test_base['Label'] == test_rag['Label']).all()
print(" Pairing verified: 100% identical test instances.")
```

---

## 📈 Summary of Paired Results (McNemar Test, $N = 7,623$)

Evaluated on the exact shared held-out test set ($N = 7,623$) with Edwards continuity correction and Holm-Bonferroni FWER step-down adjustment:

| Model | Base Acc (%) | RAG Acc (%) | $\Delta$ Acc | McNemar $\chi^2$ | $p$-value | $p_{\text{Holm}}$ | Significance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **BERT** | 98.47% | 98.37% | -0.09% | 0.3025 | 0.5823 | 1.0000 | Baseline Ceiling |
| **ModernBERT** | **97.81%** | **98.37%** | **+0.56%** | **11.6821** | **0.000631** | **0.003155** | **Statistically Significant ($p < 0.01$)** |
| **DeBERTa-v3** | 98.39% | 98.23% | -0.16% | 1.3750 | 0.2410 | 0.7229 | Performance Parity |
| **DistilBERT** | 98.52% | 98.22% | -0.30% | 4.3604 | 0.0368 | 0.1471 | Non-significant after Holm |
| **GPT-2** | **97.84%** | **98.50%** | **+0.67%** | **20.3252** | **0.000007** | **0.000039** | **Statistically Significant ($p < 0.001$)** |
| **RoBERTa** | 98.35% | 98.43% | +0.08% | 0.3676 | 0.5443 | 1.0000 | Baseline Ceiling |

---

## 📜 Citation

If you use this benchmark dataset or framework in your research, please cite:

```bibtex
@article{khandoudi2026urban,
  title={Urban Traffic Congestion Detection from Social Media Using Transformer-Based Models and Retrieval-Augmented Generation},
  author={Khandoudi, Amina and Makni, Aicha and Wolf, Armin and Ben Hedia, Béchir},
  journal={Springer Nature Applied Sciences},
  year={2026}
}
```

---

## 📄 License
This dataset and code are released under the [MIT License](LICENSE).
