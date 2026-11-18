# Six Decades of Billboard Lyrics: Topic and Sentiment Evolution (1959–2023)

**Course**: KM6312 Information Mining & Analysis, Nanyang Technological University
**Team**: Team 6

This repository contains the dataset and code for our group project, which analyses
how the lyrics of Billboard year-end chart songs have evolved over six decades.
We combine exploratory data analysis, LDA topic modelling, regression analysis, and
a fine-tuned BERT emotion classifier, and we present the results through an
interactive Streamlit application.

**Live demo**: https://km6312group6finalproject-ndwgnkaaa46xbersjhwdjk.streamlit.app *(deployed on Streamlit Community Cloud)*

## Repository structure

```text
.
├── notebooks/                     # All analysis code, in pipeline order
│   ├── 01_data_preprocessing.py       # Data cleaning: language detection, quality flags
│   ├── 02_EDA_LDA.ipynb               # EDA + LDA topic modelling (5 topics)
│   ├── 03_regression_analysis.py      # Topic-evolution OLS, sentiment logit, rank OLS
│   ├── 04_emotion_preprocessing.ipynb # Emotion data prep (lexicon-labelled training set)
│   ├── 05_emotion_classification.ipynb# TF-IDF baselines + BERT fine-tuning, evaluation
│   └── 06_emotion_analysis.ipynb      # Emotion trends, emotion × topic, emotion × rank
├── data/
│   ├── songs_modeling_ready_no_lyrics.csv  # Cleaned master dataset, 6,224 songs
│   │                                       # (lyrics columns removed for copyright)
│   └── emotion_predictions_6075.csv        # Song-level emotion predictions, 6,075 songs
│                                           # (label + 4-class probabilities, no lyrics)
├── KM6312_Streamlit_Demo/         # Interactive demo (see its own README for how to run)
│   ├── app.py
│   ├── requirements.txt
│   ├── data/                        # Aggregated CSVs the app reads
│   └── model/LDA_5Topics_Model.joblib
└── README.md
```

## Methods at a glance

| Module | Approach | Key outputs |
|---|---|---|
| Data cleaning | Language detection, duplicate/boilerplate flags | 6,224 chart records, 1959–2023 |
| EDA | Lyric length, lexical diversity (TTR) trends | Length 182 → 400+ words; TTR 0.41 → 0.36 |
| Topic modelling | LDA, 5 topics (scikit-learn) | Topic 5 (wealth/explicit) +4.7 pp per decade, R² = 0.74 |
| Regression | OLS on yearly topic shares; logit on sentiment; OLS on rank | Rank R² ≈ 0.01 — lyrics barely predict chart position |
| Emotion classification | BERT (bert-base-uncased) fine-tuned on 1,000 lexicon-labelled songs; 4 classes (Happy / Relaxed / Sad / Angry) | Test accuracy 0.60, macro-F1 0.61; positive share 82% → 57% across six decades, Angry 5% → 31% |

## Data source and copyright note

The raw dataset is the public Kaggle dataset
[*Top 100 Songs & Lyrics from 1959 to 2019*](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019),
extended by our team to 2023.

**No raw lyrics are redistributed in this repository**, in line with copyright
guidance. The CSVs in `data/` contain metadata and derived features only;
lyrics columns have been removed. Song-level emotion predictions contain labels
and probabilities only.

## Reproducibility

- Run order: `notebooks/01` → `02` → `03` (regression) and `04` → `05` → `06` (emotion).
- The LDA model was trained with **scikit-learn 1.4.2**; other versions may print an
  `InconsistentVersionWarning` when loading `model/LDA_5Topics_Model.joblib`.
- The fine-tuned BERT weights (~440 MB) are **not hosted** in this repository due to
  GitHub file-size limits. To reproduce them, run `notebooks/05_emotion_classification.ipynb`,
  which trains the model from the lexicon-labelled training split and saves
  `final_BERT_model/` locally.
- To run the interactive demo locally, see `KM6312_Streamlit_Demo/README.md`
  (three commands: install requirements, then `streamlit run app.py`).
