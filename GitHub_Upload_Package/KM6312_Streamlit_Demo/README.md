# Six Decades of Billboard Lyrics — Interactive Analytics Demo

**Course**: KM6312 Information Mining & Analysis, Nanyang Technological University
**Project**: Topic and Sentiment Evolution in Billboard Year-End Charts (1959–2023)
**Team**: Team 6

This repository contains an interactive Streamlit application that presents the
results of our group project: data cleaning, exploratory data analysis (EDA),
LDA topic modelling, and regression analysis of 6,224 Billboard year-end chart
songs from 1959 to 2023.

---

## Option A — View the live demo (recommended, no installation)

The app is deployed on Streamlit Community Cloud:

**https://km6312group6finalproject-ndwgnkaaa46xbersjhwdjk.streamlit.app**

*(If the link is unavailable, the app can always be run locally — see Option B.)*

---

## Option B — Run locally (about 3 minutes)

You only need **Python 3.9+** installed. Then, in a terminal
(Terminal on macOS, Command Prompt / PowerShell on Windows):

```bash
# 1. Download or clone this repository, then enter the demo folder
cd KM6312_Streamlit_Demo

# 2. Install the required packages (one-time, ~1 minute)
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**.
If it does not, copy that address into any browser.

To stop the app, press `Ctrl + C` in the terminal.

### Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Use `python3 -m streamlit run app.py` instead |
| `pip: command not found` | Use `pip3 install -r requirements.txt` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` and open `http://localhost:8502` |
| `InconsistentVersionWarning` on startup | Harmless. The LDA model was trained with scikit-learn 1.4.2; `pip install scikit-learn==1.4.2` removes the warning |

---

## What the app contains

Eight modules, selectable from the left sidebar:

| Page | Content |
|---|---|
| **Overview** | Project summary, sample KPIs, language distribution, data quality notes |
| **EDA** | Lyric length and vocabulary richness trends (1959–2023), rank-group comparisons, decade summaries |
| **LDA Topics** | Preprocessing and topic-count comparison; five topic cards with keywords and representative songs |
| **Topic Trends** | Yearly topic-probability trends with 5-year smoothing; decade heatmap |
| **Regression** | Topic-evolution OLS slopes with Holm-corrected significance; sentiment logistic regression (odds ratios, test AUC); chart-rank regression with coefficient table |
| **Sentiment** | Four-emotion BERT classifier results: class KPIs, yearly positive/angry share trends with 5-year smoothing, topic × emotion heatmap, most confident predictions, model evaluation notes |
| **Song Explorer** | Search all 5,913 analysed songs by year, dominant topic, title, or artist; view per-song topic distribution and predicted emotion |
| **Try Your Lyrics** | Paste any English lyrics to get a live five-topic probability profile from our trained LDA model |

The **Sentiment** page reads pre-computed predictions from
`data/emotion_predictions_6075.csv` (a fine-tuned `bert-base-uncased`
classifier, 4 classes: Happy / Relaxed / Sad / Angry; test accuracy 0.60,
macro-F1 0.61). No BERT inference runs inside the app.

---

## Repository structure

```text
KM6312_Streamlit_Demo/
├── app.py                 # Streamlit application (all seven pages)
├── requirements.txt       # Python dependencies
├── README.md
├── data/                  # Aggregated analysis outputs (CSV)
│   ├── yearly_summary.csv
│   ├── decade_summary.csv
│   ├── language_summary.csv
│   ├── data_quality_summary.csv
│   ├── LDA_Topic_Descriptions.csv
│   ├── LDA_Yearly_Topic_Trends.csv
│   ├── LDA_Song_Level_For_Regression.csv
│   ├── LDA_Preprocessing_Comparison.csv
│   ├── LDA_Topic_Number_Comparison.csv
│   ├── topic_evolution_ols.csv
│   ├── sentiment_logit_coefficients.csv
│   ├── sentiment_logit_metrics.csv
│   ├── rank_ols_coefficients.csv
│   ├── rank_ols_metrics.csv
│   └── rank_by_dominant_topic.csv
└── model/
    └── LDA_5Topics_Model.joblib   # Refined 5-topic LDA (model + vectorizer)
```

## Data source and copyright note

The raw dataset is the public Kaggle dataset
[*Top 100 Songs & Lyrics from 1959 to 2019*](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019),
extended by our team to 2023.

**No raw lyrics are redistributed in this repository**, in line with copyright
guidance. The `data/` folder contains only aggregated statistics, topic
probabilities, and model outputs derived from the dataset.

## Reproducibility

- The LDA model was trained with **scikit-learn 1.4.2**; later versions load it
  correctly but may print an `InconsistentVersionWarning`.
- All charts in the app are rendered from the CSVs in `data/` — no external
  data or network access is required at runtime.
- The full analysis pipeline (data cleaning, EDA, LDA training, regression)
  is documented in the Jupyter notebooks included elsewhere in this repository.
