# -*- coding: utf-8 -*-
"""
KM6312 Group Project - Billboard Lyrics Analytics Demo
======================================================
Streamlit demo integrating the group's data cleaning, EDA and LDA
topic-modeling results (Billboard Year-End Hot 100, 1959-2023).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & constants
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Billboard Lyrics Analytics | KM6312",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

TOPIC_COLS = ["Topic_1", "Topic_2", "Topic_3", "Topic_4", "Topic_5"]
TOPIC_LABELS = {
    "Topic_1": "Narrative & Imagery",
    "Topic_2": "Attraction & Intimacy",
    "Topic_3": "Love, Life & Connection",
    "Topic_4": "Desire & Emotional Drive",
    "Topic_5": "Wealth, Status & Explicit Expression",
}
TOPIC_COLORS = {
    "Topic_1": "#4C78A8",
    "Topic_2": "#F58518",
    "Topic_3": "#E45756",
    "Topic_4": "#B279A2",
    "Topic_5": "#54A24B",
}

EMOTIONS = ["Happy", "Relaxed", "Sad", "Angry"]
EMO_COLORS = {
    "Happy": "#F2C14E",
    "Relaxed": "#72B7B2",
    "Sad": "#4C78A8",
    "Angry": "#E45756",
}

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem;}
.topic-card {
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    border-left: 8px solid; background: #f8f9fb;
}
.topic-card h4 {margin: 0 0 .3rem 0;}
.topic-card p {margin: 0; color: #555; font-size: .92rem;}
.kpi-note {color: #777; font-size: .85rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    data = {
        "songs": pd.read_csv(DATA_DIR / "LDA_Song_Level_For_Regression.csv"),
        "trends": pd.read_csv(DATA_DIR / "LDA_Yearly_Topic_Trends.csv"),
        "topics": pd.read_csv(DATA_DIR / "LDA_Topic_Descriptions.csv"),
        "prep_cmp": pd.read_csv(DATA_DIR / "LDA_Preprocessing_Comparison.csv"),
        "k_cmp": pd.read_csv(DATA_DIR / "LDA_Topic_Number_Comparison.csv"),
        "yearly": pd.read_csv(DATA_DIR / "yearly_summary.csv"),
        "decade": pd.read_csv(DATA_DIR / "decade_summary.csv"),
        "language": pd.read_csv(DATA_DIR / "language_summary.csv"),
        "quality": pd.read_csv(DATA_DIR / "data_quality_summary.csv"),
        "reg_topic": pd.read_csv(DATA_DIR / "topic_evolution_ols.csv"),
        "reg_sent_coef": pd.read_csv(DATA_DIR / "sentiment_logit_coefficients.csv"),
        "reg_sent_metrics": pd.read_csv(DATA_DIR / "sentiment_logit_metrics.csv"),
        "reg_rank_coef": pd.read_csv(DATA_DIR / "rank_ols_coefficients.csv"),
        "reg_rank_metrics": pd.read_csv(DATA_DIR / "rank_ols_metrics.csv"),
        "reg_rank_topic": pd.read_csv(DATA_DIR / "rank_by_dominant_topic.csv"),
        "emotion": pd.read_csv(DATA_DIR / "emotion_predictions_6075.csv"),
    }
    data["songs"]["Rank Group"] = pd.cut(
        data["songs"]["Rank"],
        bins=[0, 10, 50, 100],
        labels=["Top 10", "Rank 11-50", "Rank 51-100"],
    )
    # Attach emotion labels to the LDA song table (merge on Year + Rank,
    # the same key used in the emotion-analysis notebook).
    emo = data["emotion"][["year", "rank", "predicted_emotion",
                           "emotion_confidence"]].rename(
        columns={"year": "Year", "rank": "Rank",
                 "predicted_emotion": "Emotion",
                 "emotion_confidence": "Emotion Confidence"})
    data["songs"] = data["songs"].merge(emo, on=["Year", "Rank"], how="left")
    return data


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_DIR / "LDA_5Topics_Model.joblib")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load LDA model: {exc}")
        return None


DATA = load_data()


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def topic_display_name(topic_col: str) -> str:
    return f"{topic_col.replace('_', ' ')} · {TOPIC_LABELS[topic_col]}"


def bar_chart_topic_probs(probs, title="Topic probability distribution"):
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=[topic_display_name(t) for t in TOPIC_COLS],
            orientation="h",
            marker_color=[TOPIC_COLORS[t] for t in TOPIC_COLS],
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, max(0.15, float(np.max(probs)) * 1.25)], title="Probability"),
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def methodology_expander():
    with st.expander("ℹ️ Methodology notes & limitations (read before citing)"):
        st.markdown(
            """
- **Corpus**: 6,224 cleaned Billboard Year-End chart records (1959-2023).
  The LDA model uses the **5,913** records detected as English with reliable
  language detection. Results cannot be extrapolated to the remaining songs.
- **Topic labels are candidate labels** assigned by manual inspection of top
  keywords and representative songs — they are *not* validated categories from
  supervised learning. Always keep the `Topic_1`-style IDs alongside the labels.
- **Topic probabilities** sum to 1 per song (a distribution, not an accuracy).
  Yearly trends are *average topic probabilities* per year — descriptive, not
  causal evidence.
- **Lexical diversity** is Type-Token Ratio and is influenced by lyric length;
  it is not a direct measure of linguistic complexity.
- **LDA config**: `CountVectorizer` (term counts, `min_df=10`, `max_df=0.9`,
  English stopwords + lyric-specific filler words, plus `know`/`just` removed in
  the Refined variant), `LatentDirichletAllocation(n_components=5,
  random_state=42, max_iter=10, learning_method="batch")`.
- **Evaluation**: C_v coherence on 1,000 songs sampled with seed 42 (top-10
  words per topic); Topic Diversity = share of unique words among the top-12
  words across the 5 topics. Small C_v differences between preprocessing
  variants are not necessarily meaningful.
- Lyrics are copyrighted material: this demo intentionally ships **no raw
  lyrics**, only metadata, aggregate statistics and model outputs.
            """
        )


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_overview():
    st.header("🎵 Billboard Lyrics Analytics — Project Overview")
    st.markdown(
        "How did the **lyrics of Billboard Year-End chart songs (1959–2023)** evolve "
        "in length, vocabulary and topics? This demo integrates the group's data "
        "cleaning, exploratory data analysis (EDA) and LDA topic modeling results."
    )

    songs, yearly = DATA["songs"], DATA["yearly"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chart records (cleaned)", f"{6224:,}")
    c2.metric("Songs in LDA model", f"{len(songs):,}")
    c3.metric("Years covered", f"{int(yearly['Year'].min())}–{int(yearly['Year'].max())}")
    c4.metric("LDA topics", "5")
    st.caption(
        "LDA uses English records with reliable language detection only; "
        "the two sample sizes above are expected to differ."
    )

    st.subheader("Pipeline")
    st.markdown(
        """
1. **Data collection & cleaning** — Billboard Year-End charts merged with lyrics;
   missing / duplicated / boilerplate / non-lyric text filtered out; language
   detection applied.
2. **EDA** — lyric length, lexical diversity (TTR) and rank-group differences.
3. **LDA topic modeling** — three preprocessing variants and 5/8/10-topic models
   compared by C_v coherence & topic diversity; the **Refined 5-topic** model was
   selected.
4. **Next steps (teammates)** — sentiment classification, regression on topic /
   sentiment vs. chart rank.
        """
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Language distribution")
        lang = DATA["language"].head(8)
        fig = px.bar(
            lang, x="Language", y="Song Count",
            text="Song Count", color="Language",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, height=340,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"English accounts for {DATA['language'].iloc[0]['Percentage'] * 100:.1f}% "
            "of all cleaned records."
        )
    with right:
        st.subheader("Data quality summary")
        quality = DATA["quality"].dropna(how="all")
        st.dataframe(quality, use_container_width=True, hide_index=True, height=360)

    methodology_expander()


def page_eda():
    st.header("📊 Exploratory Data Analysis")
    yearly = DATA["yearly"].sort_values("Year")

    y0, y1 = st.slider(
        "Year range", int(yearly["Year"].min()), int(yearly["Year"].max()),
        (int(yearly["Year"].min()), int(yearly["Year"].max())),
    )
    yr = yearly[(yearly["Year"] >= y0) & (yearly["Year"] <= y1)]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average lyric length by year")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yr["Year"], y=yr["Average_Word_Count"], mode="lines",
            name="Mean", line=dict(color="#4C78A8", width=2)))
        fig.add_trace(go.Scatter(
            x=yr["Year"], y=yr["Median_Word_Count"], mode="lines",
            name="Median", line=dict(color="#F58518", width=2, dash="dot")))
        fig.update_layout(height=380, yaxis_title="Word count",
                          margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Average lexical diversity (TTR) by year")
        fig = px.line(yr, x="Year", y="Average_Lexical_Diversity")
        fig.update_traces(line_color="#E45756", line_width=2)
        fig.update_layout(height=380, yaxis_title="Type-Token Ratio",
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lyric length by Billboard rank group")
    songs = DATA["songs"]
    songs_r = songs[(songs["Year"] >= y0) & (songs["Year"] <= y1)]
    fig = px.box(
        songs_r, x="Rank Group", y="Word Count New", color="Rank Group",
        color_discrete_sequence=["#4C78A8", "#F58518", "#54A24B"],
        category_orders={"Rank Group": ["Top 10", "Rank 11-50", "Rank 51-100"]},
    )
    fig.update_layout(height=420, showlegend=False, yaxis_title="Word count",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    summary = (
        songs_r.groupby("Rank Group", observed=True)["Word Count New"]
        .agg(["count", "mean", "median"]).round(1)
    )
    st.dataframe(summary, use_container_width=True)

    st.subheader("Decade summary")
    st.dataframe(DATA["decade"], use_container_width=True, hide_index=True)
    st.caption(
        "Note: lexical diversity (TTR) tends to fall as lyrics get longer — "
        "the two trends are mechanically related, not independent findings."
    )


def page_topics():
    st.header("🧩 LDA Topic Modeling")

    st.subheader("Model selection")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Preprocessing variants (5 topics each)**")
        prep = DATA["prep_cmp"].melt("Model", var_name="Metric", value_name="Score")
        fig = px.bar(prep, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=["#4C78A8", "#F58518"])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Number of topics (Refined preprocessing)**")
        kcmp = DATA["k_cmp"].melt("Number of Topics", var_name="Metric", value_name="Score")
        kcmp["Number of Topics"] = kcmp["Number of Topics"].astype(str)
        fig = px.bar(kcmp, x="Number of Topics", y="Score", color="Metric",
                     barmode="group",
                     color_discrete_sequence=["#4C78A8", "#F58518"])
        fig.update_xaxes(type="category")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(fig, use_container_width=True)
    st.success(
        "Selected: **Refined, 5 topics** — best topic diversity (0.750) with "
        "comparable C_v coherence (0.440); more topics reduced coherence without "
        "gaining diversity."
    )

    st.subheader("The five topics (candidate labels)")
    topics = DATA["topics"]
    songs = DATA["songs"]
    for _, row in topics.iterrows():
        t = row["Topic"]
        color = TOPIC_COLORS[t]
        rep = songs.nlargest(3, t)[["Song Title", "Artist", "Year", t]]
        rep_str = " · ".join(
            f"*{r['Song Title']}* — {r['Artist']} ({int(r['Year'])})"
            for _, r in rep.iterrows()
        )
        st.markdown(
            f"""<div class="topic-card" style="border-left-color:{color}">
<h4 style="color:{color}">{t.replace('_', ' ')} · {row['Candidate Label']}</h4>
<p><b>Top keywords:</b> {row['Top 12 Keywords']}</p>
<p><b>Representative songs:</b> {rep_str}</p>
</div>""",
            unsafe_allow_html=True,
        )
    st.caption(
        "Candidate labels come from manual interpretation of keywords and "
        "representative songs; Topic_5 in particular must not be equated with a genre."
    )


def page_trends():
    st.header("📈 Topic Evolution, 1959–2023")
    trends = DATA["trends"].sort_values("Year")

    selected = st.multiselect(
        "Topics to display", TOPIC_COLS, default=TOPIC_COLS,
        format_func=topic_display_name,
    )
    smooth = st.checkbox("Apply 5-year rolling average", value=False)

    plot_df = trends.copy()
    if smooth:
        plot_df[TOPIC_COLS] = plot_df[TOPIC_COLS].rolling(5, center=True, min_periods=1).mean()

    fig = go.Figure()
    for t in selected:
        fig.add_trace(go.Scatter(
            x=plot_df["Year"], y=plot_df[t], mode="lines", name=topic_display_name(t),
            line=dict(color=TOPIC_COLORS[t], width=2)))
    fig.update_layout(
        height=460, yaxis_title="Average topic probability",
        yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.14),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Yearly values are the mean of per-song topic probabilities "
        "(descriptive statistics, not causal evidence)."
    )

    st.subheader("Topic share heatmap by decade")
    trends["Decade"] = (trends["Year"] // 10 * 10).astype(int)
    dec = trends.groupby("Decade")[TOPIC_COLS].mean()
    dec.columns = [topic_display_name(t) for t in TOPIC_COLS]
    fig = px.imshow(
        dec.T, text_auto=".1%", aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Decade", y="Topic", color="Avg. prob."),
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def page_explorer():
    st.header("🔍 Song Explorer")
    songs = DATA["songs"].copy()
    songs["Dominant Topic"] = songs[TOPIC_COLS].idxmax(axis=1)

    c1, c2, c3 = st.columns(3)
    with c1:
        years = st.slider("Year range", int(songs["Year"].min()),
                          int(songs["Year"].max()),
                          (int(songs["Year"].min()), int(songs["Year"].max())),
                          key="explorer_years")
    with c2:
        topic_filter = st.multiselect(
            "Dominant topic", TOPIC_COLS, default=[],
            format_func=topic_display_name, key="explorer_topics")
    with c3:
        query = st.text_input("Search title / artist", "")

    f = songs[(songs["Year"] >= years[0]) & (songs["Year"] <= years[1])]
    if topic_filter:
        f = f[f["Dominant Topic"].isin(topic_filter)]
    if query:
        mask = (f["Song Title"].str.contains(query, case=False, na=False)
                | f["Artist"].str.contains(query, case=False, na=False))
        f = f[mask]

    st.caption(f"{len(f):,} records match the filters "
               f"(LDA sample: {len(songs):,} songs).")
    show_cols = (["Song_ID", "Year", "Rank", "Song Title", "Artist",
                  "Word Count New", "Lexical Diversity", "Dominant Topic",
                  "Emotion"]
                 + TOPIC_COLS)
    st.dataframe(
        f[show_cols].style.format({t: "{:.3f}" for t in TOPIC_COLS}
                                  | {"Lexical Diversity": "{:.3f}"}),
        use_container_width=True, hide_index=True, height=380,
    )

    st.subheader("Inspect one song")
    if len(f) == 0:
        st.info("No songs under the current filters.")
        return
    options = (f["Year"].astype(str) + " · #" + f["Rank"].astype(str)
               + " · " + f["Song Title"] + " — " + f["Artist"])
    choice = st.selectbox("Pick a song", options.tolist())
    row = f.iloc[options.tolist().index(choice)]
    probs = row[TOPIC_COLS].to_numpy(dtype=float)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            bar_chart_topic_probs(probs, title=f"Topic mix — {row['Song Title']}"),
            use_container_width=True)
    with c2:
        st.metric("Dominant topic",
                  topic_display_name(row["Dominant Topic"]).split("·")[1].strip())
        st.metric("Dominant probability", f"{probs.max() * 100:.1f}%")
        emo = row.get("Emotion")
        if pd.notna(emo):
            st.metric("Predicted emotion",
                      f"{emo} ({row['Emotion Confidence'] * 100:.0f}%)")
        st.metric("Word count", f"{int(row['Word Count New'])}")
        st.metric("Lexical diversity", f"{row['Lexical Diversity']:.3f}")


def page_predict():
    st.header("🎤 Try Your Own Lyrics")
    st.markdown(
        "Paste **English** lyrics below and the Refined 5-topic LDA model will "
        "estimate the topic distribution — the same model behind the results pages."
    )
    package = load_model()
    if package is None:
        return

    default_text = (
        "I love the way you look at me tonight, "
        "your eyes are shining bright under the moonlight, "
        "baby let me hold you tight, everything will be alright"
    )
    text = st.text_area("Lyrics (English works best)", value=default_text, height=160)

    if st.button("Analyze topics", type="primary"):
        cleaned = text.strip()
        if len(cleaned.split()) < 5:
            st.warning("Please enter at least a few words of lyrics.")
            return
        dtm = package["vectorizer"].transform([cleaned])
        if dtm.nnz == 0:
            st.warning(
                "None of the words are in the model vocabulary "
                "(all were stopwords, too short, or unseen words). Try different lyrics."
            )
            return
        probs = package["model"].transform(dtm)[0]
        labels = package["topic_labels"]

        st.subheader("Estimated topic distribution")
        for label, p, t in zip(labels, probs, TOPIC_COLS):
            c1, c2 = st.columns([3, 1])
            c1.progress(float(p), text=f"{t.replace('_', ' ')} · {label}")
            c2.markdown(f"**{p * 100:.1f}%**")
        st.plotly_chart(bar_chart_topic_probs(probs, title="Your lyrics — topic mix"),
                        use_container_width=True)
        best = int(np.argmax(probs))
        st.success(
            f"Dominant topic: **Topic_{best + 1} · {labels[best]}** "
            f"({probs[best] * 100:.1f}%)"
        )
        st.caption(
            f"Vocabulary hits: {dtm.nnz} unique in-vocabulary tokens "
            f"(vocabulary size {len(package['vectorizer'].vocabulary_):,})."
        )

    with st.expander("⚠️ Interpretation caveats"):
        st.markdown(
            """
- The model was trained on **cleaned English lyrics**; the saved package applies
  the same tokenizer + stopword list, but not the full upstream cleaning pipeline,
  so results on raw pasted text are indicative rather than perfectly comparable.
- Short texts, non-English lyrics, or texts with many out-of-vocabulary words
  produce unstable estimates.
- Topic labels are human-assigned candidate labels, not validated categories.
            """
        )


# ----------------------------------------------------------------------------
# Sidebar & routing
# ----------------------------------------------------------------------------
def page_regression():
    st.header("📈 Regression Analysis")
    st.caption(
        "Three regression families on the 5,913-song LDA sample — topic-share "
        "evolution (OLS on yearly means), sentiment evolution (logistic) and "
        "Billboard rank (OLS). Lower rank number = better chart position."
    )

    # --- 1. Topic evolution ------------------------------------------------
    st.subheader("1 · Topic-share evolution (OLS on yearly means, n = 65 years)")
    te = DATA["reg_topic"].copy()
    te["slope_pp"] = te["Slope_per_decade"] * 100
    fig = go.Figure(
        go.Bar(
            x=te["slope_pp"],
            y=[topic_display_name(t) for t in te["Topic"]],
            orientation="h",
            marker_color=[TOPIC_COLORS[t] for t in te["Topic"]],
            text=[f"{v:+.1f}" for v in te["slope_pp"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="#999", line_width=1)
    fig.update_layout(
        xaxis_title="Change in average topic share (percentage points per decade)",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    show = te[["Topic", "Label", "Mean_share", "slope_pp", "R2", "p_HAC_Holm", "Sig_Holm"]].copy()
    show["Mean_share"] = (show["Mean_share"] * 100).round(1).astype(str) + "%"
    show["slope_pp"] = show["slope_pp"].round(2)
    show["R2"] = show["R2"].round(3)
    show["p_HAC_Holm"] = show["p_HAC_Holm"].map(lambda p: f"{p:.2g}")
    show.columns = ["Topic", "Candidate label", "Mean share", "Slope (pp/decade)",
                    "R²", "p (Newey–West, Holm-adj.)", "Sig."]
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.markdown(
        "<span class='kpi-note'>Wealth, Status & Explicit Expression rises by "
        "≈ +4.7 pp/decade (R² = 0.74) — the strongest trend; Narrative & Imagery "
        "and Love, Life & Connection decline steadily. Quadratic terms further "
        "improve Topics 1, 3 and 5 (acceleration after ~2000).</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- 2. Sentiment logit --------------------------------------------------
    st.subheader("2 · Sentiment evolution (logistic regression, positive = 1)")
    sc = DATA["reg_sent_coef"].rename(columns={"Unnamed: 0": "Term"})
    sm = DATA["reg_sent_metrics"].rename(columns={"Unnamed: 0": "Split"})
    test_row = sm[sm["Split"] == "Test (20%)"].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test AUC", f"{test_row['AUC']:.3f}")
    m2.metric("Test accuracy", f"{test_row['Accuracy']:.3f}",
              f"baseline {test_row['Majority_baseline_acc']:.3f}")
    m3.metric("McFadden R² (full model)", f"{test_row['McFadden_R2(full)']:.3f}")
    m4.metric("Songs used", "5,684")
    disp = sc[["Term", "odds_ratio", "OR_ci_low", "OR_ci_high", "p_value", "sig"]].copy()
    disp["odds_ratio"] = disp["odds_ratio"].round(3)
    disp["OR_ci_low"] = disp["OR_ci_low"].round(3)
    disp["OR_ci_high"] = disp["OR_ci_high"].round(3)
    disp["p_value"] = disp["p_value"].map(lambda p: f"{p:.2g}")
    disp.columns = ["Term", "Odds ratio", "CI low", "CI high", "p value", "Sig."]
    st.dataframe(disp, hide_index=True, use_container_width=True)
    st.markdown(
        "<span class='kpi-note'>Odds of a positive lyric fall by ≈ 14% per decade "
        "(OR ≈ 0.86) after controlling for topics, length and lexical diversity; "
        "≈ 20% per decade (OR ≈ 0.80) with year alone.</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- 3. Rank regression --------------------------------------------------
    st.subheader("3 · Billboard rank (OLS; lower rank = better position)")
    rk = dict(zip(DATA["reg_rank_metrics"]["Metric"], DATA["reg_rank_metrics"]["Value"]))
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("R²", f"{rk.get('R2', float('nan')):.3f}")
    r2.metric("10-fold CV R²", f"{rk.get('CV10_R2', float('nan')):.3f}")
    r3.metric("RMSE (ranks)", f"{rk.get('RMSE_in_sample', float('nan')):.1f}")
    r4.metric("n", f"{int(rk.get('n', 0)):,}")
    rc = DATA["reg_rank_coef"].rename(columns={"Unnamed: 0": "Term"})
    rd = rc[["Term", "coef", "std_beta", "p_value", "sig_HC3"]].copy()
    rd["coef"] = rd["coef"].round(3)
    rd["std_beta"] = rd["std_beta"].round(3)
    rd["p_value"] = rd["p_value"].map(lambda p: f"{p:.2g}")
    rd.columns = ["Term", "Coef (ranks)", "Std. beta", "p value", "Sig. (HC3)"]
    st.dataframe(rd, hide_index=True, use_container_width=True)

    rt = DATA["reg_rank_topic"].sort_values("mean")
    fig2 = go.Figure(
        go.Bar(
            x=rt["mean"],
            y=rt["dominant"],
            orientation="h",
            error_x=dict(type="data", array=rt["sem"]),
            marker_color=[TOPIC_COLORS[t] for t in ["Topic_4", "Topic_2", "Topic_3", "Topic_5", "Topic_1"]],
            text=[f"{m:.1f} (n={n:,})" for m, n in zip(rt["mean"], rt["size"])],
            textposition="outside",
        )
    )
    fig2.update_layout(
        title="Mean year-end rank by dominant topic (lower = better)",
        xaxis=dict(range=[0, 70], title="Mean rank"),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "<span class='kpi-note'>Lyric text features explain almost nothing about "
        "chart position (R² ≈ 0.01): year-end rank is driven mainly by factors "
        "outside the lyrics.</span>",
        unsafe_allow_html=True,
    )

    with st.expander("⚠️ Interpretation caveats"):
        st.markdown(
            """
- The sentiment feature used here is **lexicon-based** (162 positive / 190
  negative words), not the output of the sentiment-classification module;
  the two should not be conflated in the report.
- Topic shares sum to 1, so `Topic_3` was dropped as the reference topic.
- Songs appearing in multiple year-end charts can leak across random
  train/test splits; the time-based hold-out (train ≤ 2009, test ≥ 2010)
  is the stricter evaluation.
            """
        )


def page_sentiment():
    st.header("🎭 Emotion Classification")
    emo = DATA["emotion"].copy()

    st.markdown(
        "A fine-tuned **BERT** classifier (`bert-base-uncased`) labels each of "
        f"the **{len(emo):,}** analysed songs with one of four emotions — "
        "**Happy, Relaxed, Sad or Angry** — together with the four-class "
        "probabilities. Predictions are pre-computed; this page explores them."
    )

    counts = emo["predicted_emotion"].value_counts()
    cols = st.columns(4)
    for col, e in zip(cols, EMOTIONS):
        n = int(counts.get(e, 0))
        col.metric(e, f"{n:,} songs", f"{n / len(emo) * 100:.1f}%")

    emo["positive"] = emo["predicted_emotion"].isin(["Happy", "Relaxed"])
    yr = (emo.groupby("year")
          .agg(pos=("positive", "mean"),
               angry=("predicted_emotion", lambda s: (s == "Angry").mean()))
          .reset_index().sort_values("year"))
    yr["pos_sm"] = yr["pos"].rolling(5, center=True, min_periods=1).mean()
    yr["ang_sm"] = yr["angry"].rolling(5, center=True, min_periods=1).mean()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Positive share by year (Happy + Relaxed)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yr["year"], y=yr["pos"] * 100, mode="lines", name="Yearly",
            line=dict(color="#72B7B2", width=1), opacity=0.6))
        fig.add_trace(go.Scatter(
            x=yr["year"], y=yr["pos_sm"] * 100, mode="lines",
            name="5-year average", line=dict(color="#1F9E97", width=3)))
        fig.update_layout(height=380, yaxis_title="% of songs",
                          margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Angry share by year")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yr["year"], y=yr["angry"] * 100, mode="lines", name="Yearly",
            line=dict(color="#F2A2A0", width=1), opacity=0.6))
        fig.add_trace(go.Scatter(
            x=yr["year"], y=yr["ang_sm"] * 100, mode="lines",
            name="5-year average", line=dict(color="#E45756", width=3)))
        fig.update_layout(height=380, yaxis_title="% of songs",
                          margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Emotion mix within each LDA topic")
    songs = DATA["songs"].dropna(subset=["Emotion"]).copy()
    songs["Dominant Topic"] = songs[TOPIC_COLS].idxmax(axis=1)
    ct = (pd.crosstab(songs["Dominant Topic"], songs["Emotion"],
                      normalize="index") * 100)[EMOTIONS]
    ct.index = [topic_display_name(t) for t in ct.index]
    fig = px.imshow(ct, text_auto=".1f", color_continuous_scale="Reds",
                    labels=dict(x="Predicted emotion", y="", color="%"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Topic 5 (Wealth, Status & Explicit Expression) is dominated by Angry "
        "songs (~56%), versus 5–11% in the other four topics — the rise of "
        "Topic 5 and the rise of Angry are two views of the same shift."
    )

    st.subheader("Most confident predictions")
    c1, c2 = st.columns(2)
    for col, e in zip((c1, c2), ["Happy", "Angry"]):
        top = (emo[emo["predicted_emotion"] == e]
               .nlargest(10, "emotion_confidence")
               [["song_title", "artist", "year", "emotion_confidence"]]
               .rename(columns={"song_title": "Song", "artist": "Artist",
                                "year": "Year",
                                "emotion_confidence": "Confidence"}))
        col.markdown(f"**Top 10 {e}**")
        col.dataframe(top.style.format({"Confidence": "{:.0%}"}),
                      hide_index=True, use_container_width=True)

    with st.expander("ℹ️ Model & evaluation details"):
        st.markdown(
            """
- **Model**: `bert-base-uncased` fine-tuned for 4-class emotion
  classification (Happy / Relaxed / Sad / Angry).
- **Training data**: 1,000 songs labelled by an emotion lexicon
  (700 train / 150 validation / 150 test).
- **Baselines compared**: TF-IDF + Logistic Regression (validation macro-F1
  0.31), balanced LR (0.51), balanced Linear SVM (0.54); the fine-tuned BERT
  achieved the best validation macro-F1 (0.61) and was selected.
- **Test performance** (150 held-out songs): accuracy **0.60**,
  macro-F1 **0.61**. Song-level labels are therefore noisy; the trends shown
  here rely on aggregation over thousands of songs, not individual labels.
- **Labels are silver labels**: the lexicon assigned the training labels, so
  the classifier partly *distils* the lexicon — agreement between the two
  sentiment measurements is not fully independent evidence.
- Emotion coverage: 6,075 songs; 5,867 of them match the LDA modelling sample
  (merged on year + chart rank).
            """
        )


PAGES = {
    "Overview": page_overview,
    "EDA": page_eda,
    "LDA Topics": page_topics,
    "Topic Trends": page_trends,
    "Regression": page_regression,
    "Sentiment": page_sentiment,
    "Song Explorer": page_explorer,
    "Try Your Lyrics": page_predict,
}

st.sidebar.title("🎵 Billboard Lyrics")
st.sidebar.caption("KM6312 Group Project Demo")
page = st.sidebar.radio("Module", list(PAGES.keys()))
st.sidebar.divider()
st.sidebar.markdown(
    "**Data**: Billboard Year-End charts, 1959–2023  \n"
    "**Models**: Refined 5-topic LDA · fine-tuned BERT (4 emotions)  \n"
    "**Modules shipped here**: data cleaning + EDA + LDA + regression "
    "+ emotion classification"
)
st.sidebar.divider()
st.sidebar.caption(
    "Demo for the group report and Zoom presentation. "
    "No raw lyrics are bundled, in line with copyright guidance."
)

PAGES[page]()
