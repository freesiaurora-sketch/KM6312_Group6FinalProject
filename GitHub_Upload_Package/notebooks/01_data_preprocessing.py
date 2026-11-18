"""Prepare the US Top 100 song-lyrics dataset for downstream analysis.

Inputs expected in one folder:
    all_songs_data.csv
    all_songs_data.json
    all_songs_data_processed.csv

Example:
    pip install pandas langdetect
    python song_data_preprocessing.py --input-dir . --output-dir results

The script does not translate lyrics. It detects the original language and
preserves all non-English songs for later multilingual analysis or translation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import regex as uregex


RAW_CSV = "all_songs_data.csv"
RAW_JSON = "all_songs_data.json"
PROCESSED_CSV = "all_songs_data_processed.csv"

KEY_COLUMNS = ["Song Title", "Artist", "Year", "Rank"]
EXPECTED_RAW_COLUMNS = [
    "Album",
    "Album URL",
    "Artist",
    "Featured Artists",
    "Lyrics",
    "Media",
    "Rank",
    "Release Date",
    "Song Title",
    "Song URL",
    "Writers",
    "Year",
]
PROCESSED_FEATURES = [
    "Verbs",
    "Nouns",
    "Adverbs",
    "Corpus",
    "Word Counts",
    "Unique Word Counts",
]

EXTREME_WORD_LIMIT = 3000
MANUAL_REVIEW_WORD_LIMIT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, validate, and summarize the song-lyrics dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Folder containing the three Kaggle files (default: current folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Folder for generated files (default: results).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum cleaned word count for a usable lyric (default: 5).",
    )
    parser.add_argument(
        "--skip-language",
        action="store_true",
        help="Skip language detection. Useful only for a quick test run.",
    )
    return parser.parse_args()


def require_files(input_dir: Path) -> dict[str, Path]:
    paths = {
        "raw_csv": input_dir / RAW_CSV,
        "raw_json": input_dir / RAW_JSON,
        "processed_csv": input_dir / PROCESSED_CSV,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n- " + "\n- ".join(missing)
        )
    return paths


def load_data(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    raw = pd.read_csv(paths["raw_csv"])
    # Use Python's standard JSON parser because some lyric strings contain
    # characters that pandas.read_json may interpret incorrectly.
    json_status = "success"
    try:
        with paths["raw_json"].open("r", encoding="utf-8") as file:
            json_records = json.load(file)
        json_df = pd.DataFrame(json_records)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        # JSON is a duplicate-format copy, not the analytical source. Continue
        # with the complete raw CSV and record the failed validation.
        json_df = pd.DataFrame()
        json_status = f"failed: {type(error).__name__}"
        print(
            "Warning: the JSON copy could not be parsed. "
            "CSV preprocessing will continue because JSON is validation-only."
        )
    processed = pd.read_csv(paths["processed_csv"])
    return raw, json_df, processed, json_status


def standardize_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce").astype("Int64")
    for column in ["Song Title", "Artist"]:
        df[column] = df[column].astype("string").str.strip()
    return df


def validate_inputs(
    raw: pd.DataFrame,
    json_df: pd.DataFrame,
    processed: pd.DataFrame,
    json_status: str,
) -> dict[str, object]:
    missing_columns = [c for c in EXPECTED_RAW_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"Raw CSV is missing columns: {missing_columns}")

    for name, df in [("processed CSV", processed)]:
        missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]
        if missing_keys:
            raise ValueError(f"{name} is missing key columns: {missing_keys}")

    csv_json_same_keys = False
    if not json_df.empty:
        missing_json_keys = [c for c in KEY_COLUMNS if c not in json_df.columns]
        if not missing_json_keys:
            raw_keys = standardize_key_types(raw)[KEY_COLUMNS].sort_values(KEY_COLUMNS)
            json_keys = standardize_key_types(json_df)[KEY_COLUMNS].sort_values(KEY_COLUMNS)
            raw_keys = raw_keys.reset_index(drop=True)
            json_keys = json_keys.reset_index(drop=True)
            csv_json_same_keys = bool(raw_keys.equals(json_keys))

    return {
        "raw_rows": len(raw),
        "json_rows": len(json_df) if not json_df.empty else np.nan,
        "processed_rows": len(processed),
        "raw_columns": raw.shape[1],
        "json_columns": json_df.shape[1] if not json_df.empty else np.nan,
        "processed_columns": processed.shape[1],
        "json_read_status": json_status,
        "csv_json_same_keys": csv_json_same_keys if not json_df.empty else "not_checked",
        "exact_duplicate_rows_raw": int(raw.duplicated().sum()),
        "duplicate_year_rank_raw": int(raw.duplicated(["Year", "Rank"]).sum()),
        "duplicate_song_artist_year_raw": int(
            raw.duplicated(["Song Title", "Artist", "Year"]).sum()
        ),
    }


def clean_lyrics(text: object) -> str:
    """Remove annotations while retaining letters from every Unicode script."""
    if pd.isna(text):
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = re.sub(r"\[[^\]]*\]", " ", value)  # [Verse], [Chorus: Artist], etc.
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.lower()
    value = re.sub(r"_", " ", value)
    value = uregex.sub(r"[^\p{L}\p{N}'’\-\s]", " ", value)
    value = uregex.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return uregex.findall(
        r"\p{L}+(?:['’\-]\p{L}+)*|\p{N}+",
        text.lower(),
    )


def count_words(text: str) -> int:
    return len(tokenize(text))


def count_unique_words(text: str) -> int:
    return len(set(tokenize(text)))


def normalize_identity(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).lower().strip()
    normalized = uregex.sub(r"[^\p{L}\p{N}]+", " ", normalized)
    return uregex.sub(r"\s+", " ", normalized).strip()


def detect_languages(texts: pd.Series, skip: bool) -> pd.DataFrame:
    if skip:
        return pd.DataFrame(
            {
                "Language": pd.Series("not_run", index=texts.index, dtype="string"),
                "Language Confidence": np.nan,
                "Language Reliable": False,
            }
        )

    try:
        from langdetect import DetectorFactory, LangDetectException, detect_langs
    except ImportError as exc:
        raise ImportError(
            "Language detection requires the 'langdetect' package. "
            "Install it with: pip install langdetect\n"
            "For a quick pipeline test only, add --skip-language."
        ) from exc

    DetectorFactory.seed = 0

    def safe_detect(text: str) -> tuple[str, float, bool]:
        tokens = tokenize(text)
        if len(tokens) < 20 or len(set(tokens)) < 10:
            return "unknown", np.nan, False
        try:
            result = detect_langs(text[:5000])[0]
            confidence = float(result.prob)
            reliable = bool(
                confidence >= 0.90
                and len(tokens) >= 50
                and len(set(tokens)) >= 20
            )
            return result.lang, confidence, reliable
        except LangDetectException:
            return "unknown", np.nan, False

    detected = texts.apply(safe_detect)
    result = pd.DataFrame(
        detected.tolist(),
        index=texts.index,
        columns=["Language", "Language Confidence", "Language Reliable"],
    )
    result["Language"] = result["Language"].astype("string")
    result["Language Reliable"] = result["Language Reliable"].astype(bool)
    return result


def merge_processed_features(raw: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    raw = standardize_key_types(raw)
    processed = standardize_key_types(processed)

    available_features = [c for c in PROCESSED_FEATURES if c in processed.columns]
    feature_df = processed[KEY_COLUMNS + available_features].copy()
    feature_df = feature_df.drop_duplicates(KEY_COLUMNS, keep="first")
    feature_df = feature_df.rename(
        columns={c: f"Existing Processed {c}" for c in available_features}
    )

    return raw.merge(feature_df, on=KEY_COLUMNS, how="left", validate="one_to_one")


def prepare_master(
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    min_words: int,
    skip_language: bool,
) -> pd.DataFrame:
    master = merge_processed_features(raw, processed)
    master["Release Date"] = pd.to_datetime(master["Release Date"], errors="coerce")

    master["Lyrics Missing"] = master["Lyrics"].isna()
    master["Clean Lyrics"] = master["Lyrics"].apply(clean_lyrics)
    master["Word Count New"] = master["Clean Lyrics"].apply(count_words)
    master["Unique Word Count New"] = master["Clean Lyrics"].apply(count_unique_words)
    master["Lexical Diversity"] = np.where(
        master["Word Count New"] > 0,
        master["Unique Word Count New"] / master["Word Count New"],
        np.nan,
    )
    master["Decade"] = (master["Year"] // 10 * 10).astype("Int64")
    language_info = detect_languages(master["Clean Lyrics"], skip_language)
    master = pd.concat([master, language_info], axis=1)
    master["Translation Required"] = np.where(
        master["Language Reliable"]
        & ~master["Language"].isin(["en", "unknown", "not_run"]),
        True,
        False,
    )

    master["Text Hash"] = master["Clean Lyrics"].apply(
        lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()
        if value
        else ""
    )
    # Title-only identity prevents the same chart song from being treated as
    # different when artist credit formatting changes across years.
    master["Song Identity"] = master["Song Title"].apply(normalize_identity)
    master["Same Text Song Count"] = 0
    nonempty = master["Text Hash"].ne("")
    master.loc[nonempty, "Same Text Song Count"] = (
        master.loc[nonempty]
        .groupby("Text Hash")["Song Identity"]
        .transform("nunique")
        .astype(int)
    )

    non_lyrics_pattern = (
        r"\bchapter\s+(?:one|two|three|i|ii|iii|1|2|3)\b"
        r"|\bscreenplay\b"
        r"|\brevised first draft\b"
        r"|\bfade in\b"
        r"|\bproject gutenberg\b"
        r"|\btable of contents\b"
        r"|\ba memoir of a childhood\b"
        r"|\bessays that still need\b"
        r"|\balphabetical order by title\b"
        r"|\bcan['’]?t find your song\b"
        r"|\b100 best songs\b"
        r"|\b100 greatest songs\b"
        r"|\branked by the .* community\b"
        r"|\boften when discussing the elite pack\b"
        r"|\balabama is filled with southern hospitality\b"
        r"|\bthe music of louisiana is one of\b"
        r"|\bthe catalogue of hits\b"
        r"|\bmen and women think differently\b"
    )
    master["Suspected Boilerplate"] = master["Clean Lyrics"].str.contains(
        non_lyrics_pattern,
        case=False,
        regex=True,
        na=False,
    )
    master["Extreme Length"] = master["Word Count New"] > EXTREME_WORD_LIMIT
    master["Suspected Reused Text"] = (
        (master["Word Count New"] >= 500)
        & (master["Same Text Song Count"] > 1)
    )
    master["Suspected Non-Lyrics"] = (
        master["Suspected Boilerplate"]
        | master["Extreme Length"]
        | master["Suspected Reused Text"]
    )
    master["Manual Review Required"] = (
        (master["Word Count New"] > MANUAL_REVIEW_WORD_LIMIT)
        & (master["Word Count New"] <= EXTREME_WORD_LIMIT)
        & ~master["Suspected Non-Lyrics"]
    )

    master["Valid for Text Modeling"] = (
        ~master["Lyrics Missing"]
        & (master["Word Count New"] >= min_words)
        & master["Clean Lyrics"].ne("")
        & ~master["Suspected Non-Lyrics"]
        & ~master["Manual Review Required"]
    )

    def exclusion_reason(row: pd.Series) -> str:
        if row["Lyrics Missing"]:
            return "missing_lyrics"
        if row["Clean Lyrics"] == "":
            return "empty_after_cleaning"
        if row["Word Count New"] < min_words:
            return f"fewer_than_{min_words}_words"
        if row["Suspected Non-Lyrics"]:
            return "suspected_non_lyrics"
        if row["Manual Review Required"]:
            return "long_text_manual_review"
        return ""

    master["Exclusion Reason"] = master.apply(exclusion_reason, axis=1)
    master = master.drop(columns=["Text Hash", "Song Identity"])
    return master


def build_quality_summary(
    master: pd.DataFrame,
    validation: dict[str, object],
    min_words: int,
) -> pd.DataFrame:
    metrics: list[tuple[str, object]] = [
        ("Original CSV rows", validation["raw_rows"]),
        ("Original JSON rows", validation["json_rows"]),
        ("Processed CSV rows", validation["processed_rows"]),
        ("Original CSV columns", validation["raw_columns"]),
        ("Original JSON columns", validation["json_columns"]),
        ("Processed CSV columns", validation["processed_columns"]),
        ("Original JSON read status", validation["json_read_status"]),
        ("CSV and JSON contain the same key records", validation["csv_json_same_keys"]),
        ("Exact duplicate rows in original CSV", validation["exact_duplicate_rows_raw"]),
        ("Duplicate Year-Rank combinations", validation["duplicate_year_rank_raw"]),
        ("Duplicate Song-Artist-Year combinations", validation["duplicate_song_artist_year_raw"]),
        ("Minimum year", int(master["Year"].min())),
        ("Maximum year", int(master["Year"].max())),
        ("Number of years", int(master["Year"].nunique())),
        ("Minimum songs per year", int(master.groupby("Year").size().min())),
        ("Maximum songs per year", int(master.groupby("Year").size().max())),
        ("Missing lyrics", int(master["Lyrics Missing"].sum())),
        ("Suspected non-lyrics", int(master["Suspected Non-Lyrics"].sum())),
        ("Long texts requiring manual review", int(master["Manual Review Required"].sum())),
        ("Lyrics longer than 3000 words", int(master["Extreme Length"].sum())),
        ("Reused long texts across different songs", int(master["Suspected Reused Text"].sum())),
        ("Valid text-modeling records", int(master["Valid for Text Modeling"].sum())),
        ("Excluded text-modeling records", int((~master["Valid for Text Modeling"]).sum())),
        ("Minimum words required", min_words),
        ("Missing release dates", int(master["Release Date"].isna().sum())),
        ("Missing album values", int(master["Album"].isna().sum())),
        (
            "Rows matched to processed Corpus",
            int(master.get("Existing Processed Corpus", pd.Series(dtype=object)).notna().sum()),
        ),
    ]
    return pd.DataFrame(metrics, columns=["Metric", "Value"])


def build_yearly_summary(master: pd.DataFrame) -> pd.DataFrame:
    summary_source = master.copy()
    summary_source["Modeling Word Count"] = summary_source["Word Count New"].where(
        summary_source["Valid for Text Modeling"]
    )
    summary_source["Modeling Unique Words"] = summary_source[
        "Unique Word Count New"
    ].where(summary_source["Valid for Text Modeling"])
    summary_source["Modeling Lexical Diversity"] = summary_source[
        "Lexical Diversity"
    ].where(summary_source["Valid for Text Modeling"])
    return (
        summary_source.groupby("Year", as_index=False)
        .agg(
            Total_Songs=("Rank", "size"),
            Valid_Lyrics=("Valid for Text Modeling", "sum"),
            Missing_Lyrics=("Lyrics Missing", "sum"),
            Suspected_Non_Lyrics=("Suspected Non-Lyrics", "sum"),
            Manual_Review=("Manual Review Required", "sum"),
            Average_Word_Count=("Modeling Word Count", "mean"),
            Median_Word_Count=("Modeling Word Count", "median"),
            Average_Unique_Words=("Modeling Unique Words", "mean"),
            Average_Lexical_Diversity=("Modeling Lexical Diversity", "mean"),
        )
        .sort_values("Year")
    )


def build_decade_summary(master: pd.DataFrame) -> pd.DataFrame:
    summary_source = master.copy()
    summary_source["Modeling Word Count"] = summary_source["Word Count New"].where(
        summary_source["Valid for Text Modeling"]
    )
    summary_source["Modeling Unique Words"] = summary_source[
        "Unique Word Count New"
    ].where(summary_source["Valid for Text Modeling"])
    summary_source["Modeling Lexical Diversity"] = summary_source[
        "Lexical Diversity"
    ].where(summary_source["Valid for Text Modeling"])
    return (
        summary_source.groupby("Decade", as_index=False)
        .agg(
            Total_Songs=("Rank", "size"),
            Valid_Lyrics=("Valid for Text Modeling", "sum"),
            Missing_Lyrics=("Lyrics Missing", "sum"),
            Suspected_Non_Lyrics=("Suspected Non-Lyrics", "sum"),
            Manual_Review=("Manual Review Required", "sum"),
            Average_Word_Count=("Modeling Word Count", "mean"),
            Median_Word_Count=("Modeling Word Count", "median"),
            Average_Unique_Words=("Modeling Unique Words", "mean"),
            Average_Lexical_Diversity=("Modeling Lexical Diversity", "mean"),
        )
        .sort_values("Decade")
    )


def build_language_summary(master: pd.DataFrame) -> pd.DataFrame:
    valid = master.loc[master["Valid for Text Modeling"]].copy()
    summary = (
        valid.groupby("Language", dropna=False)
        .size()
        .rename("Song Count")
        .reset_index()
        .sort_values("Song Count", ascending=False)
    )
    summary["Percentage"] = summary["Song Count"] / len(valid)
    return summary


def build_exclusion_summary(master: pd.DataFrame) -> pd.DataFrame:
    excluded = master.loc[~master["Valid for Text Modeling"]]
    return (
        excluded["Exclusion Reason"]
        .value_counts(dropna=False)
        .rename_axis("Exclusion Reason")
        .reset_index(name="Song Count")
    )


def save_plots(output_dir: Path, decade: pd.DataFrame, language: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_decade = decade.dropna(subset=["Decade"]).copy()
    plot_decade["Decade Label"] = (
        plot_decade["Decade"].astype(int).astype(str) + "s"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_decade["Decade Label"], plot_decade["Median_Word_Count"], color="#4C78A8")
    ax.set_title("Median Lyrics Word Count by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Median Word Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "median_word_count_by_decade.png", dpi=300)
    plt.close(fig)

    top_language = language.head(10).sort_values("Song Count")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_language["Language"].astype(str), top_language["Song Count"], color="#F58518")
    ax.set_title("Automatically Detected Lyrics Languages")
    ax.set_xlabel("Number of Modeling-Ready Songs")
    ax.set_ylabel("Automatically Detected Language")
    fig.tight_layout()
    fig.savefig(output_dir / "automatic_language_summary.png", dpi=300)
    plt.close(fig)


def write_readme(output_dir: Path, master: pd.DataFrame, min_words: int) -> None:
    valid_count = int(master["Valid for Text Modeling"].sum())
    excluded_count = len(master) - valid_count
    content = f"""Song Lyrics Preprocessing Outputs
==================================

Primary source:
- all_songs_data.csv

Validation-only source:
- all_songs_data.json (same underlying records as the original CSV)

Supplementary features:
- all_songs_data_processed.csv

Generated files:
1. songs_master_cleaned.csv
   All {len(master):,} original observations plus cleaned lyrics, language,
   quality flags, new text features, and matched processed-file features.

2. songs_modeling_ready.csv
   {valid_count:,} observations with usable lyrics and at least {min_words} cleaned words.

3. songs_excluded.csv
   {excluded_count:,} observations excluded from text modeling, with reasons.

4. songs_manual_review.csv
   Suspected non-lyrics and unusually long texts removed from modeling data.

5. data_quality_summary.csv
   Dataset dimensions, duplicates, missingness, and matching checks.

6. yearly_summary.csv
   Annual counts and descriptive text statistics.

7. decade_summary.csv
   Decade-level counts and descriptive text statistics.

8. language_summary.csv
   Automatically detected language counts among modeling-ready songs.

9. exclusion_summary.csv
   Counts of records excluded for each reason.

Important notes:
- The script does not translate lyrics.
- Keep non-English songs because the research concerns the US music market.
- Language labels are automatic auxiliary estimates, not official labels.
- Texts over {EXTREME_WORD_LIMIT:,} words, repeated long texts, and strong
  webpage/book/script markers are excluded as suspected non-lyrics.
- Texts between {MANUAL_REVIEW_WORD_LIMIT + 1:,} and {EXTREME_WORD_LIMIT:,}
  words are isolated for manual review.
- Use Clean Lyrics for sentiment analysis or later translation.
- For LDA, translate non-English Clean Lyrics first or use a multilingual method.
- Do not append the JSON file to the CSV file because this would duplicate records.
"""
    (output_dir / "README_preprocessing.txt").write_text(content, encoding="utf-8")


def save_outputs(
    master: pd.DataFrame,
    validation: dict[str, object],
    output_dir: Path,
    min_words: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    modeling = master.loc[master["Valid for Text Modeling"]].copy()
    excluded = master.loc[~master["Valid for Text Modeling"]].copy()
    manual_review = master.loc[
        master["Suspected Non-Lyrics"] | master["Manual Review Required"]
    ].copy()

    quality = build_quality_summary(master, validation, min_words)
    yearly = build_yearly_summary(master)
    decade = build_decade_summary(master)
    language = build_language_summary(master)
    exclusion_summary = build_exclusion_summary(master)

    csv_options = {"index": False, "encoding": "utf-8-sig"}
    master.to_csv(output_dir / "songs_master_cleaned.csv", **csv_options)
    modeling.to_csv(output_dir / "songs_modeling_ready.csv", **csv_options)
    excluded.to_csv(output_dir / "songs_excluded.csv", **csv_options)
    manual_review.to_csv(output_dir / "songs_manual_review.csv", **csv_options)
    quality.to_csv(output_dir / "data_quality_summary.csv", **csv_options)
    yearly.to_csv(output_dir / "yearly_summary.csv", **csv_options)
    decade.to_csv(output_dir / "decade_summary.csv", **csv_options)
    language.to_csv(output_dir / "language_summary.csv", **csv_options)
    exclusion_summary.to_csv(output_dir / "exclusion_summary.csv", **csv_options)
    save_plots(output_dir, decade, language)
    write_readme(output_dir, master, min_words)


def main() -> None:
    args = parse_args()
    paths = require_files(args.input_dir)
    raw, json_df, processed, json_status = load_data(paths)
    validation = validate_inputs(raw, json_df, processed, json_status)
    master = prepare_master(
        raw=raw,
        processed=processed,
        min_words=args.min_words,
        skip_language=args.skip_language,
    )
    save_outputs(master, validation, args.output_dir, args.min_words)

    print("Preprocessing completed successfully.")
    print(f"Original observations: {len(master):,}")
    print(
        "Modeling-ready observations: "
        f"{int(master['Valid for Text Modeling'].sum()):,}"
    )
    print(
        "Excluded observations: "
        f"{int((~master['Valid for Text Modeling']).sum()):,}"
    )
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
