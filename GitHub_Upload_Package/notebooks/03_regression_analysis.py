"""
KM6312 - Regression Analysis (follows EDA_LDA.ipynb)
=====================================================
1. Topic Evolution      : OLS  (yearly mean topic probability ~ Year)
2. Sentiment Evolution  : Logistic Regression (positive vs negative lyrics)
3. Billboard Rank       : OLS  (Rank ~ topics + sentiment + length + lexical diversity + year)

Inputs  (same folder layout as the uploaded files):
    songs_modeling_ready.csv, LDA_5Topics_Model.joblib
Outputs:
    ../results/*.csv, ../figures/*.png

NOTE: statsmodels is not required. OLS / Logit (IRLS) / robust SE / diagnostics
are implemented with numpy + scipy so the script runs in a minimal environment.
"""
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_DIR = Path("/mnt/user-data/uploads")          # <- change if running elsewhere
OUT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR, RES_DIR = OUT_DIR / "figures", OUT_DIR / "results"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RES_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------
# Small statistics toolkit (OLS / Logit / robust covariance / diagnostics)
# ----------------------------------------------------------------------------
def ols(X, y, names, cov="nonrobust", lag=3):
    """OLS with intercept already inside X. cov: nonrobust | HC3 | HAC (Newey-West)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    sse, sst = resid @ resid, ((y - y.mean()) ** 2).sum()
    if cov == "nonrobust":
        V = XtX_inv * sse / (n - k)
    elif cov == "HC3":
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        Xe = X * (resid / (1 - h))[:, None]
        V = XtX_inv @ (Xe.T @ Xe) @ XtX_inv
    elif cov == "HAC":
        Xe = X * resid[:, None]
        S = Xe.T @ Xe
        for l in range(1, lag + 1):
            G = Xe[l:].T @ Xe[:-l]
            S += (1 - l / (lag + 1)) * (G + G.T)
        V = XtX_inv @ S @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    tcrit = stats.t.ppf(0.975, n - k)
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    F = ((sst - sse) / (k - 1)) / (sse / (n - k)) if k > 1 else np.nan
    Fp = stats.f.sf(F, k - 1, n - k) if k > 1 else np.nan
    ll = -n / 2 * (np.log(2 * np.pi) + np.log(sse / n) + 1)
    table = pd.DataFrame({"coef": beta, "std_err": se, "t": t, "p_value": p,
                          "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se}, index=names)
    return dict(table=table, beta=beta, fitted=fitted, resid=resid, n=n, k=k, sse=sse,
                r2=r2, adj_r2=adj, F=F, F_p=Fp, rmse=np.sqrt(sse / n), aic=2 * k - 2 * ll,
                bic=k * np.log(n) - 2 * ll, dw=np.sum(np.diff(resid) ** 2) / sse, cov=cov, X=X, y=y)


def logit(X, y, names, max_iter=100, tol=1e-10):
    """Logistic regression by IRLS. X includes intercept."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    W = mu * (1 - mu)
    V = np.linalg.inv(X.T @ (X * W[:, None]))
    se = np.sqrt(np.diag(V))
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    ll = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    p0 = y.mean()
    ll0 = n * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    lr = 2 * (ll - ll0)
    table = pd.DataFrame({"coef": beta, "std_err": se, "z": z, "p_value": p,
                          "odds_ratio": np.exp(beta),
                          "OR_ci_low": np.exp(beta - 1.96 * se),
                          "OR_ci_high": np.exp(beta + 1.96 * se)}, index=names)
    return dict(table=table, beta=beta, prob=mu, ll=ll, ll0=ll0, lr=lr, lr_p=stats.chi2.sf(lr, k - 1),
                mcfadden=1 - ll / ll0, aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll, n=n, k=k)


def predict_logit(beta, X):
    return 1 / (1 + np.exp(-(np.asarray(X, float) @ beta)))


def hosmer_lemeshow(y, p, g=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), g, labels=False)
    grp = d.groupby("bin").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    chi = (((grp.obs - grp.exp) ** 2) / (grp.exp * (1 - grp.exp / grp.n))).sum()
    return chi, stats.chi2.sf(chi, g - 2)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1, running)
    return adj


def vif(X, names):
    out = {}
    for j, nme in enumerate(names):
        others = np.delete(X, j, axis=1)
        others = np.column_stack([np.ones(len(X)), others])
        r2 = ols(others, X[:, j], ["c"] + [n for i, n in enumerate(names) if i != j])["r2"]
        out[nme] = 1 / (1 - r2)
    return pd.Series(out, name="VIF")


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .1 else ""


# ----------------------------------------------------------------------------
# 0. Load data and score songs with the saved 5-topic LDA
# ----------------------------------------------------------------------------
print("=" * 78, "\n0. DATA PREPARATION\n" + "=" * 78)
df = pd.read_csv(DATA_DIR / "songs_modeling_ready.csv")
pkg = joblib.load(DATA_DIR / "LDA_5Topics_Model.joblib")
lda, vec, short_labels = pkg["model"], pkg["vectorizer"], pkg["topic_labels"]
TOPICS = [f"Topic_{i}" for i in range(1, 6)]

# same sample as the LDA notebook: reliable English lyrics
d = df[(df["Language"] == "en") & (df["Language Reliable"] == True)].copy().reset_index(drop=True)
theta = lda.transform(vec.transform(d["Clean Lyrics"].fillna("")))
d[TOPICS] = theta
print(f"Songs in LDA sample: {len(d)}  (topic prob. rows sum to 1: {np.allclose(theta.sum(1), 1)})")
print("Mean topic share:", dict(zip(short_labels, theta.mean(0).round(3))))

# ----------------------------------------------------------------------------
# Sentiment proxy: transparent lexicon score (no external resource available offline)
# ----------------------------------------------------------------------------
POS = set("""love loved loves loving lovely beautiful beauty sweet sweetest happy happiness joy joyful smile smiling smiles
laugh laughing laughter kiss kisses kissing hug hold holding darling dear honey angel heaven paradise dream dreams dreaming
sunshine sunny bright shine shining shines glow golden gold wonderful wonder amazing amazed great good best better fine
nice fun party dance dancing dances celebrate celebration free freedom alive life live lively hope hopeful hoping
trust faith believe believing blessed bless blessing grace gentle gently kind kindness tender tenderness warm warmth
safe safety peace peaceful calm easy comfort comforting pleasure pleased glad delight delightful cheer cheerful
treasure precious perfect perfectly true truly honest forever together heart hearts romance romantic charm charming
pretty handsome cute lucky luck win winner winning victory strong strength brave courage proud pride thank thanks
thankful grateful welcome care caring cared adore adored admire desire passion passionate magic magical miracle
beloved devotion devoted enjoy enjoyed enjoying excited exciting excite thrill thrilled wow yes sure kissed""".split())

NEG = set("""hate hated hates hating pain painful hurt hurts hurting cry crying cried cries tears tear sad sadness sorrow
sorry lonely alone loneliness lonesome miss missing lost lose losing loss leave leaving left gone goodbye die dying died
dead death kill killed killing blood grave bury buried cold dark darkness night shadow shadows fear afraid scared
scary terror nightmare hell devil evil wicked sin cruel mean angry anger rage mad crazy insane break broke broken
breaking breaks shatter shattered fall falling fell fail failed failure fool foolish stupid ugly bad worst worse
trouble troubles problem problems wrong lie lies lying liar cheat cheating cheated betray betrayed betrayal
bitter suffer suffering suffered misery miserable heartache heartbreak heartbroken weep weeping sick sickness
tired weary bored boring empty emptiness hollow useless worthless nothing never hopeless despair desperate
trap trapped chains prison jail guilty shame ashamed regret fight fighting war gun guns shoot shot knife
bleed bleeding scar scars wound wounded burn burning fire hurt abuse abused enemy hater haters hurtin
shit fuck fucking fucked bitch bitches nigga niggas damn ass dope drug drugs high drunk
cruel curse cursed doom doomed poor broke crash crashed danger dangerous threat trouble""".split())
NEGATORS = {"not", "no", "never", "dont", "cant", "wont", "aint", "didnt", "isnt", "wasnt", "couldnt",
            "wouldnt", "doesnt", "without", "nobody"}


def sentiment_counts(text):
    toks = re.findall(r"[a-z']+", str(text).lower())
    toks = [t.replace("'", "") for t in toks]
    pos = neg = 0
    for i, t in enumerate(toks):
        sign = 0
        if t in POS:
            sign = 1
        elif t in NEG:
            sign = -1
        if sign and i > 0 and toks[i - 1] in NEGATORS:      # one-word negation window
            sign = -sign
        pos += sign == 1
        neg += sign == -1
    return pos, neg


cnt = d["Clean Lyrics"].apply(sentiment_counts)
d["pos_words"], d["neg_words"] = cnt.str[0], cnt.str[1]
d["polar_words"] = d.pos_words + d.neg_words
d["sentiment_net_per100"] = 100 * (d.pos_words - d.neg_words) / d["Word Count New"]
d["sentiment_ratio"] = np.where(d.polar_words > 0, (d.pos_words - d.neg_words) / d.polar_words.replace(0, np.nan), np.nan)
d["positive"] = np.select([d.pos_words > d.neg_words, d.neg_words > d.pos_words], [1, 0], default=-1)
print(f"Lexicon sizes: {len(POS)} positive / {len(NEG)} negative words")
print("Sentiment label counts (1=positive, 0=negative, -1=tie/no polar word, excluded from logit):")
print(d["positive"].value_counts().sort_index().to_string())

d["Decade10"] = (d["Year"] - 1990) / 10          # per-decade scale, centred at 1990
d["log_wc"] = np.log(d["Word Count New"])
d["Song_ID"] = d["Year"].astype(str) + "_" + d["Rank"].astype(str).str.zfill(3)
d[["Song_ID", "Year", "Rank", "Song Title", "Artist", "Word Count New", "Lexical Diversity"] + TOPICS +
  ["pos_words", "neg_words", "sentiment_net_per100", "positive"]].to_csv(
    RES_DIR / "song_level_features.csv", index=False, encoding="utf-8-sig")

# ============================================================================
# 1. TOPIC EVOLUTION  -  OLS: mean topic probability (by year) ~ Year
# ============================================================================
print("\n" + "=" * 78, "\n1. TOPIC EVOLUTION (OLS, yearly means, n = number of years)\n" + "=" * 78)
yearly = d.groupby("Year")[TOPICS].mean()
years = yearly.index.values.astype(float)
Xy = np.column_stack([np.ones(len(years)), (years - 1959) / 10])          # slope per decade
Xq = np.column_stack([Xy, ((years - 1959) / 10) ** 2])
rows, fits = [], {}
for t, lab in zip(TOPICS, short_labels):
    y = yearly[t].values
    m = ols(Xy, y, ["Intercept", "Slope_per_decade"])
    mh = ols(Xy, y, ["Intercept", "Slope_per_decade"], cov="HAC", lag=3)
    mq = ols(Xq, y, ["Intercept", "Decade", "Decade^2"])
    f_inc = ((m["sse"] - mq["sse"]) / 1) / (mq["sse"] / (mq["n"] - 3))             # quadratic vs linear
    p_inc = stats.f.sf(f_inc, 1, mq["n"] - 3)
    sh = stats.shapiro(m["resid"])
    s = m["table"].loc["Slope_per_decade"]
    rows.append({"Topic": t, "Label": lab, "Mean_share": y.mean(),
                 "Slope_per_decade": s.coef, "SE": s.std_err, "p_OLS": s.p_value,
                 "p_NeweyWest_HAC": mh["table"].loc["Slope_per_decade", "p_value"],
                 "CI95_low": s.ci_low, "CI95_high": s.ci_high,
                 "R2": m["r2"], "Adj_R2": m["adj_r2"], "F": m["F"], "RMSE": m["rmse"],
                 "Durbin_Watson": m["dw"], "Shapiro_p_resid": sh.pvalue,
                 "Quadratic_gain_F_p": p_inc, "R2_quadratic": mq["r2"],
                 "Change_1959_to_2023": s.coef * (2023 - 1959) / 10})
    fits[t] = (m, mq)
topic_res = pd.DataFrame(rows)
topic_res["p_HAC_Holm"] = holm(topic_res["p_NeweyWest_HAC"].values)
topic_res["Sig_Holm"] = topic_res["p_HAC_Holm"].apply(stars)
topic_res.to_csv(RES_DIR / "topic_evolution_ols.csv", index=False, encoding="utf-8-sig")
print(topic_res[["Topic", "Label", "Mean_share", "Slope_per_decade", "p_OLS", "p_NeweyWest_HAC", "p_HAC_Holm",
                 "R2", "Adj_R2", "RMSE", "Durbin_Watson", "Quadratic_gain_F_p", "R2_quadratic"]]
      .round(4).to_string(index=False))
print("Sum of the 5 slopes (should be ~0 because shares sum to 1):", round(topic_res.Slope_per_decade.sum(), 6))

# Figure 1: fits
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
colors = plt.cm.tab10(np.arange(5))
xg = np.linspace(years.min(), years.max(), 200)
for ax, t, lab, c in zip(axes, TOPICS, short_labels, colors):
    m, mq = fits[t]
    ax.scatter(years, yearly[t], s=18, color=c, alpha=.75, label="Yearly mean")
    Xg = np.column_stack([np.ones_like(xg), (xg - 1959) / 10])
    pred = Xg @ m["beta"]
    s2 = m["sse"] / (m["n"] - 2)
    covb = np.linalg.inv(Xy.T @ Xy) * s2
    se_pred = np.sqrt(np.einsum("ij,jk,ik->i", Xg, covb, Xg))
    tc = stats.t.ppf(.975, m["n"] - 2)
    ax.plot(xg, pred, color="black", lw=2, label="Linear fit")
    ax.fill_between(xg, pred - tc * se_pred, pred + tc * se_pred, color="gray", alpha=.25, label="95% CI of mean")
    Xqg = np.column_stack([Xg, ((xg - 1959) / 10) ** 2])
    ax.plot(xg, Xqg @ mq["beta"], "--", color="crimson", lw=1.2, label="Quadratic (check)")
    r = topic_res[topic_res.Topic == t].iloc[0]
    ax.set_title(f"{t}: {lab}\nslope = {r.Slope_per_decade:+.4f}/decade, R² = {r.R2:.2f}, p(HAC,Holm) = {r.p_HAC_Holm:.3g}",
                 fontsize=9.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean topic probability")
axes[0].legend(fontsize=8)
axes[5].axis("off")
axes[5].text(0, .5, "Linear OLS per topic on yearly means\nSlope expressed per decade.\n"
             "HAC = Newey-West (lag 3) p-values;\nHolm correction across 5 topics.\n"
             "Dashed red = quadratic check.", fontsize=10, va="center")
plt.suptitle("Topic Evolution: Linear Regression of Mean Topic Probability on Year (1959-2023)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, .96])
plt.savefig(FIG_DIR / "REG_01_Topic_Evolution_Fits.png", dpi=250)
plt.close()

# Figure 2: slopes
fig, ax = plt.subplots(figsize=(9, 5))
o = topic_res.sort_values("Slope_per_decade")
ax.barh(o.Label, o.Slope_per_decade, xerr=[o.Slope_per_decade - o.CI95_low, o.CI95_high - o.Slope_per_decade],
        color=["#d62728" if v < 0 else "#2ca02c" for v in o.Slope_per_decade], alpha=.85, capsize=4)
for i, (_, r) in enumerate(o.iterrows()):
    ax.text(r.CI95_high + .0005 if r.Slope_per_decade > 0 else r.CI95_low - .0005, i, stars(r.p_HAC_Holm) or "n.s.",
            va="center", ha="left" if r.Slope_per_decade > 0 else "right", fontsize=11)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Change in mean topic probability per decade (95% CI)")
ax.set_title("Topic trends 1959-2023 (stars: Holm-adjusted Newey-West p)")
plt.tight_layout()
plt.savefig(FIG_DIR / "REG_02_Topic_Slopes.png", dpi=250)
plt.close()

# ============================================================================
# 2. SENTIMENT EVOLUTION  -  Logistic regression
# ============================================================================
print("\n" + "=" * 78, "\n2. SENTIMENT EVOLUTION (Logistic Regression: positive=1 vs negative=0)\n" + "=" * 78)
s = d[d.positive >= 0].copy().reset_index(drop=True)
print(f"Songs used: {len(s)}  | positive share = {s.positive.mean():.3f}")
ref = TOPICS[int(np.argmax(theta.mean(0)))]
topic_x = [t for t in TOPICS if t != ref]
print(f"Reference topic dropped to avoid perfect collinearity: {ref} ({short_labels[TOPICS.index(ref)]})")

# 2a. pure time trend
XA = np.column_stack([np.ones(len(s)), s["Decade10"]])
mA = logit(XA, s["positive"], ["Intercept", "Year (per decade)"])
print("\n[Model A] positive ~ Year")
print(mA["table"].round(4).to_string())
print(f"McFadden R2 = {mA['mcfadden']:.4f}, LR chi2 = {mA['lr']:.2f}, p = {mA['lr_p']:.3g}")

# 2b. full model
zs = lambda v: (v - v.mean()) / v.std()
s["log_wc_z"], s["lexdiv_z"] = zs(s["log_wc"]), zs(s["Lexical Diversity"])
for t in topic_x:
    s[t + "_x10"] = s[t] * 10
feat_names = ["Decade10"] + [t + "_x10" for t in topic_x] + ["log_wc_z", "lexdiv_z"]
nice = ["Year (per decade)"] + [f"{t} {short_labels[TOPICS.index(t)]} (+10pp)" for t in topic_x] + \
       ["Log word count (+1 SD)", "Lexical diversity (+1 SD)"]
XB = np.column_stack([np.ones(len(s)), s[feat_names].values])
yb = s["positive"].values
idx_tr, idx_te = train_test_split(np.arange(len(s)), test_size=.2, stratify=yb, random_state=SEED)
mB_tr = logit(XB[idx_tr], yb[idx_tr], ["Intercept"] + nice)
mB_all = logit(XB, yb, ["Intercept"] + nice)
print("\n[Model B] positive ~ Year + topics + length + lexical diversity (full sample, for inference)")
print(mB_all["table"].round(4).to_string())
print(f"McFadden R2 = {mB_all['mcfadden']:.4f}, LR chi2 = {mB_all['lr']:.2f}, p = {mB_all['lr_p']:.3g}, "
      f"AIC = {mB_all['aic']:.1f}, BIC = {mB_all['bic']:.1f}")
print(f"Likelihood-ratio test B vs A: chi2 = {2 * (mB_all['ll'] - mA['ll']):.2f}, "
      f"df = {mB_all['k'] - mA['k']}, p = {stats.chi2.sf(2 * (mB_all['ll'] - mA['ll']), mB_all['k'] - mA['k']):.3g}")
vifB = vif(XB[:, 1:], nice)
print("\nVIF:\n", vifB.round(2).to_string())

# evaluation
def clf_metrics(y, p, thr=.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"AUC": roc_auc_score(y, p), "Accuracy": accuracy_score(y, yhat),
            "Balanced_accuracy": balanced_accuracy_score(y, yhat), "Precision": precision_score(y, yhat),
            "Recall": recall_score(y, yhat), "F1": f1_score(y, yhat), "Brier": brier_score_loss(y, p),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


p_te = predict_logit(mB_tr["beta"], XB[idx_te])
p_tr = predict_logit(mB_tr["beta"], XB[idx_tr])
met = pd.DataFrame({"Train (80%)": clf_metrics(yb[idx_tr], p_tr), "Test (20%)": clf_metrics(yb[idx_te], p_te)}).T
maj = max(yb[idx_te].mean(), 1 - yb[idx_te].mean())
met["Majority_baseline_acc"] = maj
# 5-fold CV
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
cvp = cross_val_predict(LogisticRegression(C=1e9, max_iter=2000), XB[:, 1:], yb, cv=skf, method="predict_proba")[:, 1]
met.loc["5-fold CV (all)"] = {**clf_metrics(yb, cvp), "Majority_baseline_acc": max(yb.mean(), 1 - yb.mean())}
# time-based hold-out: train on <=2009, test on >=2010
tr_t, te_t = (s.Year <= 2009).values, (s.Year >= 2010).values
m_t = logit(XB[tr_t], yb[tr_t], ["Intercept"] + nice)
met.loc["Time split (train<=2009, test>=2010)"] = {**clf_metrics(yb[te_t], predict_logit(m_t["beta"], XB[te_t])),
                                                  "Majority_baseline_acc": max(yb[te_t].mean(), 1 - yb[te_t].mean())}
hl_chi, hl_p = hosmer_lemeshow(yb, mB_all["prob"])
print("\nEvaluation:\n", met.round(4).to_string())
print(f"Hosmer-Lemeshow (full sample): chi2 = {hl_chi:.2f}, p = {hl_p:.3f}")
met["Hosmer_Lemeshow_p(full)"] = hl_p
met["McFadden_R2(full)"] = mB_all["mcfadden"]
met.to_csv(RES_DIR / "sentiment_logit_metrics.csv", encoding="utf-8-sig")
tabB = mB_all["table"].copy()
tabB["sig"] = tabB.p_value.apply(stars)
tabB.to_csv(RES_DIR / "sentiment_logit_coefficients.csv", encoding="utf-8-sig")
mA["table"].to_csv(RES_DIR / "sentiment_logit_time_only.csv", encoding="utf-8-sig")

# Figure 3: sentiment evolution
yr = s.groupby("Year").agg(share=("positive", "mean"), n=("positive", "size"))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.scatter(yr.index, yr.share, s=yr.n / 2.2, color="#1f77b4", alpha=.7, label="Yearly share of positive songs")
xg = np.linspace(1959, 2023, 200)
ax.plot(xg, predict_logit(mA["beta"], np.column_stack([np.ones_like(xg), (xg - 1990) / 10])), color="black", lw=2.2,
        label="Logistic fit (Model A)")
rA = mA["table"].loc["Year (per decade)"]
ax.set_title(f"Sentiment Evolution: P(positive) vs Year   |   OR per decade = {rA.odds_ratio:.3f} "
             f"(p = {rA.p_value:.3g}), McFadden R² = {mA['mcfadden']:.3f}")
ax.set_xlabel("Year")
ax.set_ylabel("Share of lyrics with net-positive sentiment")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "REG_03_Sentiment_Evolution.png", dpi=250)
plt.close()

# Figure 4: ROC + calibration + confusion
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for name, yy, pp in [("Test (20%)", yb[idx_te], p_te), ("5-fold CV", yb, cvp),
                     ("Time split (>=2010)", yb[te_t], predict_logit(m_t["beta"], XB[te_t]))]:
    fpr, tpr, _ = roc_curve(yy, pp)
    axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(yy, pp):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
axes[0].legend(loc="lower right", fontsize=9)
cal = pd.DataFrame({"y": yb, "p": mB_all["prob"]})
cal["bin"] = pd.qcut(cal.p.rank(method="first"), 10, labels=False)
cg = cal.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"))
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].plot(cg.p, cg.y, "o-", color="#d62728")
axes[1].set(xlabel="Mean predicted probability (decile)", ylabel="Observed positive share",
            title=f"Calibration (Hosmer-Lemeshow p = {hl_p:.2f})")
cm = confusion_matrix(yb[idx_te], (p_te >= .5).astype(int))
axes[2].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred neg", "Pred pos"], yticklabels=["True neg", "True pos"],
            title="Confusion matrix (test set, cutoff 0.5)")
axes[2].grid(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "REG_04_Sentiment_Model_Evaluation.png", dpi=250)
plt.close()

# Figure 5: odds ratios
tb = mB_all["table"].drop("Intercept").iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.errorbar(tb.odds_ratio, range(len(tb)), xerr=[tb.odds_ratio - tb.OR_ci_low, tb.OR_ci_high - tb.odds_ratio],
            fmt="o", color="#1f77b4", capsize=4)
ax.axvline(1, color="k", ls="--")
ax.set_yticks(range(len(tb)))
ax.set_yticklabels([f"{n} {stars(p)}" for n, p in zip(tb.index, tb.p_value)], fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Odds ratio for positive sentiment (log scale, 95% CI)")
ax.set_title(f"Model B odds ratios (reference topic: {ref} {short_labels[TOPICS.index(ref)]})")
plt.tight_layout()
plt.savefig(FIG_DIR / "REG_05_Sentiment_Odds_Ratios.png", dpi=250)
plt.close()

# ============================================================================
# 3. BILLBOARD RANK REGRESSION  -  OLS
# ============================================================================
print("\n" + "=" * 78, "\n3. BILLBOARD RANK REGRESSION (OLS; lower Rank number = better chart position)\n" + "=" * 78)
r = d.copy()
r["sent_z"], r["log_wc_z"], r["lexdiv_z"] = zs(r["sentiment_net_per100"]), zs(r["log_wc"]), zs(r["Lexical Diversity"])
for t in topic_x:
    r[t + "_x10"] = r[t] * 10
rf = ["Decade10"] + [t + "_x10" for t in topic_x] + ["sent_z", "log_wc_z", "lexdiv_z"]
rn = ["Year (per decade)"] + [f"{t} {short_labels[TOPICS.index(t)]} (+10pp)" for t in topic_x] + \
     ["Sentiment net/100 words (+1 SD)", "Log word count (+1 SD)", "Lexical diversity (+1 SD)"]
XR = np.column_stack([np.ones(len(r)), r[rf].values])
yR = r["Rank"].values.astype(float)
mR = ols(XR, yR, ["Intercept"] + rn)
mR3 = ols(XR, yR, ["Intercept"] + rn, cov="HC3")
tab = mR["table"].copy()
tab["p_HC3"] = mR3["table"]["p_value"]
tab["sig_HC3"] = tab.p_HC3.apply(stars)
tab["std_beta"] = np.r_[np.nan, [tab.coef.iloc[i + 1] * XR[:, i + 1].std() / yR.std() for i in range(len(rn))]]
print(f"n = {mR['n']}, Rank mean = {yR.mean():.2f}, SD = {yR.std():.2f}")
print(tab.round(4).to_string())
print(f"\nR2 = {mR['r2']:.5f}, Adj R2 = {mR['adj_r2']:.5f}, F({mR['k'] - 1},{mR['n'] - mR['k']}) = {mR['F']:.3f}, "
      f"p = {mR['F_p']:.3g}, RMSE = {mR['rmse']:.3f}, AIC = {mR['aic']:.1f}")

# diagnostics
bp_X = np.column_stack([np.ones(len(r)), XR[:, 1:]])
aux = ols(bp_X, mR["resid"] ** 2, ["c"] + rn)
bp_lm = mR["n"] * aux["r2"]
bp_p = stats.chi2.sf(bp_lm, mR["k"] - 1)
jb = stats.jarque_bera(mR["resid"])
vifR = vif(XR[:, 1:], rn)
print(f"Breusch-Pagan LM = {bp_lm:.2f}, p = {bp_p:.3g} | Jarque-Bera p = {jb.pvalue:.3g}")
print("VIF:\n", vifR.round(2).to_string())

# cross-validation + baselines
kf = KFold(10, shuffle=True, random_state=SEED)
cvr = cross_val_predict(LinearRegression(), XR[:, 1:], yR, cv=kf)
cv_r2 = 1 - ((yR - cvr) ** 2).sum() / ((yR - yR.mean()) ** 2).sum()
cv_rmse = np.sqrt(np.mean((yR - cvr) ** 2))
base_rmse = yR.std()
# grouped by year (leave future years out): train<=2009, test>=2010
trm, tem = (r.Year <= 2009).values, (r.Year >= 2010).values
m_time = ols(XR[trm], yR[trm], ["Intercept"] + rn)
pred_time = XR[tem] @ m_time["beta"]
time_r2 = 1 - ((yR[tem] - pred_time) ** 2).sum() / ((yR[tem] - yR[trm].mean()) ** 2).sum()
time_rmse = np.sqrt(np.mean((yR[tem] - pred_time) ** 2))
# nested F-test: do lyric features add anything beyond Year?
m0 = ols(XR[:, :2], yR, ["Intercept", "Year"])
F_nest = ((m0["sse"] - mR["sse"]) / (mR["k"] - 2)) / (mR["sse"] / (mR["n"] - mR["k"]))
p_nest = stats.f.sf(F_nest, mR["k"] - 2, mR["n"] - mR["k"])
print(f"10-fold CV: R2 = {cv_r2:.5f}, RMSE = {cv_rmse:.3f} (baseline SD {base_rmse:.3f})")
print(f"Time hold-out (>=2010): R2 = {time_r2:.5f}, RMSE = {time_rmse:.3f}")
print(f"Nested F-test (lyric features vs Year-only): F = {F_nest:.3f}, p = {p_nest:.3g}")
pd.DataFrame({"Metric": ["n", "R2", "Adj_R2", "F", "F_p", "RMSE_in_sample", "Baseline_RMSE(SD of Rank)", "AIC", "BIC",
                         "CV10_R2", "CV10_RMSE", "TimeSplit_R2(>=2010)", "TimeSplit_RMSE", "Nested_F_vs_YearOnly",
                         "Nested_F_p", "Breusch_Pagan_p", "Jarque_Bera_p", "Max_VIF"],
              "Value": [mR["n"], mR["r2"], mR["adj_r2"], mR["F"], mR["F_p"], mR["rmse"], base_rmse, mR["aic"], mR["bic"],
                        cv_r2, cv_rmse, time_r2, time_rmse, F_nest, p_nest, bp_p, jb.pvalue, vifR.max()]}
             ).to_csv(RES_DIR / "rank_ols_metrics.csv", index=False, encoding="utf-8-sig")
tab.to_csv(RES_DIR / "rank_ols_coefficients.csv", encoding="utf-8-sig")
vifR.to_csv(RES_DIR / "rank_ols_vif.csv", encoding="utf-8-sig")

# Figure 6: coefficient plot (standardised)
tt = tab.drop("Intercept").iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.errorbar(tt.coef, range(len(tt)), xerr=[tt.coef - tt.ci_low, tt.ci_high - tt.coef], fmt="o", color="#9467bd", capsize=4)
ax.axvline(0, color="k", ls="--")
ax.set_yticks(range(len(tt)))
ax.set_yticklabels([f"{n} {stars(p)}" for n, p in zip(tt.index, tt.p_HC3)], fontsize=9)
ax.set_xlabel("Change in Billboard Rank position (positive = lower on chart); 95% CI, stars from HC3 p-values")
ax.set_title(f"Rank regression coefficients (R² = {mR['r2']:.4f}, CV R² = {cv_r2:.4f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "REG_06_Rank_Coefficients.png", dpi=250)
plt.close()

# Figure 7: diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0, 0].scatter(mR["fitted"], mR["resid"], s=5, alpha=.35)
axes[0, 0].axhline(0, color="r")
axes[0, 0].set(xlabel="Fitted Rank", ylabel="Residual", title=f"Residuals vs fitted (Breusch-Pagan p = {bp_p:.3f})")
stats.probplot(mR["resid"], plot=axes[0, 1])
axes[0, 1].set_title("Normal Q-Q of residuals")
axes[1, 0].scatter(mR["fitted"], yR, s=5, alpha=.35)
axes[1, 0].plot([yR.min(), yR.max()], [yR.min(), yR.max()], "r--")
axes[1, 0].set(xlabel="Predicted Rank", ylabel="Actual Rank", title="Predicted vs actual (red = perfect)")
axes[1, 1].hist(yR - cvr, bins=40, color="#9467bd", alpha=.8)
axes[1, 1].set(xlabel="10-fold CV residual", title=f"CV residuals (RMSE = {cv_rmse:.2f} vs baseline {base_rmse:.2f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "REG_07_Rank_Diagnostics.png", dpi=250)
plt.close()

# Figure 8: mean rank by topic dominance, descriptive support
r["dominant"] = r[TOPICS].idxmax(axis=1).map(dict(zip(TOPICS, short_labels)))
g = r.groupby("dominant")["Rank"].agg(["mean", "sem", "size"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(g.index, g["mean"], xerr=1.96 * g["sem"], color="#8c564b", alpha=.8, capsize=4)
ax.axvline(yR.mean(), color="k", ls="--", label=f"Overall mean {yR.mean():.1f}")
ax.set_xlabel("Mean Billboard Rank (95% CI)")
ax.set_title("Mean rank by dominant LDA topic (descriptive)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "REG_08_Rank_by_Dominant_Topic.png", dpi=250)
plt.close()
g.to_csv(RES_DIR / "rank_by_dominant_topic.csv", encoding="utf-8-sig")
print("\nMean rank by dominant topic:\n", g.round(3).to_string())
print("\nDONE. Figures:", sorted(p.name for p in FIG_DIR.glob("*.png")))
