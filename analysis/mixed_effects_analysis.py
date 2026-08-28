"""Mixed-effects analysis of the LiveNewsBench full Sonnet-v2 matrix.

Question: what matters most, and what drives changes in evidence, answers, and quality?

Design facts that shape every model here:
  * 1,129 unique task keys / 1,329 dataset rows x 14 model-arm conditions, one
    generation per cell. Rows are paired across conditions, so the question is a
    crossed random factor and inference must cluster on task_key.
  * The Native arm exists only for Claude and GPT, so model x arm is not a full
    factorial. The design matrix is built cell-by-cell and empty interaction
    cells are omitted, which keeps the model full rank (14 cells = 1 + 3 model
    + 3 arm + 7 estimable interactions).
  * Retrieved-evidence metrics exist only in the normalized-harness arms
    (Normalized, Wide). Evidence models are therefore a clean 4 x 2 factorial.
  * Mediators (evidence quality, search count, answer length) are
    post-treatment. They live in a separate, explicitly descriptive block.

Primary estimator is a random-intercept linear mixed model. On 0/1 outcomes that
is a linear probability model with partial pooling, so every coefficient reads
directly in percentage points; GEE logistic with an exchangeable working
correlation is reported as a robustness check on sign and significance.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent
ROWS = ROOT / "variable_row_level.csv"
OUT = ROOT
GROUP = "task_key"

MODEL_REF = "GPT-5.6 Terra"
ARM_REF = "Normalized"

QUALITY = ["judge", "gated"]
EVIDENCE = [
    "snippet_sufficiency",
    "evidence_precision",
    "token_discounted_gain",
    "temporal_grounding",
    "domain_entropy",
    "compression_redundancy",
]
BEHAVIOR = ["used_searches", "log_latency", "log_answer_words"]

MEDIATORS = [
    "snippet_sufficiency",
    "evidence_precision",
    "temporal_grounding",
    "domain_entropy",
    "compression_redundancy",
    "used_searches",
    "log_answer_words",
]

PRETTY = {
    "judge": "semantic judge pass",
    "gated": "gated answer match",
    "snippet_sufficiency": "literal gold visible in snippets",
    "evidence_precision": "evidence precision",
    "token_discounted_gain": "token-discounted gain",
    "temporal_grounding": "temporal grounding",
    "domain_entropy": "source diversity",
    "compression_redundancy": "distinctness / low redundancy",
    "used_searches": "searches issued",
    "log_latency": "log latency (s)",
    "log_answer_words": "log answer length (words)",
}


def slug(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_")


# --------------------------------------------------------------------------- #
# data + explicit design construction
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    d = pd.read_csv(ROWS)
    d["log_latency"] = np.log(d["latency_s"].clip(lower=1e-3))
    d["log_answer_words"] = np.log(d["answer_words"].clip(lower=1))
    d["age30"] = (d["event_age_days"] - d["event_age_days"].mean()) / 30.0
    d["quant"] = (d["question_type"] == "Quantitative / composed").astype(float)
    d["retrieval"] = (d["arm"] != "No search").astype(float)
    d["wide"] = (d["arm"] == "Wide").astype(float)
    counts = d["category"].value_counts()
    small = set(counts[counts < 500].index)
    d["cat"] = np.where(d["category"].isin(small), "Other / small categories", d["category"])
    return d


def build_design(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add explicit dummy columns and return the block -> column-name map.

    Interaction columns are created only for model-arm cells that were actually
    run, so the resulting design is full rank even though Native is missing for
    two models.
    """
    d = df.copy()
    blocks: dict[str, list[str]] = {"task_context": [], "model_identity": [],
                                    "retrieval_arm": [], "model_x_arm": []}

    cats = sorted(d["cat"].unique())
    cat_ref = d["cat"].value_counts().idxmax()
    for c in cats:
        if c == cat_ref:
            continue
        col = f"cat_{slug(c)}"
        d[col] = (d["cat"] == c).astype(float)
        blocks["task_context"].append(col)
    blocks["task_context"] += ["quant", "age30"]

    models = [m for m in sorted(d["model"].unique()) if m != MODEL_REF]
    for m in models:
        col = f"mdl_{slug(m)}"
        d[col] = (d["model"] == m).astype(float)
        blocks["model_identity"].append(col)

    arms = [a for a in ["No search", "Wide", "Native"] if a in set(d["arm"]) and a != ARM_REF]
    for a in arms:
        col = f"arm_{slug(a)}"
        d[col] = (d["arm"] == a).astype(float)
        blocks["retrieval_arm"].append(col)

    for m in models:
        for a in arms:
            mask = (d["model"] == m) & (d["arm"] == a)
            if mask.sum() == 0:
                continue  # cell was never run
            col = f"ix_{slug(m)}__{slug(a)}"
            d[col] = mask.astype(float)
            blocks["model_x_arm"].append(col)

    return d, {k: v for k, v in blocks.items() if v}


def rhs_of(cols: list[str]) -> str:
    return " + ".join(cols) if cols else "1"


# --------------------------------------------------------------------------- #
# mixed model helpers
# --------------------------------------------------------------------------- #
def fit_lmm(data: pd.DataFrame, y: str, cols: list[str], re_formula: str | None = None):
    sub = data.dropna(subset=[y] + [c for c in cols if c in data]).copy()
    md = smf.mixedlm(f"{y} ~ {rhs_of(cols)}", sub, groups=sub[GROUP], re_formula=re_formula)
    try:
        res = md.fit(reml=True, method=["lbfgs", "bfgs"])
    except Exception:
        res = md.fit(reml=True, method="cg")
    return res, sub


def r2_parts(res) -> dict[str, float]:
    cov = np.asarray(res.cov_re)
    tau = float(cov[0, 0]) if cov.size else 0.0
    sigma = float(res.scale)
    fixed = np.asarray(res.model.exog) @ np.asarray(res.fe_params)
    vf = float(np.var(fixed, ddof=0))
    denom = vf + tau + sigma
    return {
        "var_fixed": vf,
        "var_task": tau,
        "var_resid": sigma,
        "r2_marginal": vf / denom if denom else np.nan,
        "r2_conditional": (vf + tau) / denom if denom else np.nan,
        "icc_task": tau / (tau + sigma) if (tau + sigma) else np.nan,
    }


def wald(res, names: list[str]) -> tuple[float, int, float]:
    idx = [i for i, n in enumerate(res.model.exog_names) if n in names]
    if not idx:
        return np.nan, 0, np.nan
    b = np.asarray(res.fe_params)[idx]
    k = len(res.model.exog_names)
    V = np.asarray(res.cov_params())[:k, :k][np.ix_(idx, idx)]
    try:
        stat = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        stat = float(b @ np.linalg.pinv(V) @ b)
    return stat, len(idx), float(st.chi2.sf(stat, len(idx)))


def tidy(res, sub: pd.DataFrame, y: str, spec: str) -> pd.DataFrame:
    ci = res.conf_int()
    sd = float(sub[y].std(ddof=1))
    recs = []
    for n in res.model.exog_names:
        recs.append({
            "outcome": y, "spec": spec, "term": n,
            "estimate": float(res.fe_params[n]), "se": float(res.bse[n]),
            "ci_low": float(ci.loc[n, 0]), "ci_high": float(ci.loc[n, 1]),
            "z": float(res.tvalues[n]), "p": float(res.pvalues[n]),
            "std_effect": float(res.fe_params[n]) / sd if sd else np.nan,
            "n_obs": int(len(sub)), "n_groups": int(sub[GROUP].nunique()),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# importance: sequential and drop-one variance decomposition
# --------------------------------------------------------------------------- #
def importance(data: pd.DataFrame, y: str, blocks: dict[str, list[str]], order: list[str]):
    all_cols = [c for b in order for c in blocks.get(b, [])]
    full, sub = fit_lmm(data, y, all_cols)
    rf = r2_parts(full)
    empty, _ = fit_lmm(data, y, [])
    re0 = r2_parts(empty)

    recs = [{
        "outcome": y, "block": "__full_model__",
        "r2_marginal": rf["r2_marginal"], "r2_conditional": rf["r2_conditional"],
        "icc_task_empty": re0["icc_task"], "icc_task_full": rf["icc_task"],
        "var_task_empty": re0["var_task"], "var_resid_empty": re0["var_resid"],
        "delta_r2_sequential": np.nan, "delta_r2_unique": np.nan,
        "wald_chi2": np.nan, "wald_df": np.nan, "wald_p": np.nan,
        "n_obs": len(sub), "n_groups": sub[GROUP].nunique(),
    }]

    prev, so_far = 0.0, []
    for name in order:
        cols = blocks.get(name, [])
        if not cols:
            continue
        so_far += cols
        res_i, sub_i = fit_lmm(data, y, so_far)
        r2_i = r2_parts(res_i)["r2_marginal"]
        seq, prev = r2_i - prev, r2_i

        drop = [c for c in all_cols if c not in cols]
        res_d, _ = fit_lmm(data, y, drop)
        uniq = rf["r2_marginal"] - r2_parts(res_d)["r2_marginal"]

        chi2, df, p = wald(full, cols)
        recs.append({
            "outcome": y, "block": name,
            "r2_marginal": r2_i, "r2_conditional": np.nan,
            "icc_task_empty": np.nan, "icc_task_full": np.nan,
            "var_task_empty": np.nan, "var_resid_empty": np.nan,
            "delta_r2_sequential": seq, "delta_r2_unique": uniq,
            "wald_chi2": chi2, "wald_df": df, "wald_p": p,
            "n_obs": len(sub), "n_groups": sub[GROUP].nunique(),
        })
    return pd.DataFrame(recs), full, sub


# --------------------------------------------------------------------------- #
# named linear contrasts with delta-method CIs
# --------------------------------------------------------------------------- #
def contrast(res, weights: dict[str, float]) -> tuple[float, float, float, float, float]:
    names = list(res.model.exog_names)
    L = np.zeros(len(names))
    for k, w in weights.items():
        if k not in names:
            continue
        L[names.index(k)] += w
    k = len(names)
    V = np.asarray(res.cov_params())[:k, :k]  # drop variance-component rows
    est = float(L @ np.asarray(res.fe_params))
    var = float(L @ V @ L)
    se = float(np.sqrt(max(var, 0)))
    z = est / se if se else np.nan
    return est, se, est - 1.96 * se, est + 1.96 * se, float(2 * st.norm.sf(abs(z)))


def arm_contrasts(res, y: str, models: list[str], spec: str) -> pd.DataFrame:
    """Per-model arm contrasts, all expressed as 'arm minus Normalized'."""
    recs = []
    for m in models:
        ms = slug(m)
        for arm, label in [("No_search", "Normalized - No search (retrieval gain)"),
                           ("Wide", "Wide - Normalized (result tier)"),
                           ("Native", "Native - Normalized (implementation)")]:
            main, ix = f"arm_{arm}", f"ix_{ms}__{arm}"
            if main not in res.model.exog_names:
                continue
            if m != MODEL_REF and ix not in res.model.exog_names:
                continue  # cell not run
            sign = -1.0 if arm == "No_search" else 1.0
            w = {main: sign}
            if m != MODEL_REF:
                w[ix] = sign
            est, se, lo, hi, p = contrast(res, w)
            recs.append({"outcome": y, "spec": spec, "model": m, "contrast": label,
                         "estimate": est, "se": se, "ci_low": lo, "ci_high": hi, "p": p})
    # differences in retrieval gain between models
    if "arm_No_search" in res.model.exog_names:
        for a in models:
            for b in models:
                if a >= b:
                    continue
                w = {}
                if a != MODEL_REF:
                    w[f"ix_{slug(a)}__No_search"] = -1.0
                if b != MODEL_REF:
                    w[f"ix_{slug(b)}__No_search"] = 1.0
                if not w or any(k not in res.model.exog_names for k in w):
                    continue
                est, se, lo, hi, p = contrast(res, w)
                recs.append({"outcome": y, "spec": spec, "model": f"{a} - {b}",
                             "contrast": "difference in retrieval gain",
                             "estimate": est, "se": se, "ci_low": lo, "ci_high": hi, "p": p})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# mediation of the Wide-vs-Normalized tier effect
# --------------------------------------------------------------------------- #
def mediation(data: pd.DataFrame, y: str) -> pd.DataFrame:
    sub = data[data["arm"].isin(["Normalized", "Wide"])].copy()
    sub, blocks = build_design(sub)
    base = blocks["model_identity"] + blocks["task_context"]
    tier = blocks["retrieval_arm"]  # just arm_Wide here
    sub = sub.dropna(subset=[y] + MEDIATORS)
    recs = []

    total, s1 = fit_lmm(sub, y, tier + base)
    tot = float(total.fe_params[tier[0]])
    recs.append({"outcome": y, "path": "total_wide_effect", "mediator": "-",
                 "estimate": tot, "se": float(total.bse[tier[0]]),
                 "p": float(total.pvalues[tier[0]]), "n_obs": len(s1)})

    for m in MEDIATORS:
        a, sa = fit_lmm(sub, m, tier + base)
        recs.append({"outcome": m, "path": "a_wide_to_mediator", "mediator": m,
                     "estimate": float(a.fe_params[tier[0]]), "se": float(a.bse[tier[0]]),
                     "p": float(a.pvalues[tier[0]]), "n_obs": len(sa)})

    z = sub.copy()
    zc = []
    for m in MEDIATORS:
        col = m + "_z"
        z[col] = (z[m] - z[m].mean()) / z[m].std(ddof=1)
        zc.append(col)
    b, sb = fit_lmm(z, y, tier + base + zc)
    for m, col in zip(MEDIATORS, zc):
        recs.append({"outcome": y, "path": "b_mediator_to_outcome_per_SD", "mediator": m,
                     "estimate": float(b.fe_params[col]), "se": float(b.bse[col]),
                     "p": float(b.pvalues[col]), "n_obs": len(sb)})
    direct = float(b.fe_params[tier[0]])
    recs.append({"outcome": y, "path": "direct_wide_effect_adj_mediators", "mediator": "all",
                 "estimate": direct, "se": float(b.bse[tier[0]]),
                 "p": float(b.pvalues[tier[0]]), "n_obs": len(sb)})
    recs.append({"outcome": y, "path": "implied_indirect_share", "mediator": "all",
                 "estimate": (tot - direct) / tot if tot else np.nan,
                 "se": np.nan, "p": np.nan, "n_obs": len(sb)})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# moderation: does the retrieval gain depend on task properties?
# --------------------------------------------------------------------------- #
def moderation(data: pd.DataFrame, y: str) -> pd.DataFrame:
    """No-search vs Normalized only. `ret` is 1 when the harness is available."""
    sub = data[data["arm"].isin(["No search", "Normalized"])].copy()
    sub, blocks = build_design(sub)
    sub["ret"] = sub["retrieval"]
    sub["ret_x_age30"] = sub["ret"] * sub["age30"]
    sub["ret_x_quant"] = sub["ret"] * sub["quant"]
    cols = (["ret", "ret_x_age30", "ret_x_quant"]
            + blocks["model_identity"] + blocks["task_context"]
            + [c for c in blocks.get("model_x_arm", [])])
    # arm dummy is collinear with `ret`; drop the arm block and its interactions
    cols = [c for c in cols if not c.startswith("arm_")]
    cols = [c for c in cols if "__No_search" not in c] + [
        c.replace("__No_search", "__ret") for c in blocks.get("model_x_arm", [])
        if "__No_search" in c
    ]
    for m in sorted(sub["model"].unique()):
        if m == MODEL_REF:
            continue
        sub[f"ix_{slug(m)}__ret"] = ((sub["model"] == m) & (sub["ret"] == 1)).astype(float)
    res, s2 = fit_lmm(sub, y, cols)
    keep = ["ret", "ret_x_age30", "ret_x_quant"]
    ci = res.conf_int()
    return pd.DataFrame([{
        "outcome": y, "term": t, "estimate": float(res.fe_params[t]),
        "se": float(res.bse[t]), "ci_low": float(ci.loc[t, 0]),
        "ci_high": float(ci.loc[t, 1]), "p": float(res.pvalues[t]), "n_obs": len(s2),
    } for t in keep])


# --------------------------------------------------------------------------- #
# heterogeneity of the retrieval effect across questions
# --------------------------------------------------------------------------- #
def heterogeneity(data: pd.DataFrame, y: str) -> dict:
    sub = data[data["arm"].isin(["No search", "Normalized"])].copy()
    sub, blocks = build_design(sub)
    cols = blocks["retrieval_arm"] + blocks["model_identity"] + blocks["task_context"] \
        + blocks.get("model_x_arm", [])
    ri, s = fit_lmm(sub, y, cols)
    rs, _ = fit_lmm(sub, y, cols, re_formula="~retrieval")
    cov = np.asarray(rs.cov_re)
    arm_col = blocks["retrieval_arm"][0]  # arm_No_search
    lr = float(2 * (rs.llf - ri.llf))
    return {
        "outcome": y,
        "sd_task_intercept": float(np.sqrt(max(cov[0, 0], 0))) if cov.size else np.nan,
        "sd_task_retrieval_slope": float(np.sqrt(max(cov[-1, -1], 0))) if cov.shape[0] > 1 else np.nan,
        "no_search_effect_ref_model": float(ri.fe_params[arm_col]),
        "icc_task_random_intercept": r2_parts(ri)["icc_task"],
        "lr_chi2_slope_vs_intercept": lr,
        "lr_p": float(st.chi2.sf(max(lr, 0), 2)),
        "n_obs": len(s),
    }


# --------------------------------------------------------------------------- #
# GEE logistic robustness
# --------------------------------------------------------------------------- #
def gee_logit(data: pd.DataFrame, y: str, cols: list[str]) -> pd.DataFrame:
    sub = data.dropna(subset=[y]).copy()
    res = smf.gee(f"{y} ~ {rhs_of(cols)}", groups=GROUP, data=sub,
                  family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=100)
    ci = res.conf_int()
    return pd.DataFrame({
        "outcome": y, "term": res.params.index,
        "log_odds": res.params.values, "odds_ratio": np.exp(res.params.values),
        "or_low": np.exp(ci[0].values), "or_high": np.exp(ci[1].values),
        "p": res.pvalues.values,
    })


# --------------------------------------------------------------------------- #
def main() -> None:
    raw = load()
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    d, blocks = build_design(raw)
    order = ["task_context", "model_identity", "retrieval_arm", "model_x_arm"]
    say(f"rows={len(d)}  task_keys={d[GROUP].nunique()}  conditions={d.condition.nunique()}")
    say(f"design columns: " + ", ".join(f"{k}={len(v)}" for k, v in blocks.items()))
    say(f"reference cell: {MODEL_REF} / {ARM_REF}")

    imp_all, coef_all, con_all = [], [], []

    def report(y, data, blks, ordr, spec, wide=False):
        imp, full, sub = importance(data, y, blks, ordr)
        imp_all.append(imp)
        coef_all.append(tidy(full, sub, y, spec))
        con_all.append(arm_contrasts(full, y, sorted(data["model"].unique()), spec))
        r = imp[imp.block == "__full_model__"].iloc[0]
        say(f"  {PRETTY.get(y,y):<34} n={int(r.n_obs):>6}  ICC0={r.icc_task_empty:.3f}  "
            f"R2m={r.r2_marginal:.3f}  R2c={r.r2_conditional:.3f}")
        for _, row in imp[imp.block != "__full_model__"].iterrows():
            say(f"      {row.block:<15} dR2seq={row.delta_r2_sequential: .4f}  "
                f"dR2uniq={row.delta_r2_unique: .4f}  chi2={row.wald_chi2:>9.1f} "
                f"df={int(row.wald_df)} p={row.wald_p:.3g}")
        return full, sub

    say("\n=== QUALITY (all four arms) ===")
    for y in QUALITY:
        report(y, d, blocks, order, "treatment_only_all_arms")

    say("\n=== ANSWER / BEHAVIOUR (all four arms) ===")
    for y in BEHAVIOR:
        report(y, d, blocks, order, "treatment_only_all_arms")

    say("\n=== RETRIEVED EVIDENCE (Normalized vs Wide, 4x2) ===")
    ev_raw = raw[raw["arm"].isin(["Normalized", "Wide"])].copy()
    ev, ev_blocks = build_design(ev_raw)
    for y in EVIDENCE:
        report(y, ev, ev_blocks, order, "treatment_only_wide_vs_norm")
    say("  (quality on the same 4x2 subset, for a like-for-like comparison)")
    for y in QUALITY:
        report(y, ev, ev_blocks, order, "quality_on_wide_vs_norm_subset")

    pd.concat(imp_all, ignore_index=True).to_csv(OUT / "mixed_importance.csv", index=False)
    pd.concat(coef_all, ignore_index=True).to_csv(OUT / "mixed_fixed_effects.csv", index=False)
    con = pd.concat(con_all, ignore_index=True)
    # Benjamini-Hochberg within each outcome family: many contrasts are tested.
    from statsmodels.stats.multitest import multipletests

    con["p_bh"] = np.nan
    for y, idx in con.groupby("outcome").groups.items():
        ps = con.loc[idx, "p"].to_numpy()
        ok = np.isfinite(ps)
        adj = np.full(len(ps), np.nan)
        if ok.sum():
            adj[ok] = multipletests(ps[ok], method="fdr_bh")[1]
        con.loc[idx, "p_bh"] = adj
    con.to_csv(OUT / "mixed_contrasts.csv", index=False)

    say("\n=== PER-MODEL ARM CONTRASTS (percentage points / native units) ===")
    for y in QUALITY + BEHAVIOR + EVIDENCE:
        c = con[(con.outcome == y) & (con.contrast != "difference in retrieval gain")]
        c = c[c.spec.str.startswith("treatment_only")]
        if c.empty:
            continue
        say(f"  -- {PRETTY.get(y, y)}")
        say(c[["model", "contrast", "estimate", "ci_low", "ci_high", "p", "p_bh"]]
            .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    say("\n=== MODEL x RETRIEVAL INTERACTION (difference in retrieval gain) ===")
    dif = con[(con.contrast == "difference in retrieval gain")
              & con.spec.str.startswith("treatment_only")]
    say(dif[["outcome", "model", "estimate", "ci_low", "ci_high", "p", "p_bh"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    say("\n=== MEDIATION: Wide vs Normalized (descriptive; mediators are post-treatment) ===")
    med = pd.concat([mediation(raw, y) for y in QUALITY], ignore_index=True)
    med.to_csv(OUT / "mixed_mediation.csv", index=False)
    a = med[med.path == "a_wide_to_mediator"].drop_duplicates("mediator")
    say("  a-paths (Wide - Normalized on each mediator):")
    for _, r in a.iterrows():
        say(f"      {PRETTY.get(r.mediator, r.mediator):<34} {r.estimate:+.4f}  (p={r.p:.3g})")
    for y in QUALITY:
        m = med[med.outcome == y]
        tot = m[m.path == "total_wide_effect"].estimate.iloc[0]
        dr = m[m.path == "direct_wide_effect_adj_mediators"].estimate.iloc[0]
        say(f"  {PRETTY[y]}: total Wide={tot:+.4f}  direct(adj)={dr:+.4f}  "
            f"indirect share={(tot - dr) / tot if tot else float('nan'):+.2f}")
        bb = m[m.path == "b_mediator_to_outcome_per_SD"].copy()
        bb["abs"] = bb.estimate.abs()
        for _, r in bb.sort_values("abs", ascending=False).iterrows():
            say(f"      per-SD {PRETTY.get(r.mediator, r.mediator):<32} "
                f"{r.estimate:+.4f}  (p={r.p:.3g})")

    say("\n=== MODERATION: does the retrieval gain depend on the task? ===")
    mod = pd.concat([moderation(raw, y) for y in QUALITY], ignore_index=True)
    mod.to_csv(OUT / "mixed_moderation.csv", index=False)
    say(mod.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    say("\n=== HETEROGENEITY of the retrieval effect across questions ===")
    het = pd.DataFrame([heterogeneity(raw, y) for y in QUALITY])
    het.to_csv(OUT / "mixed_heterogeneity.csv", index=False)
    say(het.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    say("\n=== GEE LOGISTIC robustness (exchangeable, clustered on task_key) ===")
    cols = [c for b in order for c in blocks[b]]
    gee = pd.concat([gee_logit(d, y, cols) for y in QUALITY], ignore_index=True)
    gee.to_csv(OUT / "mixed_gee_logistic.csv", index=False)
    for y in QUALITY:
        g = gee[(gee.outcome == y) & gee.term.str.startswith(("arm_", "mdl_", "ix_"))]
        say(f"  -- {PRETTY[y]}")
        say(g[["term", "odds_ratio", "or_low", "or_high", "p"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    (OUT / "mixed_effects_run.log").write_text("\n".join(log) + "\n")
    print("\nwrote mixed_importance.csv mixed_fixed_effects.csv mixed_contrasts.csv "
          "mixed_mediation.csv mixed_moderation.csv mixed_heterogeneity.csv "
          "mixed_gee_logistic.csv "
          "mixed_effects_run.log")


if __name__ == "__main__":
    main()
