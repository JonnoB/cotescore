import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import marimo as mo
    from string import ascii_uppercase
    from cot_score.metrics import jensen_shannon_divergence
    return ascii_uppercase, jensen_shannon_divergence, np, pd, plt, sns


@app.cell
def _():
    # === Configuration ===
    BASE_COUNT = 2000       # count per character A-T
    FRAC_MAX = 0.2        # maximum corruption fraction
    FRAC_STEP = 0.001       # step size for corruption fractions
    return BASE_COUNT, FRAC_MAX, FRAC_STEP


@app.cell
def _(BASE_COUNT, ascii_uppercase, jensen_shannon_divergence, np):
    # Ground truth distribution Q
    # A:T (first 20 letters) = BASE_COUNT each, U = 1, "0" = 0
    VOCAB = list(ascii_uppercase[:21]) + ["0"]  # A-U + "0"

    Q_gt = {c: BASE_COUNT for c in ascii_uppercase[:20]}  # A-T
    Q_gt["U"] = 1
    Q_gt["0"] = 0

    def counts_to_prob(counts):
        """Convert a count dict to a numpy probability array aligned to VOCAB."""
        _arr = np.array([counts.get(c, 0) for c in VOCAB], dtype=float)
        _total = _arr.sum()
        if _total > 0:
            return _arr / _total
        return _arr

    def compute_cdd(counts_a, counts_b):
        """CDD = sqrt(JSD) between two count dicts."""
        _p = counts_to_prob(counts_a)
        _q = counts_to_prob(counts_b)
        _jsd = jensen_shannon_divergence(_p, _q)
        return np.sqrt(max(_jsd, 0.0))

    def compute_tv(counts_a, counts_b):
        """Total Variation distance V(P,Q) = (1/2) * sum(|p_i - q_i|)."""
        _p = counts_to_prob(counts_a)
        _q = counts_to_prob(counts_b)
        return 0.5 * np.sum(np.abs(_p - _q))
    return Q_gt, compute_cdd, compute_tv


@app.cell
def _(ascii_uppercase):
    def apply_parse_corruption(counts, fraction):
        """Remove fraction of total counts from A->T sequentially, place on U."""
        _r = dict(counts)
        _total = sum(_r.values())
        _to_remove = round(fraction * _total)
        _source_chars = list(ascii_uppercase[:20])  # A-T

        for _c in _source_chars:
            if _to_remove <= 0:
                break
            _take = min(_r[_c], _to_remove)
            _r[_c] -= _take
            _r["U"] += _take
            _to_remove -= _take
        return _r

    def apply_ocr_corruption(counts, fraction, mode):
        """Remove fraction of total counts and place on '0'.

        Modes:
            'hiding': preferentially remove from U first, then A->T
            'proportional': remove evenly from all A:U with non-zero counts
            'sequential': remove from A->T sequentially (no interaction)
        """
        _r = dict(counts)
        _total = sum(_r.values())
        _to_remove = round(fraction * _total)

        if mode == "hiding":
            # Take from U first
            _take = min(_r["U"], _to_remove)
            _r["U"] -= _take
            _r["0"] += _take
            _to_remove -= _take
            # Then A->T sequentially
            for _c in list(ascii_uppercase[:20]):
                if _to_remove <= 0:
                    break
                _take = min(_r[_c], _to_remove)
                _r[_c] -= _take
                _r["0"] += _take
                _to_remove -= _take

        elif mode == "proportional":
            # Distribute removal evenly across A:U (characters with non-zero counts)
            _candidates = [c for c in list(ascii_uppercase[:21]) if _r.get(c, 0) > 0]
            if _candidates:
                _per_char = _to_remove // len(_candidates)
                _remainder = _to_remove % len(_candidates)
                for _i, _c in enumerate(_candidates):
                    _take = _per_char + (1 if _i < _remainder else 0)
                    _take = min(_r[_c], _take)
                    _r[_c] -= _take
                    _r["0"] += _take

        elif mode == "sequential":
            # A->T sequentially, never touches U
            for _c in list(ascii_uppercase[:20]):
                if _to_remove <= 0:
                    break
                _take = min(_r[_c], _to_remove)
                _r[_c] -= _take
                _r["0"] += _take
                _to_remove -= _take

        return _r
    return apply_ocr_corruption, apply_parse_corruption


@app.cell
def _(
    FRAC_MAX,
    FRAC_STEP,
    Q_gt,
    apply_ocr_corruption,
    apply_parse_corruption,
    compute_cdd,
    compute_tv,
    pd,
):
    # Run simulation across all modes
    _n_steps = round(FRAC_MAX / FRAC_STEP)
    _round_digits = max(0, len(str(FRAC_STEP).rstrip('0').split('.')[-1]))
    _fracs = [round(i * FRAC_STEP, _round_digits) for i in range(_n_steps + 1)]
    _modes = ["hiding", "proportional", "sequential"]
    _results = {m: [] for m in _modes}

    for _pf in _fracs:
        _R = apply_parse_corruption(Q_gt, _pf)
        _d_parse = compute_cdd(_R, Q_gt)

        for _of in _fracs:
            for _mode in _modes:
                _P_ocr = apply_ocr_corruption(_R, _of, _mode)
                _d_ocr = compute_cdd(_P_ocr, _R)
                _d_total = compute_cdd(_P_ocr, Q_gt)
                _tv_ocr = compute_tv(_P_ocr, _R)
                _tv_total = compute_tv(_P_ocr, Q_gt)
                _results[_mode].append({
                    "parse_frac": _pf,
                    "ocr_frac": _of,
                    "d_total": _d_total,
                    "d_parse": _d_parse,
                    "d_ocr": _d_ocr,
                    "tv_ocr": _tv_ocr,
                    "tv_total": _tv_total,
                })

    df_hiding = pd.DataFrame(_results["hiding"])
    df_proportional = pd.DataFrame(_results["proportional"])
    df_sequential = pd.DataFrame(_results["sequential"])
    return df_hiding, df_proportional, df_sequential


@app.cell
def _(df_hiding, df_proportional, df_sequential, np):
    # Add derived columns
    def add_derived(df):
        _df = df.copy()
        _df["d_sum"] = _df["d_parse"] + _df["d_ocr"]
        _df["triangle_gap"] = _df["d_sum"] - _df["d_total"]
        _df["parse_minus_ocr"] = _df["d_parse"] - _df["d_ocr"]
        # Lin upper bound: d_ocr <= sqrt(CER), so d_total <= 2*sqrt(CER) at critical point
        _df["cer_bound_lin"] = 2 * np.sqrt(_df["tv_ocr"])
        _df["exceeds_lin"] = _df["d_total"] > _df["cer_bound_lin"]
        # Pinsker lower bound: d_ocr >= CER / (2*sqrt(2*ln2))
        _pinsker_const = 1 / (2 * np.sqrt(2 * np.log(2)))
        _df["pinsker_lower"] = _pinsker_const * _df["tv_ocr"]
        return _df

    df_hiding_d = add_derived(df_hiding)
    df_proportional_d = add_derived(df_proportional)
    df_sequential_d = add_derived(df_sequential)
    return df_hiding_d, df_proportional_d, df_sequential_d


@app.cell
def _(np, plt, sns):
    def plot_mode_heatmaps(df, mode_name):
        """Plot 3 heatmaps for a given mode: d_total, triangle_gap, parse_minus_ocr."""
        _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
        _fig.suptitle(f"Mode: {mode_name}", fontsize=14, fontweight="bold")

        _metrics = [
            ("d_total", "Total CDD", "rocket"),
            ("triangle_gap", "Triangle Gap (d_parse + d_ocr - d_total)", "mako"),
            ("parse_minus_ocr", "d_parse - d_ocr", "coolwarm"),
        ]

        _n_cells = df["parse_frac"].nunique()
        _do_annot = _n_cells <= 15

        for _ax, (_col, _title, _cmap) in zip(_axes, _metrics):
            _pivot = df.pivot(index="parse_frac", columns="ocr_frac", values=_col)
            _pivot = _pivot.sort_index(ascending=False)

            _center = 0.0 if _col == "parse_minus_ocr" else None
            _annot_kws = {"size": max(4, 9 - _n_cells // 4)} if _do_annot else {}
            sns.heatmap(
                _pivot,
                ax=_ax,
                cmap=_cmap,
                center=_center,
                annot=_do_annot,
                fmt=".3f" if _do_annot else "",
                annot_kws=_annot_kws,
                linewidths=0.5 if _n_cells <= 20 else 0,
            )
            _ax.set_title(_title)
            _ax.set_xlabel("OCR error fraction")
            _ax.set_ylabel("Parse error fraction")

            # Add contour at 0 for parse_minus_ocr
            if _col == "parse_minus_ocr":
                _pivot_asc = df.pivot(index="parse_frac", columns="ocr_frac", values=_col)
                _pivot_asc = _pivot_asc.sort_index(ascending=True)
                _x = np.arange(len(_pivot_asc.columns)) + 0.5
                _y = np.arange(len(_pivot_asc.index)) + 0.5
                _X, _Y = np.meshgrid(_x, _y)
                _ax.contour(
                    _X,
                    len(_pivot_asc.index) - _Y,
                    _pivot_asc.values,
                    levels=[0.0],
                    colors="black",
                    linewidths=2,
                )

        plt.tight_layout()
        return _fig
    return (plot_mode_heatmaps,)


@app.cell
def _(df_hiding_d, plot_mode_heatmaps):
    fig_hiding = plot_mode_heatmaps(df_hiding_d, "Hiding")
    fig_hiding
    return


@app.cell
def _(df_proportional_d, plot_mode_heatmaps):
    fig_proportional = plot_mode_heatmaps(df_proportional_d, "Proportional")
    fig_proportional
    return


@app.cell
def _(df_sequential_d, plot_mode_heatmaps):
    fig_sequential = plot_mode_heatmaps(df_sequential_d, "Sequential")
    fig_sequential
    return


@app.cell
def _(df_hiding_d, df_proportional_d, df_sequential_d, np, plt):
    # Comparative summary: d_parse = d_ocr contour for all three modes
    _fig, _ax = plt.subplots(figsize=(8, 6))

    _modes_data = [
        (df_hiding_d, "Hiding", "tab:red"),
        (df_proportional_d, "Proportional", "tab:blue"),
        (df_sequential_d, "Sequential", "tab:green"),
    ]

    for _df, _label, _color in _modes_data:
        _pivot = _df.pivot(index="parse_frac", columns="ocr_frac", values="parse_minus_ocr")
        _pivot = _pivot.sort_index(ascending=True)
        _x = _pivot.columns.values
        _y = _pivot.index.values
        _X, _Y = np.meshgrid(_x, _y)
        _ax.contour(
            _X, _Y, _pivot.values,
            levels=[0.0],
            colors=_color,
            linewidths=2,
        )
        # Invisible plot for legend
        _ax.plot([], [], color=_color, linewidth=2, label=_label)

    _ax.set_xlabel("OCR error fraction")
    _ax.set_ylabel("Parse error fraction")
    _ax.set_title("Boundary where d_parse = d_ocr")
    _ax.legend()
    _ax.set_aspect("equal")
    plt.tight_layout()
    fig_comparison = _fig
    fig_comparison
    return


@app.cell
def _(df_sequential_d):
    df_sequential_d
    return


@app.cell
def _(df_hiding_d, df_proportional_d, df_real, df_sequential_d, np, plt):
    # Balance point figure: d_total at d_parse=d_ocr vs CER, with theoretical bounds
    _fig, _ax = plt.subplots(figsize=(9, 6))

    _pinsker_const = 1 / (2 * np.sqrt(2 * np.log(2)))

    _modes_data = [
        (df_hiding_d, "Hiding (synthetic)", "tab:red"),
        (df_proportional_d, "Proportional (synthetic)", "tab:blue"),
        (df_sequential_d, "Sequential (synthetic)", "tab:green"),
    ]

    # For each synthetic mode, find the balance point (d_parse = d_ocr) at each ocr_frac
    for _df, _label, _color in _modes_data:
        _balance_cer = []
        _balance_dtotal = []
        for _of in sorted(_df["ocr_frac"].unique()):
            if _of == 0:
                continue
            _group = _df[_df["ocr_frac"] == _of].sort_values("parse_frac")
            _pmo = _group["parse_minus_ocr"].values
            _dt = _group["d_total"].values
            # Find zero crossing of parse_minus_ocr
            _sign_changes = np.where(np.diff(np.sign(_pmo)))[0]
            if len(_sign_changes) > 0:
                _i = _sign_changes[0]
                _frac = -_pmo[_i] / (_pmo[_i + 1] - _pmo[_i])
                _dt_interp = _dt[_i] + _frac * (_dt[_i + 1] - _dt[_i])
                _balance_cer.append(_of)
                _balance_dtotal.append(_dt_interp)

        _ax.plot(
            _balance_cer, _balance_dtotal,
            color=_color, linewidth=1.5, label=_label,
        )

    # Realistic balance points (group by target_cer, interpolate across parse_frac)
    _balance_cer_real = []
    _balance_dtotal_real = []
    for _tc in sorted(df_real["target_cer"].unique()):
        if _tc == 0:
            continue
        _group = df_real[df_real["target_cer"] == _tc].sort_values("parse_frac")
        _pmo = _group["parse_minus_ocr"].values
        _dt = _group["d_total"].values
        _cer_vals = _group["cer"].values
        _sign_changes = np.where(np.diff(np.sign(_pmo)))[0]
        if len(_sign_changes) > 0:
            _i = _sign_changes[0]
            _frac = -_pmo[_i] / (_pmo[_i + 1] - _pmo[_i])
            _dt_interp = _dt[_i] + _frac * (_dt[_i + 1] - _dt[_i])
            _cer_interp = _cer_vals[_i] + _frac * (_cer_vals[_i + 1] - _cer_vals[_i])
            _balance_cer_real.append(_cer_interp)
            _balance_dtotal_real.append(_dt_interp)

    _ax.plot(
        _balance_cer_real, _balance_dtotal_real,
        color="tab:purple", linewidth=2, label="Realistic (scrambledtext)",
    )

    # Bound curves
    _cer_max = max(
        max((_df["ocr_frac"].max() for _df, _, _ in _modes_data)),
        max(_balance_cer_real) if _balance_cer_real else 0,
    )
    _cer_range = np.linspace(1e-6, _cer_max * 1.1, 200)
    _ax.plot(
        _cer_range, 2 * np.sqrt(_cer_range),
        "k--", linewidth=2,
        label=r"$2\sqrt{CER}$",
    )
    _ax.plot(
        _cer_range, np.sqrt(_cer_range),
        "k-.", linewidth=2,
        label=r"$\sqrt{CER}$",
    )
    _ax.plot(
        _cer_range, _cer_range,
        "k-", linewidth=1, alpha=0.6,
        label=r"$CER$",
    )
    _ax.plot(
        _cer_range, 2 * _pinsker_const * _cer_range,
        "k:", linewidth=1.5,
        label=r"$\frac{CER}{\sqrt{2\ln 2}}$ (Pinsker lower)",
    )

    _ax.set_xlabel(r"$CER_{ocr}$ (OCR error fraction)")
    _ax.set_ylabel(r"$d_{total}$ at balance point ($d_{parse} = d_{ocr}$)")
    _ax.set_title(r"Balance point $d_{total}$ vs CER with theoretical bounds")
    _ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig_balance = _fig
    fig_balance
    return


@app.cell
def _(FRAC_MAX, df_hiding_d, df_proportional_d, df_sequential_d, np, plt):
    # Precision & Recall for parse-dominance classification by CER threshold
    _fig, _axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")
    _fig.suptitle(
        "Precision & Recall for parse-dominance detection by CER threshold",
        fontsize=13, fontweight="bold",
    )

    _d_cap = 2 * np.sqrt(FRAC_MAX)  # d_total ceiling

    _modes_data = [
        (df_hiding_d, "Hiding"),
        (df_proportional_d, "Proportional"),
        (df_sequential_d, "Sequential"),
    ]

    _thresholds = [
        (r"$2\sqrt{CER}$", lambda cer: 2 * np.sqrt(cer), "tab:red", "--"),
        (r"$\sqrt{CER}$", lambda cer: np.sqrt(cer), "tab:blue", "-"),
        (r"$CER$", lambda cer: cer, "tab:green", "-."),
    ]

    for _col_idx, (_df, _mode_name) in enumerate(_modes_data):
        _ax_prec = _axes[0, _col_idx]
        _ax_rec = _axes[1, _col_idx]

        for _thresh_label, _thresh_fn, _color, _ls in _thresholds:
            _cers = []
            _precisions = []
            _recalls = []
            for _of in sorted(_df["ocr_frac"].unique()):
                if _of == 0:
                    continue
                _group = _df[(_df["ocr_frac"] == _of) & (_df["d_total"] < _d_cap)]
                if len(_group) == 0:
                    continue
                _T = _thresh_fn(_of)
                _actual_parse = _group["d_parse"] > _group["d_ocr"]
                _predicted_parse = _group["d_total"] > _T
                _tp = (_actual_parse & _predicted_parse).sum()
                _fp = (~_actual_parse & _predicted_parse).sum()
                _fn = (_actual_parse & ~_predicted_parse).sum()
                # Precision: of those flagged, how many are truly parse-dominant
                _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else float("nan")
                # Recall: of truly parse-dominant, how many are flagged
                _rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else float("nan")
                _cers.append(_of)
                _precisions.append(_prec)
                _recalls.append(_rec)

            _ax_prec.plot(_cers, _precisions, color=_color, linestyle=_ls, linewidth=2, label=_thresh_label)
            _ax_rec.plot(_cers, _recalls, color=_color, linestyle=_ls, linewidth=2, label=_thresh_label)

        _ax_prec.set_title(_mode_name)
        _ax_prec.set_ylim(-0.05, 1.05)
        _ax_prec.legend(fontsize=8)
        _ax_rec.set_xlabel("CER (OCR error fraction)")
        _ax_rec.set_ylim(-0.05, 1.05)
        _ax_rec.legend(fontsize=8)

    _axes[0, 0].set_ylabel("Precision")
    _axes[1, 0].set_ylabel("Recall")
    plt.tight_layout()
    fig_prec_recall = _fig
    fig_prec_recall
    return


@app.cell
def _(
    FRAC_MAX,
    df_hiding_d,
    df_proportional_d,
    df_real,
    df_sequential_d,
    np,
    plt,
):
    # F1 score for parse-dominance classification by CER threshold (2x2: 3 synthetic + realistic)
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    _fig.suptitle(
        "F1 score for parse-dominance detection by CER threshold based on corruption type",
        fontsize=13, fontweight="bold",
    )

    _d_cap = 2 * np.sqrt(FRAC_MAX)  # d_total ceiling

    _thresholds = [
        (r"$2\sqrt{CER}$", lambda cer: 2 * np.sqrt(cer), "tab:red", "--"),
        (r"$\sqrt{CER}$", lambda cer: np.sqrt(cer), "tab:blue", "-"),
        (r"$CER$", lambda cer: cer, "tab:green", "-."),
    ]

    # Synthetic modes (first 3 panels)
    _synthetic_modes = [
        (df_hiding_d, "Hiding (synthetic)"),
        (df_proportional_d, "Proportional (synthetic)"),
        (df_sequential_d, "Sequential (synthetic)"),
    ]

    for _ax, (_df, _mode_name) in zip(_axes.flat[:3], _synthetic_modes):
        for _thresh_label, _thresh_fn, _color, _ls in _thresholds:
            _cers = []
            _f1s = []
            for _of in sorted(_df["ocr_frac"].unique()):
                if _of == 0:
                    continue
                _group = _df[(_df["ocr_frac"] == _of) & (_df["d_total"] < _d_cap)]
                if len(_group) == 0:
                    continue
                _T = _thresh_fn(_of)
                _actual_parse = _group["d_parse"] > _group["d_ocr"]
                _predicted_parse = _group["d_total"] > _T
                _tp = (_actual_parse & _predicted_parse).sum()
                _fp = (~_actual_parse & _predicted_parse).sum()
                _fn = (_actual_parse & ~_predicted_parse).sum()
                _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
                _rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
                _f1 = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0
                _cers.append(_of)
                _f1s.append(_f1)

            _ax.plot(_cers, _f1s, color=_color, linestyle=_ls, linewidth=2, label=_thresh_label)

        _ax.set_title(_mode_name)
        _ax.set_ylim(-0.05, 1.05)
        _ax.legend(fontsize=8)

    # Realistic mode (4th panel) — bin by actual CER
    _ax_real = _axes[1, 1]
    _cer_bins = np.arange(0, df_real["cer"].max() + 0.005, 0.005)

    for _thresh_label, _thresh_fn, _color, _ls in _thresholds:
        _cers_plot = []
        _f1s_plot = []
        for _bin_start, _bin_end in zip(_cer_bins[:-1], _cer_bins[1:]):
            _bin_mid = (_bin_start + _bin_end) / 2
            if _bin_mid == 0:
                continue
            _group = df_real[(df_real["cer"] >= _bin_start)
                            & (df_real["cer"] < _bin_end)
                            & (df_real["d_total"] < _d_cap)]
            if len(_group) == 0:
                continue
            _T = _thresh_fn(_bin_mid)
            _actual_parse = _group["d_parse"] > _group["d_ocr"]
            _predicted_parse = _group["d_total"] > _T
            _tp = (_actual_parse & _predicted_parse).sum()
            _fp = (~_actual_parse & _predicted_parse).sum()
            _fn = (_actual_parse & ~_predicted_parse).sum()
            _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
            _rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
            _f1 = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0
            _cers_plot.append(_bin_mid)
            _f1s_plot.append(_f1)

        _ax_real.plot(_cers_plot, _f1s_plot, color=_color, linestyle=_ls,
                      linewidth=2, label=_thresh_label)

    _ax_real.set_title("Realistic (scrambledtext + limerick)")
    _ax_real.set_ylim(-0.05, 1.05)
    _ax_real.legend(fontsize=8)

    # Axis labels
    for _ax in _axes[1, :]:
        _ax.set_xlabel("CER")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("F1 Score")

    plt.tight_layout()
    fig_f1 = _fig
    fig_f1
    return


@app.cell
def _():
    # Cell R1: Load limerick characters + scrambledtext distributions
    import json
    import pkg_resources
    from scrambledtext import ProbabilityDistributions
    from collections import Counter

    # Load limerick ground truth (exported by limerick_case_study.py)
    with open("data/limerick_case_study/ground_truth.json") as _f:
        _gt = json.load(_f)

    # Extract all characters with spatial positions
    _all_chars = []
    for _block in _gt["blocks"].values():
        for _para in _block.get("paragraphs", []):
            for _c in _para.get("characters", []):
                _all_chars.append(_c)

    # Sort by reading order: top-to-bottom then left-to-right (y then x of bbox centre)
    _all_chars.sort(key=lambda c: (c["bbox"][1] + c["bbox"][3] / 2, c["bbox"][0] + c["bbox"][2] / 2))
    real_chars_sorted = _all_chars

    # Build ground truth count dict
    Q_real = dict(Counter(c["char"] for c in _all_chars))

    # Load scrambledtext distributions
    _json_path = pkg_resources.resource_filename("scrambledtext", "corruption_distribs.json")
    real_distribs = ProbabilityDistributions.load_from_json(_json_path)

    # Add overflow character for parsing error
    OVERFLOW_CHAR = "\x00"  # null byte as overflow marker

    print(f"Limerick: {len(_all_chars)} chars, {len(Q_real)} unique")
    return OVERFLOW_CHAR, Q_real, real_chars_sorted, real_distribs


@app.cell
def _(OVERFLOW_CHAR, jensen_shannon_divergence, np):
    # Cell R2: Helpers and CorruptionEngine setup
    from collections import Counter as _Counter
    from scrambledtext import CorruptionEngine
    from jiwer import cer as _jiwer_cer

    def _unified_vocab(*count_dicts):
        _keys = set()
        for _d in count_dicts:
            _keys.update(_d.keys())
        return sorted(_keys)

    def real_compute_cdd(counts_a, counts_b):
        """CDD = sqrt(JSD) between two count dicts (dynamic vocab)."""
        _vocab = _unified_vocab(counts_a, counts_b)
        _p = np.array([counts_a.get(c, 0) for c in _vocab], dtype=float)
        _q = np.array([counts_b.get(c, 0) for c in _vocab], dtype=float)
        _p_sum, _q_sum = _p.sum(), _q.sum()
        if _p_sum > 0:
            _p = _p / _p_sum
        if _q_sum > 0:
            _q = _q / _q_sum
        _jsd = jensen_shannon_divergence(_p, _q)
        return np.sqrt(max(_jsd, 0.0))

    def apply_real_parse_corruption(chars_sorted, fraction):
        """Remove last fraction of characters by reading order, move to overflow.

        Returns (R_counts, kept_text, overflow_count).
        """
        _n = len(chars_sorted)
        _n_remove = round(fraction * _n)
        _keep = chars_sorted[:_n - _n_remove] if _n_remove > 0 else chars_sorted
        _kept_text = "".join(c["char"] for c in _keep)
        _R = dict(_Counter(c["char"] for c in _keep))
        if _n_remove > 0:
            _R[OVERFLOW_CHAR] = _R.get(OVERFLOW_CHAR, 0) + _n_remove
        return _R, _kept_text, _n_remove

    def corrupt_text_realistic(text, target_cer, distribs):
        """Corrupt text using scrambledtext CorruptionEngine."""
        if target_cer == 0 or not text:
            return text, 0.0
        _engine = CorruptionEngine(
            distribs.conditional,
            distribs.substitutions,
            distribs.insertions,
            target_wer=1.0,
            target_cer=target_cer,
        )
        _corrupted, _, _, _ = _engine.corrupt_text(text)
        _actual_cer = _jiwer_cer(text, _corrupted)
        return _corrupted, _actual_cer
    return (
        apply_real_parse_corruption,
        corrupt_text_realistic,
        real_compute_cdd,
    )


@app.cell
def _(
    FRAC_MAX,
    OVERFLOW_CHAR,
    Q_real,
    apply_real_parse_corruption,
    corrupt_text_realistic,
    np,
    pd,
    real_chars_sorted,
    real_compute_cdd,
    real_distribs,
):
    # Cell R4: Run realistic simulation (text-level corruption via CorruptionEngine)
    from collections import Counter as _Counter

    _parse_step = 0.01
    _parse_fracs = [round(i * _parse_step, 2) for i in range(int(FRAC_MAX / _parse_step) + 1)]
    _target_cers = [round(i * 0.005, 3) for i in range(int(FRAC_MAX / 0.005) + 1)]
    _n_repeats = 10

    _results = []
    for _pf in _parse_fracs:
        _R_counts, _kept_text, _overflow_n = apply_real_parse_corruption(real_chars_sorted, _pf)
        _d_parse = real_compute_cdd(_R_counts, Q_real)

        for _target_cer in _target_cers:
            _d_ocrs, _d_totals, _actual_cers = [], [], []
            for _ in range(_n_repeats):
                _corrupted, _actual_cer = corrupt_text_realistic(
                    _kept_text, _target_cer, real_distribs
                )
                _obs_counts = dict(_Counter(_corrupted))
                if _overflow_n > 0:
                    _obs_counts[OVERFLOW_CHAR] = _obs_counts.get(OVERFLOW_CHAR, 0) + _overflow_n
                _d_ocrs.append(real_compute_cdd(_obs_counts, _R_counts))
                _d_totals.append(real_compute_cdd(_obs_counts, Q_real))
                _actual_cers.append(_actual_cer)

            _results.append({
                "parse_frac": _pf,
                "target_cer": _target_cer,
                "cer": np.mean(_actual_cers),
                "d_total": np.mean(_d_totals),
                "d_parse": _d_parse,
                "d_ocr": np.mean(_d_ocrs),
            })

    df_real = pd.DataFrame(_results)
    df_real["parse_minus_ocr"] = df_real["d_parse"] - df_real["d_ocr"]
    print(f"Realistic simulation: {len(df_real)} rows ({_n_repeats} repeats averaged)")
    print(f"CER range: {df_real['cer'].min():.4f} to {df_real['cer'].max():.4f}")
    return (df_real,)


if __name__ == "__main__":
    app.run()
