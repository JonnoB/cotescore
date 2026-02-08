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
def _(df_hiding_d, df_proportional_d, df_sequential_d, np, plt):
    # Balance point figure: d_total at d_parse=d_ocr vs CER, with theoretical bounds
    _fig, _ax = plt.subplots(figsize=(9, 6))

    _pinsker_const = 1 / (2 * np.sqrt(2 * np.log(2)))

    _modes_data = [
        (df_hiding_d, "Hiding", "tab:red", "o"),
        (df_proportional_d, "Proportional", "tab:blue", "s"),
        (df_sequential_d, "Sequential", "tab:green", "^"),
    ]

    # For each mode, find the balance point (d_parse = d_ocr) at each ocr_frac
    for _df, _label, _color, _marker in _modes_data:
        _balance_cer = []
        _balance_dtotal = []
        for _of in sorted(_df["ocr_frac"].unique()):
            if _of == 0:
                continue
            _group = _df[_df["ocr_frac"] == _of].sort_values("parse_frac")
            _pmo = _group["parse_minus_ocr"].values
            _pf = _group["parse_frac"].values
            _dt = _group["d_total"].values
            # Find zero crossing of parse_minus_ocr
            _sign_changes = np.where(np.diff(np.sign(_pmo)))[0]
            if len(_sign_changes) > 0:
                _i = _sign_changes[0]
                # Linear interpolation to find exact crossing
                _frac = -_pmo[_i] / (_pmo[_i + 1] - _pmo[_i])
                _dt_interp = _dt[_i] + _frac * (_dt[_i + 1] - _dt[_i])
                _balance_cer.append(_of)
                _balance_dtotal.append(_dt_interp)

        _ax.plot(
            _balance_cer, _balance_dtotal,
            color=_color, marker=_marker, markersize=5,
            linewidth=1.5, label=_label,
        )

    # Bound curves
    _cer_max = max(_df["ocr_frac"].max() for _df, _, _, _ in _modes_data)
    _cer_range = np.linspace(1e-6, _cer_max * 1.1, 200)
    _ax.plot(
        _cer_range, 2 * np.sqrt(_cer_range),
        "k--", linewidth=2,
        label=r"$2\sqrt{CER}$ (Lin upper: above $\Rightarrow$ parsing dominates)",
    )
    _ax.plot(
        _cer_range, np.sqrt(_cer_range),
        "k-.", linewidth=2,
        label=r"$\sqrt{CER}$ (half-Lin: below $\Rightarrow$ OCR dominates)",
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
def _(FRAC_MAX, df_hiding_d, df_proportional_d, df_sequential_d, np, plt):
    # F1 score for parse-dominance classification by CER threshold
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    _fig.suptitle(
        "F1 score for parse-dominance detection by CER threshold",
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

    for _ax, (_df, _mode_name) in zip(_axes, _modes_data):
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
        _ax.set_xlabel("CER (OCR error fraction)")
        _ax.set_ylim(-0.05, 1.05)
        _ax.legend(fontsize=9)

    _axes[0].set_ylabel("F1 Score")
    plt.tight_layout()
    fig_f1 = _fig
    fig_f1
    return


if __name__ == "__main__":
    app.run()
