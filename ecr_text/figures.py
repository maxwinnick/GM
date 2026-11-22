"""Phase 1 figures in the GM1 mid-term slide format (Pareto A / Pareto B).

Format reproduced from the mid-term figures:
* Navy title (Lato Bold), dark subtitle "Level n: ...", grey footnote bottom-right.
* Pareto bars: navy "Vital few (to 80%)", light "Remaining", lighter "All other (n more)".
* Orange cumulative-% line with markers on a right axis, a "Cum. %" row above the bars,
  grey dashed 80% line, bold navy value labels on bars.
* Heat tables: white-to-navy cells with % labels, row labels at left, wrapped column headers.

Numbering continues the mid-term: Pareto C = word frequency (Phase 1), Pareto D = ECR count by
group. Step 4 histograms use the same palette and typography.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import RESOURCE_DIR, TEXT_COLUMNS  # noqa: E402
from .pareto import NO_GROUP_PREFIX  # noqa: E402
from .phase1 import NGRAMS, segment_ngrams  # noqa: E402

log = logging.getLogger(__name__)

# ---- mid-term palette (sampled from the GM1 mid-term figures) --------------------------------
NAVY, ORANGE = "#12245c", "#d9822b"
REMAINING, ALL_OTHER = "#a7b1cc", "#d5d9e2"
INK, MUTED, GRID, AXIS = "#1f2937", "#6b7280", "#e6e9ef", "#c9ced8"
HEAT = LinearSegmentedColormap.from_list("midterm_navy", ["#ffffff", "#e3e7f0", "#a7b1cc", "#4a5a8a", NAVY])
SOURCE = "Source: team analysis of the GM masked ECR extract."

for ttf in (RESOURCE_DIR / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))
FONT = "Lato" if any(f.name == "Lato" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
})

DIM_TITLE = {"Program": "Lead Program", "Supplier": "Supplier", "VPPS_L1": "VPPS System (L1)",
             "VPPS_L2": "VPPS Subsystem (L2)", "VPPS_L3": "Assembly (VPPS L3)"}
DIM_NOUN = {"Program": "lead program", "Supplier": "supplier", "VPPS_L1": "VPPS L1 code",
            "VPPS_L2": "VPPS L2 code", "VPPS_L3": "assembly"}
DIM_PLURAL = {"Program": "lead programs", "Supplier": "suppliers", "VPPS_L1": "VPPS L1 codes",
              "VPPS_L2": "VPPS L2 codes", "VPPS_L3": "assemblies"}


def _a(noun: str) -> str:
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun
FIELD_TITLE = {"ALL": "Title + Description + Reason", "TITLE": "Title", "DESC_FOR_CHANGE": "Description for Change",
               "REASON_FOR_CHANGE": "Reason for Change", "COST_COMMENT": "Cost Comment"}
FIGSIZE, DPI = (13, 7.15), 160


def _wrap(s: str, width: int = 14, lines: int = 3) -> str:
    s = str(s)
    if "\n" in s:  # pre-formatted (e.g. assembly code + description)
        return s
    w = textwrap.wrap(s, width) or [""]
    if len(w) > lines:
        w = w[:lines]
        w[-1] = w[-1][: width - 1] + "…"
    return "\n".join(w)


def _frame(title: str, subtitle: str, footnote: str | None, figsize=FIGSIZE):
    fig = plt.figure(figsize=figsize, dpi=DPI)
    fig.text(0.012, 0.965, title, fontsize=20, fontweight="bold", color=NAVY, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=14, color=INK, va="top")
    if footnote:
        fig.text(0.99, 0.012, footnote, fontsize=10, color=MUTED, ha="right", va="bottom")
    return fig


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def pareto(labels, values, cum_share, *, title, subtitle, footnote, path, kinds=None,
           ylabel="Number of ECRs", wrap=14, label_size=11) -> None:
    """Mid-term Pareto: bars + 'Cum. %' row + orange cumulative line on a right axis.

    kinds: per bar "vital" | "remaining" | "other" (colors and legend entries).
    """
    labels, values, cum = list(labels), np.asarray(values, float), np.asarray(cum_share, float)
    kinds = kinds or ["vital"] * len(values)
    color = {"vital": NAVY, "remaining": REMAINING, "other": ALL_OTHER}
    fig = _frame(title, subtitle, footnote)
    ax = fig.add_axes([0.075, 0.2, 0.86, 0.55])
    x = np.arange(len(values))
    ax.bar(x, values, width=0.66, color=[color[k] for k in kinds], zorder=2)
    vmax = values.max() if len(values) else 1
    ax.set_ylim(0, vmax * 1.12)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_xticks(x, [_wrap(lb, wrap) for lb in labels], fontsize=label_size, color=INK)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=12, length=0)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", color=NAVY)
    ax.spines["left"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(x, cum * 100, color=ORANGE, lw=3, marker="o", ms=8, zorder=5)
    ax2.axhline(80, color=MUTED, lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax2.text(len(values) - 0.45, 81, "80%", ha="right", va="bottom", fontsize=10, color=MUTED)
    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax2.tick_params(axis="y", colors=ORANGE, labelsize=12, length=0)
    ax2.set_ylabel("Cumulative %", fontsize=14, fontweight="bold", color=ORANGE)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(AXIS)
    # Value labels live on the top axis so the cumulative line never hides them.
    for xi, v in zip(x, values):
        ax2.text(xi, v + vmax * 0.015, f"{v:,.0f}", transform=ax.transData, ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=NAVY, zorder=6,
                 path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])

    # "Cum. %" row above the bars
    trans = ax.get_xaxis_transform()
    ax.text(-0.6, 1.07, "Cum. %", transform=trans, ha="right", va="center", fontsize=11,
            fontweight="bold", color=ORANGE)
    for xi, c in zip(x, cum):
        ax.text(xi, 1.07, f"{c * 100:.0f}%", transform=trans, ha="center", va="center", fontsize=12,
                fontweight="bold", color=ORANGE)
    handles = [Patch(color=NAVY, label="Vital few (to 80%)")]
    if "remaining" in kinds:
        handles.append(Patch(color=REMAINING, label="Remaining"))
    if "other" in kinds:
        handles.append(Patch(color=ALL_OTHER, label="All other"))
    handles.append(Line2D([], [], color=ORANGE, lw=3, marker="o", ms=8, label="Cumulative %"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.875), ncol=len(handles),
               frameon=False, fontsize=12, handlelength=1.6, columnspacing=2.2)
    _save(fig, path)


def heat_table(values: pd.DataFrame, *, title, subtitle, footnote, path, cell_text=None, vmax=None,
               fmt=lambda v: f"{v * 100:.0f}%", col_wrap=13, row_label_width=34) -> None:
    """Mid-term Pareto B Level 2 style table: white-to-navy cells with labels."""
    nr, nc = values.shape
    fig = _frame(title, subtitle, footnote, figsize=(13, max(5.0, 2.4 + 0.52 * nr)))
    h = fig.get_figheight()
    top = 1 - 1.75 / h
    ax = fig.add_axes([0.26, 0.06, 0.73, top - 0.06])
    vals = values.to_numpy(float)
    vmax = vmax or float(np.nanmax(vals)) or 1.0
    ax.imshow(np.nan_to_num(vals), cmap=HEAT, vmin=0, vmax=vmax, aspect="auto")
    for i in range(nr + 1):
        ax.axhline(i - 0.5, color="white", lw=3)
    for j in range(nc + 1):
        ax.axvline(j - 0.5, color="white", lw=3)
    for (i, j), v in np.ndenumerate(vals):
        txt = cell_text[i][j] if cell_text is not None else ("" if np.isnan(v) else fmt(v))
        dark = (not np.isnan(v)) and v > 0.5 * vmax
        ax.text(j, i, txt, ha="center", va="center", fontsize=11 if cell_text is None else 9.5,
                fontweight="bold" if dark else "normal", color="white" if dark else INK)
    ax.set_xticks(range(nc), [_wrap(c, col_wrap) for c in values.columns], fontsize=11, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(nr), [textwrap.shorten(str(r), row_label_width, placeholder="…") for r in values.index],
                  fontsize=11, color="#4b5563")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _save(fig, path)


def histogram(data, *, bins, title, subtitle, footnote, path, xlabel, ylabel="Number of ECRs",
              logx=False, mark_median=True) -> None:
    fig = _frame(title, subtitle, footnote)
    ax = fig.add_axes([0.075, 0.14, 0.88, 0.66])
    data = np.asarray(data, float)
    ax.hist(data, bins=bins, color=NAVY, edgecolor="white", linewidth=1.2, zorder=2)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_xlabel(xlabel, fontsize=13, color=INK)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", color=NAVY)
    ax.tick_params(labelsize=12, length=0)
    ax.spines["left"].set_visible(False)
    if mark_median and len(data):
        med = float(np.median(data))
        ax.axvline(med, color=ORANGE, lw=2.5, ls=(0, (5, 4)), zorder=3)
        ax.text(med, ax.get_ylim()[1] * 0.97, f"  Median {med:,.0f}", color=ORANGE, fontsize=12,
                fontweight="bold", va="top", ha="left")
    _save(fig, path)


def small_hists(panels: list[tuple[str, np.ndarray]], *, bins, title, subtitle, footnote, path, xlabel) -> None:
    fig = _frame(title, subtitle, footnote)
    n = len(panels)
    ncol = min(3, n) if n != 4 else 2
    nrow = int(np.ceil(n / ncol))
    gs = fig.add_gridspec(nrow, ncol, left=0.07, right=0.98, top=0.78, bottom=0.11, hspace=0.75, wspace=0.18)
    for k, (name, data) in enumerate(panels):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        ax.hist(data, bins=bins, color=NAVY, edgecolor="white", linewidth=1, zorder=2)
        med = float(np.median(data)) if len(data) else 0
        ax.axvline(med, color=ORANGE, lw=2, ls=(0, (5, 4)), zorder=3)
        ax.set_title(f"{name}\nmedian {med:,.0f}", loc="left", fontsize=11.5, fontweight="bold", color=NAVY)
        ax.grid(axis="y", color=GRID, zorder=0)
        ax.tick_params(labelsize=10, length=0)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        if k // ncol == nrow - 1:
            ax.set_xlabel(xlabel, fontsize=11)
    _save(fig, path)


def rank_frequency(series: dict[str, np.ndarray], *, title, subtitle, footnote, path) -> None:
    fig = _frame(title, subtitle, footnote)
    ax = fig.add_axes([0.085, 0.14, 0.87, 0.64])
    styles = {"unigram": (NAVY, "-"), "bigram": (ORANGE, "-"), "trigram": (REMAINING, "-")}
    for name, f in series.items():
        r = np.arange(1, len(f) + 1)
        c, ls = styles[name]
        ax.plot(r, f, color=c, ls=ls, lw=2.5, label=f"{name}s ({len(f):,} distinct)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(color=GRID, which="major")
    ax.set_xlabel("Rank of term by number of ECRs (log scale)", fontsize=13)
    ax.set_ylabel("Number of ECRs (log scale)", fontsize=14, fontweight="bold", color=NAVY)
    ax.tick_params(labelsize=12, length=0)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    _save(fig, path)


def hbar_share(labels, shares, *, title, subtitle, footnote, path, xlabel) -> None:
    fig = _frame(title, subtitle, footnote)
    ax = fig.add_axes([0.2, 0.1, 0.76, 0.7])
    y = np.arange(len(labels))[::-1]
    ax.barh(y, shares, color=NAVY, height=0.66, zorder=2)
    for yi, v in zip(y, shares):
        ax.text(v + 0.01, yi, f"{v * 100:.0f}%", va="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.set_yticks(y, labels, fontsize=12, color=INK)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.tick_params(length=0, labelsize=11)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.spines["bottom"].set_visible(False)
    _save(fig, path)


# ---------------------------------------------------------------------------------------------
# figure set
# ---------------------------------------------------------------------------------------------
def _field_df(df: pd.DataFrame, field: str, n: int) -> tuple[pd.Series, int]:
    col = f"{field}_NLP"
    has = df[TEXT_COLUMNS[field]].notna()
    terms = df.loc[has, col].map(lambda v: {t for t in segment_ngrams(v) if t.count(" ") + 1 == n})
    return terms.explode().dropna().value_counts(), int(has.sum())


def _word_pareto(counts: pd.Series, n_docs: int, *, title, subtitle, path, unit: str, top=10) -> dict:
    cum = (counts.cumsum() / counts.sum()).to_numpy()
    prev = cum - counts.to_numpy() / counts.sum()
    k = min(top, len(counts))
    kinds = ["vital" if p < 0.8 else "remaining" for p in prev[:k]]
    rest_terms = len(counts) - k
    rest_mentions = int(counts.iloc[k:].sum())
    foot = (f"Top {k} of {len(counts):,} {unit}s shown; remaining {rest_terms:,} {unit}s "
            f"({rest_mentions:,} ECR-{unit} mentions) not plotted. Each ECR counted once per {unit}.")
    pareto(counts.index[:k], counts.to_numpy()[:k], cum[:k], kinds=kinds, title=title, subtitle=subtitle,
           footnote=foot, path=path, wrap=11)
    return {"n_docs": n_docs, "distinct": int(len(counts)), "top": [(t, int(v)) for t, v in counts.head(k).items()],
            "top_cum": float(cum[k - 1]) if k else 0.0,
            "terms_for_80pct": int(np.searchsorted(cum, 0.8) + 1)}


def make_phase1_figures(p1: dict, out_dir: Path, top: int = 10, top_groups_l2: int = 3) -> dict:
    out = Path(out_dir)
    for old in out.rglob("*.png"):
        old.unlink()
    df, N = p1["df"], p1["n_ecr"]
    vocab, gt, ubq, labels = p1["vocab"], p1["group_tables"], p1["ubiquity"], p1["labels"]
    facts: dict = {"pareto_c_l1": {}, "pareto_d": {}, "pareto_c_l2": {}, "heat": {}, "lift": {},
                   "ubiquity": {}, "hist": {}}

    def glabel(dim, g, multiline=False):
        if dim == "VPPS_L3":
            full = labels["VPPS_L3"].get(g, g)
            if not multiline:
                return full
            desc = full[len(g):].strip()
            return g + ("\n" + _wrap(desc, 11, 2) if desc else "")
        return f"VPPS {g}" if dim.startswith("VPPS") else g

    # ---- Pareto C Level 1: words by ECR -------------------------------------------------------------
    for n, name in NGRAMS.items():
        cnt = vocab.loc[vocab.n == n].set_index("term")["DF"].sort_values(ascending=False, kind="stable")
        facts["pareto_c_l1"][f"ALL|{name}"] = _word_pareto(
            cnt, int(df["HAS_TEXT"].sum()), unit=name,
            title=f"Pareto C: Number of ECRs by {name.capitalize()}",
            subtitle=f"Level 1: {name.capitalize()}s in title + description + reason, all {N:,} ECRs, top {top}",
            path=out / "pareto_c" / f"C1_{name}_all_text.png", top=top)
        for f in ("TITLE", "DESC_FOR_CHANGE", "REASON_FOR_CHANGE", "COST_COMMENT"):
            cnt_f, nd = _field_df(df, f, n)
            facts["pareto_c_l1"][f"{f}|{name}"] = _word_pareto(
                cnt_f, nd, unit=name,
                title=f"Pareto C: Number of ECRs by {name.capitalize()}, {FIELD_TITLE[f]}",
                subtitle=f"Level 1: {name.capitalize()}s in {TEXT_COLUMNS[f]}, {nd:,} ECRs with text, top {top}",
                path=out / "pareto_c" / "by_field" / f"C1_{name}_{f}.png", top=top)

    # ---- Pareto D: ECR count by group (Pareto B layout with "All other") --------------------------------
    for dim, sizes in p1["group_sizes"].items():
        blank = int(sizes[sizes.index.str.startswith(NO_GROUP_PREFIX)].sum())
        s = sizes[~sizes.index.str.startswith(NO_GROUP_PREFIX)]
        cum = (s.cumsum() / s.sum()).to_numpy()
        prev = cum - s.to_numpy() / s.sum()
        k = min(top, len(s))
        labs = [glabel(dim, g, multiline=True) for g in s.index[:k]]
        vals = list(s.to_numpy()[:k])
        kinds = ["vital" if p < 0.8 else "remaining" for p in prev[:k]]
        cums = list(cum[:k])
        if len(s) > k:
            labs.append(f"All other ({len(s) - k:,} more)")
            vals.append(int(s.iloc[k:].sum()))
            kinds.append("other")
            cums.append(1.0)
        pareto(labs, vals, cums, kinds=kinds, wrap=12, label_size=9.5 if dim == "VPPS_L3" else 11,
               title=f"Pareto D: Number of ECRs by {DIM_TITLE[dim]}",
               subtitle=f"Level 1: Number of ECRs by {DIM_NOUN[dim]}, all ECRs with {_a(DIM_NOUN[dim])}, top {top}",
               footnote=(f"{len(s):,} {DIM_PLURAL[dim]}; {int(s.sum()):,} ECRs shown; {blank:,} ECRs with no "
                         f"{DIM_NOUN[dim]} not plotted."
                         + (" Assembly names: most frequent FNA_DESC in the assembly's ECRs." if dim == "VPPS_L3" else "")),
               path=out / "pareto_d" / f"D1_ecr_count_by_{dim}.png")
        facts["pareto_d"][dim] = {"groups": int(len(s)), "ecrs": int(s.sum()), "blank": blank,
                                  "groups_for_50pct": int(np.searchsorted(cum, 0.5) + 1),
                                  "groups_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
                                  "top20pct_share": float(cum[max(0, int(np.ceil(len(s) * 0.2)) - 1)]),
                                  "top": [(glabel(dim, g), int(v)) for g, v in s.head(top).items()],
                                  "all_other": int(s.iloc[k:].sum()), "vital_few": int((prev < 0.8).sum())}

    # ---- Pareto C Level 2: words within the top groups -----------------------------------------------------
    for dim, t in gt.items():
        sizes = p1["group_sizes"][dim]
        groups = [g for g in sizes.index if not g.startswith(NO_GROUP_PREFIX)][:top_groups_l2]
        facts["pareto_c_l2"][dim] = {}
        for rank, g in enumerate(groups, start=1):
            for n, name in NGRAMS.items():
                sub = t[(t.group == g) & (t.n == n)].set_index("term")["DF_gw"].sort_values(ascending=False,
                                                                                         kind="stable")
                facts["pareto_c_l2"][dim][f"{g}|{name}"] = _word_pareto(
                    sub, int(sizes[g]), unit=name,
                    title=f"Pareto C: Number of ECRs by {name.capitalize()}, {DIM_TITLE[dim]}",
                    subtitle=f"Level 2: {name.capitalize()}s in {glabel(dim, g)} ({int(sizes[g]):,} ECRs), top {top}",
                    path=out / "pareto_c" / "by_group" / f"C2_{dim}_{rank}_{name}.png", top=top)

    # ---- Level 2 heat table: word mix across top 10 groups; Level 3 lift table ---------------------------------
    for dim, t in gt.items():
        sizes = p1["group_sizes"][dim]
        groups = [g for g in sizes.index if not g.startswith(NO_GROUP_PREFIX)][:10]
        for n, name in ((1, "unigram"), (2, "bigram")):
            top_terms = vocab[vocab.n == n].nlargest(8, ["DF"])["term"].tolist()
            sub = t[(t.n == n) & t.group.isin(groups) & t.term.isin(top_terms)]
            share = sub.pivot_table(index="group", columns="term", values="s_gw").reindex(index=groups,
                                                                                       columns=top_terms).fillna(0)
            share.index = [f"{glabel(dim, g)}" for g in share.index]
            heat_table(share, title=f"Pareto C: Word Mix by {DIM_TITLE[dim]}",
                       subtitle=f"Level 2: Share of each {DIM_NOUN[dim]}'s ECRs using the 8 most frequent "
                                f"{name}s overall, top 10 {DIM_PLURAL[dim]}",
                       footnote=f"Cell = DF_gw / |E_g| (Equation 3). {name.capitalize()}s ranked by number of ECRs "
                                f"across all {N:,} ECRs.",
                       path=out / "heat_tables" / f"L2_word_mix_{name}_by_{dim}.png")
            facts["heat"][f"{dim}|{name}"] = {"terms": top_terms, "groups": share.index.tolist(),
                                              "min": share.min().round(3).to_dict(), "max": share.max().round(3).to_dict()}

            lt = t[(t.n == n) & t.group.isin(groups) & t["lift_L_gw"].notna()]
            lt = lt.sort_values(["group", "lift_L_gw"], ascending=[True, False]).groupby("group").head(5)
            cells, vals, rows = [], [], []
            for g in groups:
                r = lt[lt.group == g]
                rows.append(glabel(dim, g))
                cells.append([f"{w}\n×{v:.1f}" for w, v in zip(r.term, r.lift_L_gw)] + [""] * (5 - len(r)))
                vals.append(list(np.log10(r.lift_L_gw.clip(lower=1))) + [np.nan] * (5 - len(r)))
            vdf = pd.DataFrame(vals, index=rows, columns=[f"Rank {i}" for i in range(1, 6)])
            heat_table(vdf, cell_text=cells, vmax=np.nanmax(vdf.to_numpy()) if np.isfinite(np.nanmax(vdf.to_numpy())) else 1,
                       title=f"Pareto C: Highest-Lift {name.capitalize()}s by {DIM_TITLE[dim]}",
                       subtitle=f"Level 3: Five highest-lift {name}s in each of the top 10 {DIM_PLURAL[dim]}",
                       footnote=f"Lift L_gw = s_gw / s_w (Equation 4); only {name}s in ≥ 5 of the group's ECRs. "
                                "Darker cell = higher lift.",
                       path=out / "heat_tables" / f"L3_lift_{name}_by_{dim}.png")
            facts["lift"][f"{dim}|{name}"] = {glabel(dim, g): [(w, round(float(v), 1), int(d)) for w, v, d in
                                                               zip(lt[lt.group == g].term, lt[lt.group == g].lift_L_gw,
                                                                   lt[lt.group == g].DF_gw)] for g in groups}

    # ---- Ubiquity (Equation 5) -----------------------------------------------------------------------------
    for dim, u in ubq.items():
        uu = u[u.n == 1].head(15)
        n_g = int(u["groups_ranked"].iloc[0]) if len(u) else 0
        hbar_share(uu.term.tolist(), uu.ubiquity_U_w.to_numpy(),
                   title=f"Pareto C: Ubiquity of Words across {DIM_TITLE[dim]}s",
                   subtitle=f"Share of the {n_g} {DIM_PLURAL[dim]} with ≥ 25 ECRs in which the unigram ranks in "
                            "the top 10 (Equation 5), 15 highest",
                   footnote="U_w = 1 means the word is in the top 10 of every ranked group. " + SOURCE,
                   xlabel=f"Ubiquity U_w (share of {DIM_PLURAL[dim]})",
                   path=out / "ubiquity" / f"U_unigram_by_{dim}.png")
        facts["ubiquity"][dim] = {"groups_ranked": n_g,
                                  **{name: u[u.n == k].head(10)[["term", "ubiquity_U_w"]].round(3).values.tolist()
                                     for k, name in NGRAMS.items()}}

    # ---- Step 4 histograms ------------------------------------------------------------------------------
    pe = p1["per_ecr"]
    words = pe["unigram_tokens"].to_numpy()
    histogram(words, bins=np.arange(0, min(words.max(), 150) + 5, 5),
              title="Histogram: Number of Words per ECR",
              subtitle=f"Unigram tokens after cleaning, title + description + reason, all {N:,} ECRs",
              footnote=f"{int((words > 150).sum()):,} ECRs with more than 150 words not plotted. " + SOURCE,
              xlabel="Words per ECR (bins of 5)", path=out / "histograms" / "H1_words_per_ecr.png")
    facts["hist"]["words_per_ecr"] = {"median": float(np.median(words)), "mean": float(words.mean()),
                                      "p90": float(np.percentile(words, 90)), "max": int(words.max()),
                                      "zero": int((words == 0).sum()), "over_150": int((words > 150).sum())}
    panels, fstats = [], {}
    for f in ("TITLE", "DESC_FOR_CHANGE", "REASON_FOR_CHANGE", "COST_COMMENT"):
        has = df[TEXT_COLUMNS[f]].notna().to_numpy()
        v = pe[f"{f}_unigram_tokens"].to_numpy()[has]
        panels.append((f"{FIELD_TITLE[f]} (n={has.sum():,})", np.clip(v, 0, 100)))
        fstats[f] = {"n": int(has.sum()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                     "over_100": int((v > 100).sum())}
    small_hists(panels, bins=np.arange(0, 105, 5), title="Histogram: Number of Words per ECR, by Field",
                subtitle="Unigram tokens after cleaning, ECRs with text in the field",
                footnote="Values above 100 shown in the last bin. " + SOURCE,
                xlabel="Words per ECR (bins of 5)", path=out / "histograms" / "H2_words_per_ecr_by_field.png")
    facts["hist"]["by_field"] = fstats
    facts["hist"]["ecrs_per_group"] = {}
    for dim, sizes in p1["group_sizes"].items():
        s = sizes[~sizes.index.str.startswith(NO_GROUP_PREFIX)].to_numpy()
        bins = np.unique(np.round(np.logspace(0, np.log10(max(s.max(), 2)) + 0.05, 30)))
        histogram(s, bins=bins, logx=True, title=f"Histogram: Number of ECRs per {DIM_TITLE[dim]}",
                  subtitle=f"{len(s):,} {DIM_PLURAL[dim]}; ECRs per {DIM_NOUN[dim]} (log-scale bins)",
                  footnote=SOURCE, xlabel=f"ECRs per {DIM_NOUN[dim]} (log scale)",
                  ylabel=f"Number of {DIM_PLURAL[dim]}", path=out / "histograms" / f"H3_ecrs_per_{dim}.png")
        facts["hist"]["ecrs_per_group"][dim] = {"groups": int(len(s)), "median": float(np.median(s)),
                                                "max": int(s.max()), "single_ecr": int((s == 1).sum()),
                                                "ge_25": int((s >= 25).sum())}
        col = {"Program": "DIM_PROGRAM", "Supplier": "DIM_SUPPLIER", "VPPS_L1": "DIM_VPPS_L1",
               "VPPS_L2": "DIM_VPPS_L2", "VPPS_L3": "DIM_VPPS_L3"}[dim]
        groups = [g for g in sizes.index if not g.startswith(NO_GROUP_PREFIX)][:top_groups_l2]
        gp = [(f"{textwrap.shorten(glabel(dim, g), 30, placeholder='…')} (n={int(sizes[g]):,})",
               np.clip(pe.loc[pe[col] == g, "unigram_tokens"].to_numpy(), 0, 150)) for g in groups]
        small_hists(gp, bins=np.arange(0, 155, 5), title=f"Histogram: Number of Words per ECR, by {DIM_TITLE[dim]}",
                    subtitle=f"Unigram tokens after cleaning in the {top_groups_l2} largest {DIM_PLURAL[dim]}",
                    footnote="Values above 150 shown in the last bin. " + SOURCE, xlabel="Words per ECR",
                    path=out / "histograms" / f"H4_words_per_ecr_by_{dim}.png")
    series = {name: vocab.loc[vocab.n == n, "DF"].sort_values(ascending=False).to_numpy()
              for n, name in NGRAMS.items()}
    rank_frequency(series, title="Rank-Frequency Plot of Terms",
                   subtitle=f"Number of ECRs containing each term vs. its rank, all {N:,} ECRs (log-log)",
                   footnote="A straight line on log-log axes is the pattern described by Zipf's law. " + SOURCE,
                   path=out / "histograms" / "H5_rank_frequency.png")
    dfu = vocab.loc[vocab.n == 1, "DF"].to_numpy()
    histogram(dfu, bins=np.unique(np.round(np.logspace(0, np.log10(dfu.max()) + 0.05, 35))), logx=True,
              title="Histogram: Number of ECRs per Unigram",
              subtitle=f"Document frequency DF_w of each of the {len(dfu):,} distinct unigrams (log-scale bins)",
              footnote=f"Step 5 keeps terms with DF_w ≥ {p1['vocab_summary']['min_df']}. " + SOURCE,
              xlabel="ECRs containing the unigram (log scale)", ylabel="Number of unigrams",
              path=out / "histograms" / "H6_document_frequency_unigrams.png")
    facts["hist"]["df_unigrams"] = {"distinct": int(len(dfu)), "df_1": int((dfu == 1).sum()),
                                    "df_lt_5": int((dfu < 5).sum()), "median": float(np.median(dfu))}
    log.info("Phase 1 figures written to %s", out)
    return facts
