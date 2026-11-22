"""Slide-ready Pareto charts in the mid-term Pareto A / B format (navy bars, orange cumulative line).

Reads the tables written by pareto_cost.py. Usage: python -m cost_pareto.charts <out_dir>
Writes <out_dir>/charts/<metric folder>/*.png
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

NAVY = "#12245C"       # deck header navy
NAVY_TINT = "#A7B1CC"  # bars beyond the vital few
LINE = "#D9822B"       # cumulative line
GRID = "#E6E9EF"
TEXT = "#1F2937"
MUTED = "#6B7280"

# Carlito (metric-compatible with Calibri) if installed; otherwise the Lato font shipped in resources/.
for _ttf in (Path(__file__).resolve().parents[1] / "resources" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(_ttf))
_FONTS = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Carlito", "Calibri", "Lato") if f in _FONTS), "DejaVu Sans")

plt.rcParams.update({
    "font.family": FONT, "font.size": 11, "text.color": TEXT,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": MUTED,
    "axes.edgecolor": "#C9CED8", "figure.dpi": 100, "savefig.dpi": 220,
})

METRIC = {
    "count": {"col": "ecr_count", "folder": "01_ecr_count", "what": "Number of ECRs", "ylabel": "Number of ECRs", "tag": ""},
    "cost_given": {"col": "cost_given", "folder": "02_cost_usd_as_given", "what": "COST_USD", "ylabel": "COST_USD", "tag": " (COST_USD as given)"},
    "cost_recalc": {"col": "cost_recalc", "folder": "03_cost_usd_recalculated", "what": "COST_USD", "ylabel": "COST_USD", "tag": " (recalculated: COST × EXCHANGE_RATE)"},
}


def money(v: float) -> str:
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{s}${a / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{s}${a / 1e6:,.1f}M"
    if a >= 1e3:
        return f"{s}${a / 1e3:,.0f}K"
    return f"{s}${a:,.0f}"


def fmt(v: float, metric: str) -> str:
    return f"{int(round(v)):,}" if metric == "count" else money(v)


def wrap(s: str, width: int = 15, lines: int = 3) -> str:
    out = textwrap.wrap(str(s), width) or [""]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: width - 1] + "…"
    return "\n".join(out)


NAME_NOTE = ("Names from FNA_DESC: most common description when it is on >50% of the code's ECRs (at least 2); "
             "\"(various)\" = shared leading term on >50%; * = description on the code's only ECR; no name = none meets the rule")
NAMES: dict[tuple[str, str], str] = {}
LEVEL_OF = {"A1_assembly": "assembly", "A2_vpps_code": "vpps_code", "A3_part_number": "part_number", "A3_new_part_number": "new_part_number",
            "N1_part_number": "part_number", "N2_new_part_number": "new_part_number",
            "assembly": "assembly", "VPPS": "vpps_code", "PART_NUMBER": "part_number"}


def load_names(path: Path) -> None:
    n = pd.read_csv(path, dtype={"key": str})
    n = n[n["tier"] != "none"]
    for lv, k, nm, tier in zip(n["level"], n["key"], n["name"], n["tier"]):
        NAMES[(lv, k)] = f"{nm}*" if tier == "single" else nm


def name_of(level: str | None, key: str) -> str:
    return NAMES.get((level, str(key)), "") if level else ""


def with_name(level: str | None, key: str, limit: int = 46) -> str:
    """'80.06.01 (HARNESS (various))' style text for chart titles."""
    nm = name_of(level, key)
    if len(nm) > limit:
        nm = nm[: limit - 1].rstrip() + "…" + ("*" if nm.endswith("*") else "")
    return f"{key} ({nm})" if nm else str(key)


def header(fig, title: str, level: str) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=17, fontweight="bold", color=NAVY)
    fig.text(0.012, 0.925, level, ha="left", va="top", fontsize=13, color=TEXT)


def pareto_png(t: pd.DataFrame, metric: str, path: Path, title: str, level: str, n_groups: int, total: float, note: str,
               name_level: str | None = None, names_note: bool = False) -> None:
    """t: top rows of a Pareto table (key, value column, cum_pct, vital_few), already ranked.

    name_level: when the x axis holds assemblies, VPPS codes or part numbers, their reliable names are added
    under the code. names_note: print the naming rule under the chart.
    """
    col = METRIC[metric]["col"]
    t = t.sort_values("rank").reset_index(drop=True)
    if name_level:
        labels = [f"{k}\n{wrap(name_of(name_level, k), 13, 5)}" if name_of(name_level, k) else str(k) for k in t["key"]]
    else:
        labels = [wrap(k) for k in t["key"]]
    vals = t[col].astype(float).tolist()
    cum = t["cum_pct"].tolist()
    colors = [NAVY if v else NAVY_TINT for v in t["vital_few"]]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(12, 6.6))
    x = list(range(n))
    bars = ax.bar(x, vals, color=colors, width=0.68, zorder=2)
    ax.set_xticks(x, labels, fontsize=9.2 if name_level else 10.5)
    ax.tick_params(axis="x", length=0, pad=6)
    for lab in ax.get_xticklabels():  # keep codes on one line whether or not a name follows
        lab.set_verticalalignment("top")
        lab.set_multialignment("center")
    ymax = max(max(vals), 1e-9)
    ymin = min(min(vals), 0.0)
    if metric == "count":
        ax.set_ylabel(METRIC[metric]["ylabel"], fontsize=11.5, fontweight="bold", color=NAVY)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        unit, div = ("$M", 1e6) if ymax >= 1e6 else (("$K", 1e3) if ymax >= 1e3 else ("$", 1.0))
        ax.set_ylabel(f"{METRIC[metric]['ylabel']} ({unit})", fontsize=11.5, fontweight="bold", color=NAVY)
        dec = 0 if ymax / div >= 10 else 1
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / div:,.{dec}f}"))
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(ymin * 1.14 if ymin < 0 else 0, ymax * 1.14)
    ax2 = ax.twinx()
    # Value labels are drawn on the top axes (with a white backing) so the cumulative line
    # never runs through them.
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, max(v, 0) + ymax * 0.012, fmt(v, metric), transform=ax.transData,
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
    ax2.plot(x, cum, color=LINE, lw=2.4, marker="o", ms=5.5, zorder=3, solid_capstyle="round")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_ylabel("Cumulative %", fontsize=11.5, fontweight="bold", color=LINE)
    ax2.tick_params(axis="y", colors=LINE)
    for s in ("top", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color("#C9CED8")
    ax2.axhline(0.8, color=MUTED, lw=1.1, ls=(0, (5, 4)), zorder=1)
    ax2.text(0.99, 0.81, "80%", transform=ax2.get_yaxis_transform(), ha="right", va="bottom", fontsize=9.5, color=MUTED)
    top_frac = 1.14  # y-axis headroom factor
    for i, c in enumerate(cum):
        # Keep the cumulative label clear of the bar's value label (both in axes fractions).
        p = c / 1.05
        b = (max(vals[i], 0) / (ymax * top_frac)) if ymax else 0.0
        bar_lab = (b + 0.004, b + 0.052)
        above, below = (p + 0.012, p + 0.052), (p - 0.058, p - 0.016)
        hit = lambda r: r[0] < bar_lab[1] and bar_lab[0] < r[1]
        under = hit(above) and not hit(below) and below[0] > 0.0
        lift = hit(above) and not under  # point sits on the value label: go above it
        dy = -15 if under else (7 + (bar_lab[1] - p + 0.012) * 380 if lift else 7)
        ax2.annotate(f"{c:.0%}", (i, c), textcoords="offset points", xytext=(0, dy),
                     ha="center", fontsize=8.5, color=LINE, fontweight="bold", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    handles = [Patch(color=NAVY, label="Vital few (to 80%)")]
    if not all(t["vital_few"]):
        handles.append(Patch(color=NAVY_TINT, label="Remaining"))
    handles.append(Line2D([], [], color=LINE, lw=2.4, marker="o", label="Cumulative %"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(handles),
              frameon=False, fontsize=10, handlelength=1.4, columnspacing=1.6)
    half = max(n, 5) / 2
    ax.set_xlim((n - 1) / 2 - half - 0.1, (n - 1) / 2 + half + 0.1)
    header(fig, title, level)
    shown = float(sum(vals))
    rest = total - shown
    foot = (f"Top {n} of {n_groups:,} shown; remaining {fmt(rest, metric)} not plotted" if n_groups > n else f"All {n_groups:,} shown")
    if names_note or name_level:
        fig.text(0.99, 0.030, f"{foot} · {note}", ha="right", va="bottom", fontsize=9, color=MUTED)
        fig.text(0.99, 0.006, NAME_NOTE, ha="right", va="bottom", fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, 0.05, 1, 0.875))
    else:
        fig.text(0.99, 0.008, f"{foot} · {note}", ha="right", va="bottom", fontsize=9, color=MUTED)
        fig.tight_layout(rect=(0, 0.025, 1, 0.875))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def heatmap_png(h: pd.DataFrame, metric: str, path: Path, title: str, level: str, row_note: dict, note: str) -> None:
    piv = h.pivot(index="assembly", columns="reason_grp", values="share")
    order_r = h.drop_duplicates("assembly").sort_values("assembly_rank")["assembly"].tolist()
    order_c = list(dict.fromkeys(h["reason_grp"]))
    piv = piv.reindex(index=order_r, columns=order_c)
    cmap = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", NAVY])
    fig, ax = plt.subplots(figsize=(12, 6.6))
    data = piv.to_numpy(dtype=float)
    ax.imshow(np.clip(data, 0, 1), cmap=cmap, vmin=0, vmax=max(0.6, float(np.nanmax(np.clip(data, 0, 1)))), aspect="auto")
    ax.set_xticks(range(len(order_c)), [wrap(c, 16, 3) for c in order_c], fontsize=9.5)
    rows = [f"#{i + 1}  {a}  ({row_note[a]})" + (f"\n{name_of('assembly', a)}" if name_of("assembly", a) else "") for i, a in enumerate(order_r)]
    ax.set_yticks(range(len(order_r)), rows, fontsize=10)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.33 else TEXT, fontweight="bold" if v >= 0.2 else "normal")
    ax.set_xticks(np.arange(-0.5, data.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    header(fig, title, level)
    fig.text(0.99, 0.030, note, ha="right", va="bottom", fontsize=9, color=MUTED)
    fig.text(0.99, 0.006, NAME_NOTE, ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.885))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def slug(s: str) -> str:
    return str(s).replace(".", "-").replace(" ", "_")


def main(out: Path) -> None:
    T = out / "tables"
    S = json.loads((out / "summary.json").read_text())
    L = pd.read_csv(T / "level_tables.csv", dtype={"key": str})
    AD = pd.read_csv(T / "a_drill.csv", dtype={"key": str, "parent_key": str})
    BD = pd.read_csv(T / "b_drill.csv", dtype={"key": str, "parent_key": str})
    H = pd.read_csv(T / "heatmap.csv")
    sc = S["scope"]
    load_names(T / "names.csv")
    n_made = 0

    for m, M in METRIC.items():
        d = out / "charts" / M["folder"]
        what, col = M["what"], M["col"]
        pa = (f"Pareto A: Number of ECRs by Assembly, VPPS Code and Part Number" if m == "count"
              else f"Pareto A: COST_USD by Assembly, VPPS Code and Part Number{M['tag']}")
        pb = f"Pareto B: Number of ECRs by Reason Code" if m == "count" else f"Pareto B: COST_USD by Reason Code{M['tag']}"
        pn = (f"ECRs with No VPPS Assembly: Number of ECRs" if m == "count" else f"ECRs with No VPPS Assembly: COST_USD{M['tag']}")
        asm_note = f"all 7 states · {sc['with_assembly']:,} ECRs with an assembly ({sc['without_assembly']:,} without one excluded)"

        def lvl(rep):
            t = L[(L["report"] == rep) & (L["metric"] == m)]
            return t, len(t), float(t[col].sum())

        specs = [
            ("A1_assembly", "A1", pa, f"Level 1: {what} by assembly (VPPS first 6 digits), top 10", asm_note),
            ("A2_vpps_code", "A2_00", pa, f"Level 2: {what} by VPPS code, all assemblies, top 10", asm_note),
            ("A3_part_number", "A3_00", pa, f"Level 3: {what} by PART_NUMBER, all assemblies, top 10", asm_note),
            ("A3_new_part_number", "A3N_00", pa, f"Level 3: {what} by NEW_PART_NUMBER, all assemblies, top 10", asm_note),
            ("B1_reason", "B1", pb, f"Level 1: {what} by reason code, all ECRs, top 10", f"all 7 states · {sc['ecrs']:,} ECRs"),
        ]
        names = {"A1": "by_assembly", "A2_00": "by_vpps_code_all_assemblies", "A3_00": "by_part_number_all_assemblies",
                 "A3N_00": "by_new_part_number_all_assemblies", "B1": "by_reason_code"}
        for rep, code, title, level, note in specs:
            t, n, tot = lvl(rep)
            pareto_png(t.head(10), m, d / f"{code}_{names[code]}.png", title, level, n, tot, note, name_level=LEVEL_OF.get(rep))
            n_made += 1

        a = AD[AD["metric"] == m]
        for (rep, pk), t in a.groupby(["report", "parent_key"], sort=False):
            r = int(t["parent_rank"].iloc[0])
            code, lab = {"A2_vpps_code": ("A2", "VPPS code"), "A3_part_number": ("A3", "PART_NUMBER"), "A3_new_part_number": ("A3N", "NEW_PART_NUMBER")}[rep]
            lv = "Level 2" if code == "A2" else "Level 3"
            pareto_png(t, m, d / f"{code}_{r:02d}_{lab.lower()}_in_assembly_{slug(pk)}.png".replace(" ", "_"), pa,
                       f"{lv}: {what} by {lab} within assembly {with_name('assembly', pk)}, #{r} assembly", int(t["n_groups"].iloc[0]),
                       float(t["level_total"].iloc[0]),
                       f"all 7 states · {int(t['parent_ecrs'].iloc[0]):,} ECRs in assembly {pk}", name_level=LEVEL_OF[rep])
            n_made += 1

        b = BD[BD["metric"] == m]
        for (rep, pk), t in b.groupby(["report", "parent_key"], sort=False):
            if rep == "B4_reason_in_part_number" and m != "count":
                continue  # the top parts by cost sit on 1-7 ECRs; reported as a table instead
            r = int(t["parent_rank"].iloc[0])
            code, lab, lv = {"B2_reason_in_assembly": ("B2", "assembly", "Level 2"), "B3_reason_in_vpps_code": ("B3", "VPPS code", "Level 3"),
                             "B4_reason_in_part_number": ("B4", "PART_NUMBER", "Level 3")}[rep]
            pareto_png(t, m, d / f"{code}_{r:02d}_reasons_for_{lab.lower().replace(' ', '_')}_{slug(pk)}.png", pb,
                       f"{lv}: {what} by reason code for {lab} {with_name(LEVEL_OF['VPPS' if lab == 'VPPS code' else lab], pk)}, #{r} {lab if lab != 'PART_NUMBER' else 'part number'}",
                       int(t["n_groups"].iloc[0]), float(t["level_total"].iloc[0]),
                       f"all 7 states · {int(t['parent_ecrs'].iloc[0]):,} ECR{'s' if int(t['parent_ecrs'].iloc[0]) != 1 else ''} {'in' if lab == 'assembly' else 'on'} {lab} {pk}",
                       names_note=True)
            n_made += 1

        h = H[H["metric"] == m]
        a1 = L[(L["report"] == "A1_assembly") & (L["metric"] == m)].set_index("key")
        row_note = {k: (f"{int(a1.loc[k, 'ecr_count']):,} ECRs" if m == "count" else money(float(a1.loc[k, col]))) for k in h["assembly"].unique()}
        cell = "share of each assembly's ECRs" if m == "count" else "share of each assembly's COST_USD"
        heatmap_png(h, m, d / "B2_00_reason_mix_heatmap_top10_assemblies.png", pb,
                    "Level 2: Reason-code mix across the top 10 assemblies", row_note,
                    f"Cells: {cell} · columns: top 8 reason codes overall, then all other codes · all 7 states")
        n_made += 1

        dn = out / "charts" / "04_no_vpps_assembly"
        no_note = f"all 7 states · {sc['without_assembly']:,} ECRs with no VPPS assembly"
        for rep, code, lab in (("N1_part_number", "N1", "PART_NUMBER"), ("N2_new_part_number", "N2", "NEW_PART_NUMBER"), ("N3_reason", "N3", "reason code")):
            t, n, tot = lvl(rep)
            pareto_png(t.head(10), m, dn / f"{code}_{M['folder'][3:]}_by_{lab.lower().replace(' ', '_')}.png", pn,
                       f"{what} by {lab}, ECRs with no VPPS assembly, top 10", n, tot, no_note, name_level=LEVEL_OF.get(rep))
            n_made += 1
    print("charts written:", n_made)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
