"""Write the Phase 1 write-up (Steps 1-8) from the run outputs.

Every number is computed from the outputs of ``run_phase1`` and ``make_phase1_figures``; the
literature notes describe published methods and are the only text not computed from the data.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .config import RESOURCE_DIR, TEXT_COLUMNS

DIM_TXT = {"Program": "lead programs", "Supplier": "suppliers", "VPPS_L1": "VPPS L1 codes",
           "VPPS_L2": "VPPS L2 codes", "VPPS_L3": "assemblies (VPPS L3)"}
DIM_ONE = {"Program": "lead program", "Supplier": "supplier", "VPPS_L1": "VPPS L1 code",
           "VPPS_L2": "VPPS L2 code", "VPPS_L3": "assembly (VPPS L3)"}
FIELD_TXT = {"TITLE": "Title", "DESC_FOR_CHANGE": "Description for change", "REASON_FOR_CHANGE": "Reason for change",
             "COST_COMMENT": "Cost comment"}
from .nlp import KEEP_WORDS

REFS = [
    ("MRS", 'C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge, U.K.: '
            'Cambridge Univ. Press, 2008.'),
    ("SPACY", 'M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd, "spaCy: Industrial-strength natural language '
              'processing in Python," Zenodo, 2020, doi: 10.5281/zenodo.1212303.'),
    ("DS", 'M. J. Denny and A. Spirling, "Text preprocessing for unsupervised learning: Why it matters, when it '
           'misleads, and what to do about it," *Political Analysis*, vol. 26, no. 2, pp. 168–189, 2018.'),
    ("LUHN58", 'H. P. Luhn, "The automatic creation of literature abstracts," *IBM J. Res. Develop.*, vol. 2, no. 2, '
               'pp. 159–165, 1958.'),
    ("SCH", 'A. Schofield, M. Magnusson, and D. Mimno, "Pulling out the stops: Rethinking stopword removal for topic '
            'models," in *Proc. 15th Conf. EACL*, vol. 2, 2017, pp. 432–436.'),
    ("GS", 'J. Grimmer and B. M. Stewart, "Text as data: The promise and pitfalls of automatic content analysis methods '
           'for political texts," *Political Analysis*, vol. 21, no. 3, pp. 267–297, 2013.'),
    ("ZIPF", 'G. K. Zipf, *Human Behavior and the Principle of Least Effort*. Cambridge, MA, USA: Addison-Wesley, 1949.'),
    ("SR", 'J. Silge and D. Robinson, *Text Mining with R: A Tidy Approach*. Sebastopol, CA, USA: O\'Reilly, 2017.'),
    ("SKL", 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, '
            'pp. 2825–2830, 2011.'),
    ("LUHN57", 'H. P. Luhn, "A statistical approach to mechanized encoding and searching of literary information," '
               '*IBM J. Res. Develop.*, vol. 1, no. 4, pp. 309–317, 1957.'),
    ("SWY", 'G. Salton, A. Wong, and C. S. Yang, "A vector space model for automatic indexing," *Commun. ACM*, '
            'vol. 18, no. 11, pp. 613–620, 1975.'),
    ("KSJ", 'K. Spärck Jones, "A statistical interpretation of term specificity and its application in retrieval," '
            '*J. Documentation*, vol. 28, no. 1, pp. 11–21, 1972.'),
    ("SB", 'G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Inf. Process. Manage.*, '
           'vol. 24, no. 5, pp. 513–523, 1988.'),
    ("LSA", 'S. Deerwester, S. T. Dumais, G. W. Furnas, T. K. Landauer, and R. Harshman, "Indexing by latent semantic '
            'analysis," *J. Amer. Soc. Inf. Sci.*, vol. 41, no. 6, pp. 391–407, 1990.'),
    ("GRS", 'J. Grimmer, M. E. Roberts, and B. M. Stewart, *Text as Data: A New Framework for Machine Learning and the '
            'Social Sciences*. Princeton, NJ, USA: Princeton Univ. Press, 2022.'),
    ("ARN", 'Í. Ö. Arnarsson, O. Frost, E. Gustavsson, M. Jirstrand, and J. Malmqvist, "Natural language processing '
            'methods for knowledge management—Applying document clustering for fast search and grouping of engineering '
            'documents," *Concurrent Eng.*, vol. 29, no. 2, pp. 142–152, 2021, doi: 10.1177/1063293X20982973.'),
    ("LDA", 'D. M. Blei, A. Y. Ng, and M. I. Jordan, "Latent Dirichlet allocation," *J. Mach. Learn. Res.*, vol. 3, '
            'pp. 993–1022, 2003.'),
    ("STM", 'M. E. Roberts *et al.*, "Structural topic models for open-ended survey responses," *Amer. J. Political '
            'Sci.*, vol. 58, no. 4, pp. 1064–1082, 2014.'),
    ("NMF", 'D. D. Lee and H. S. Seung, "Learning the parts of objects by non-negative matrix factorization," '
            '*Nature*, vol. 401, pp. 788–791, 1999.'),
    ("TEA", 'J. Chang, J. Boyd-Graber, S. Gerrish, C. Wang, and D. M. Blei, "Reading tea leaves: How humans interpret '
            'topic models," in *Adv. Neural Inf. Process. Syst. 22*, 2009, pp. 288–296.'),
    ("MCQ", 'B. L. Monroe, M. P. Colaresi, and K. M. Quinn, "Fightin\' words: Lexical feature selection and evaluation '
            'for identifying the content of political conflict," *Political Analysis*, vol. 16, no. 4, pp. 372–403, 2008.'),
]
R = {k: i + 1 for i, (k, _) in enumerate(REFS)}


def c(*keys) -> str:
    return "[" + ", ".join(str(R[k]) for k in keys) + "]"


def n(x) -> str:
    return f"{int(round(x)):,}"


def pct(x, d=1) -> str:
    return f"{x * 100:.{d}f}%"


def table(header, rows, align=None) -> str:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


class Fig:
    def __init__(self):
        self.k = 0

    def __call__(self, path: str, caption: str) -> str:
        self.k += 1
        return f"![Figure {self.k}](figures/{path})\n\n*Figure {self.k}: {caption} Source: team analysis of the GM masked ECR extract.*\n"


def write_report(p1: dict, facts: dict, phase_dir: Path, run_meta: dict) -> Path:
    phase_dir = Path(phase_dir)
    df, N = p1["df"], p1["n_ecr"]
    vs, ts = p1["vocab_summary"], p1["tf_summary"]
    s1 = phase_dir / "step1_lemmatize"
    s2 = phase_dir / "step2_stopwords"
    lem = pd.read_csv(s1 / "lemma_merges_by_lemma.csv")
    lem_pairs = pd.read_csv(s1 / "lemma_merges.csv")
    rem = pd.read_csv(s2 / "stopword_removals.csv")
    chk = pd.read_csv(s2 / "top_lemmas_vs_stoplist.csv")
    n_dom = sum(1 for ln in (RESOURCE_DIR / "stopwords_domain.txt").read_text().splitlines() if ln.split("#")[0].strip())
    n_abbr = len(pd.read_csv(RESOURCE_DIR / "abbreviations.csv"))
    n_acr = len(pd.read_csv(RESOURCE_DIR / "acronym_glossary.csv"))
    rem_agg = rem.groupby("reason")["times_removed"].agg(["sum", "count"])
    states = df["STATE"].astype(str).value_counts()
    fig = Fig()
    L: list[str] = []
    A = L.append

    def l1(field, name):
        return facts["pareto_c_l1"][f"{field}|{name}"]

    # -------- header ------------------------------------------------------------------------------
    A("# Phase 1 Text Frequency Model: Steps 1–8")
    A("**IOE 424 · GM Case Study No. 6 · Cost Commandos Consulting**  ")
    A(f"Prepared {dt.date.today():%B %-d, %Y} · Data: `Masking_Lookup_Reference 1.xlsx` (sheet `result`)\n")
    A("This document follows the Phase 1 formulation in the mid-term report (Equations 1–5) and the figure format "
      "of the mid-term slides (Pareto A/B). It reports Steps 1–6 as run on the data, and Steps 7–8 as procedures "
      "taken from the text-analysis literature, because the 11/2 meeting notes put any new reason framework after "
      "GM reviews the Phase 1 results. Results state measured facts only; literature notes describe published "
      "methods.\n")

    # -------- summary table -------------------------------------------------------------------------
    A("## Summary\n")
    lu = l1("ALL", "unigram")
    A(table(["Step", "Status", "What was produced", "Key measured result"], [
        ["1. Lemmatize", "Run", "`step1_lemmatize/`",
         f"{n(lem_pairs['occurrences'].sum())} kept token occurrences changed form, into {n(len(lem))} lemmas"],
        ["2. Remove stop words", "Run", "`step2_stopwords/`",
         f"{n(rem_agg['sum'].sum())} tokens removed (English list, domain list, digits, 1-character)"],
        ["3. Count words per ECR / group", "Run", "`step3_counts/`, Pareto C and D figures",
         f"Most frequent unigram: {lu['top'][0][0]} ({n(lu['top'][0][1])} ECRs, {pct(lu['top'][0][1] / N)})"],
        ["4. Histograms", "Run", "`figures/histograms/`",
         f"Median {facts['hist']['words_per_ecr']['median']:.0f} words per ECR"],
        ["5. Vocabulary", "Run", "`step5_vocabulary/`",
         f"{n(vs['vocabulary_size'])} terms with DF ≥ {vs['min_df']} ({n(vs['unigram']['in_vocabulary'])} unigrams, "
         f"{n(vs['bigram']['in_vocabulary'])} bigrams, {n(vs['trigram']['in_vocabulary'])} trigrams)"],
        ["6. Term frequency per ECR", "Run", "`step6_term_frequency/`",
         f"{n(ts['ecrs'])} × {n(ts['vocabulary_terms'])} matrix, {n(ts['nonzero_cells'])} non-zero cells"],
        ["7. Vectorize", "Procedure", "§7 (inputs from Step 6 are ready)", "–"],
        ["8. Analyze / propose reason codes", "Procedure", "§8 (after GM review)", "–"],
    ]))
    A("")

    # -------- data ----------------------------------------------------------------------------------
    A("## Data and scope\n")
    A(f"- **ECRs:** all {n(N)} rows of the masked extract (one per WO_NO), every STATE: "
      + ", ".join(f"{k} {n(v)}" for k, v in states.items()) + ".")
    A("- **Document d_i (mid-term Phase 1):** masked title + description for change + reason for change of ECR i, "
      "cleaned in Steps 1–2. Each field stays a separate segment, so bigrams and trigrams never join two fields.")
    A(f"- **Coverage:** title is filled for {n(df[TEXT_COLUMNS['TITLE']].notna().sum())} ECRs; description for "
      f"{n(df[TEXT_COLUMNS['DESC_FOR_CHANGE']].notna().sum())} and reason for "
      f"{n(df[TEXT_COLUMNS['REASON_FOR_CHANGE']].notna().sum())}, all with WO_NO ≥ 7,000,105 (both fields are blank "
      "for every WO in 4,131,983–4,800,660). Cost comment ("
      f"{n(df[TEXT_COLUMNS['COST_COMMENT']].notna().sum())} ECRs) is reported separately, as GM ranked it lower priority.")
    A("- **Groups:** lead program (`MASKED_LEAD_PROGRAM`; \"Program 1\" = blank lead program), supplier "
      "(`MASKED_SUPPLIER_NAME`), VPPS L1/L2/L3 (first one, two or three segments of `VPPS`; L3 = assembly, as in "
      "mid-term Pareto A). No SMT column exists in the extract.\n")

    # -------- step 1 -----------------------------------------------------------------------------------
    A("## Step 1: Lemmatize words\n")
    A(f"**Literature.** Lemmatization maps inflected forms to their dictionary form (\"failed\" → fail); stemming "
      f"instead cuts suffixes and can produce non-words {c('MRS')}. Preprocessing choices such as this one can change "
      f"the results of unsupervised text analysis, so each choice should be stated and checked {c('DS')}.\n")
    A(f"**Procedure.** spaCy `en_core_web_sm` POS-based lemmatizer {c('SPACY')} on each segment, after removing GM "
      f"mask tokens and leftover identifiers, lowercasing and dropping tokens that contain a digit. Abbreviations are "
      f"not expanded ({n_abbr} candidate expansions wait for GM confirmation). The {n_acr} acronyms in "
      "`resources/acronym_glossary.csv` are kept verbatim.\n")
    top_lem = lem.head(10)
    A(f"**Results.** {n(lem_pairs['occurrences'].sum())} kept token occurrences (counted over unique texts) were changed "
      f"by the lemmatizer, into {n(lem['lemma'].nunique())} lemmas. The 10 lemmas that absorbed the most occurrences:\n")
    A(table(["Lemma", "Surface forms merged into it", "Occurrences changed"],
            [[r.lemma, r.surface_forms if len(str(r.surface_forms)) < 70 else str(r.surface_forms)[:67] + "…",
              n(r.occurrences)] for r in top_lem.itertuples()], ["---", "---", "---:"]))
    A("\n**Outputs:** `step1_lemmatize/lemma_merges.csv` (every surface form → lemma pair), "
      "`lemma_merges_by_lemma.csv`, `sample_before_after.csv` (40 ECRs, original vs. cleaned, for reading checks).\n")

    # -------- step 2 -----------------------------------------------------------------------------------
    A("## Step 2: Remove stop words\n")
    A(f"**Literature.** A stop list is usually built by sorting terms by collection frequency and then hand-filtering "
      f"the most frequent ones for semantic content {c('MRS')}. Luhn proposed that words above an upper frequency "
      f"cut-off and below a lower one carry little significance {c('LUHN58')}. For topic models, removing more than a "
      f"standard list mainly changes readability, and corpus-specific frequent words can be removed after modeling "
      f"{c('SCH')}; the effect of each removal should be checked {c('DS')}.\n")
    A(f"**Procedure.** spaCy's English stop list minus {len(KEEP_WORDS)} kept words "
      f"({', '.join(sorted(KEEP_WORDS))}), plus the {n_dom} domain words in `resources/stopwords_domain.txt`. "
      "As the mid-term report specifies, the stop list was compared with the most frequent lemmas before removal "
      "(`top_lemmas_vs_stoplist.csv`).\n")
    rows = []
    for r in rem_agg.itertuples():
        top = rem[rem.reason == r.Index].nlargest(5, "times_removed")
        rows.append([r.Index.replace("_", " "), n(r.sum), n(r.count) if r.Index != "contains_digit" else "–",
                     ", ".join(f"{w} ({n(t)})" for w, t in zip(top.word, top.times_removed)) if r.Index != "contains_digit" else "–"])
    A(table(["Removal rule", "Tokens removed", "Distinct words", "Most removed"], rows, ["---", "---:", "---:", "---"]))
    A("\n**The 15 most frequent lemmas before removal and their status:**\n")
    A(table(["Rank", "Lemma", "Occurrences", "Status"],
            [[r.rank, r.lemma, n(r.occurrences), r.status.replace("_", " ")] for r in chk.head(15).itertuples()],
            ["---:", "---", "---:", "---"]))
    kept_top = chk[chk.status == "kept"].head(10)["lemma"].tolist()
    A(f"\nThe first 10 kept lemmas in that ranking are: {', '.join(kept_top)}.\n")
    A("**Outputs:** `step2_stopwords/stopword_removals.csv`, `top_lemmas_vs_stoplist.csv` (top 200).\n")

    # -------- step 3 ----------------------------------------------------------------------------------
    A("## Step 3: Count words per ECR, program, supplier and VPPS\n")
    A(f"**Literature.** Representing each document by its word counts (bag of words) and stacking the documents into "
      f"a document-term matrix is the standard first step of automated text analysis {c('GS', 'MRS')}. Document "
      f"frequency (number of documents containing a term) differs from collection frequency (total occurrences) "
      f"{c('MRS')}; the mid-term formulation uses document frequency so that long ECRs do not dominate.\n")
    A(f"**Procedure.** scikit-learn `CountVectorizer` {c('SKL')} with a segment-aware analyzer (unigrams, bigrams, "
      "trigrams) builds f_iw for every ECR. From it: DF_w and s_w (Eq. 1), the DF-sorted cumulative share S_m "
      "(Eq. 2, each ECR counted once per term), group shares s_gw (Eq. 3), lift L_gw = s_gw / s_w (Eq. 4, reported where DF_gw ≥ 5) and ubiquity "
      "U_w with N = 10 (Eq. 5, over groups with ≥ 25 ECRs).\n")
    A("### 3a. Words per ECR (Pareto C, Level 1)\n")
    for name in ("unigram", "bigram", "trigram"):
        f = l1("ALL", name)
        A(fig(f"pareto_c/C1_{name}_all_text.png",
              f"Number of ECRs by {name}, title + description + reason, top 10 of {n(f['distinct'])} {name}s. The top 10 "
              f"cover {pct(f['top_cum'])} of all ECR-{name} mentions; {n(f['terms_for_80pct'])} {name}s are needed to "
              f"reach 80%."))
    rows = []
    for fld in ("TITLE", "DESC_FOR_CHANGE", "REASON_FOR_CHANGE", "COST_COMMENT", "ALL"):
        for name in ("unigram", "bigram", "trigram"):
            f = l1(fld, name)
            rows.append([fld.replace("ALL", "ALL TEXT"), name, n(f["n_docs"]), n(f["distinct"]),
                         ", ".join(f"{t} ({n(v)})" for t, v in f["top"][:5]), pct(f["top_cum"]), n(f["terms_for_80pct"])])
    A(table(["Field", "n-gram", "ECRs with text", "Distinct terms", "Top 5 (ECRs)", "Top-10 cum. %", "Terms for 80%"],
            rows, ["---", "---", "---:", "---:", "---", "---:", "---:"]))
    A("\nField-level charts: `figures/pareto_c/by_field/` (12 charts).\n")

    A("### 3b. ECRs per group (Pareto D, Level 1)\n")
    rows = []
    for dim, d in facts["pareto_d"].items():
        rows.append([dim.replace("_", " "), n(d["groups"]), n(d["ecrs"]), n(d["blank"]), n(d["groups_for_50pct"]),
                     n(d["groups_for_80pct"]), pct(d["top20pct_share"])])
    A(table(["Dimension", "Groups", "ECRs with a group", "ECRs with none", "Groups for 50%", "Groups for 80%",
             "ECR share of top 20% of groups"], rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:"]))
    A("")
    for dim in ("Program", "VPPS_L3"):
        d = facts["pareto_d"][dim]
        A(fig(f"pareto_d/D1_ecr_count_by_{dim}.png",
              f"Number of ECRs by {DIM_ONE[dim]}, top 10 of {n(d['groups'])}. "
              f"The top 10 hold {pct(sum(v for _, v in d['top']) / d['ecrs'])} of {n(d['ecrs'])} ECRs; "
              f"{n(d['groups_for_80pct'])} groups reach 80%."))
    A("Supplier, VPPS L1 and VPPS L2 charts: `figures/pareto_d/`.\n")

    A("### 3c. Words within the largest groups (Pareto C, Level 2)\n")
    A("For each dimension, the 3 largest groups get a unigram, bigram and trigram Pareto (45 charts in "
      "`figures/pareto_c/by_group/`, named `C2_<dimension>_<rank>_<n-gram>.png`). Top 5 unigrams:\n")
    rows = []
    for dim, dd in facts["pareto_c_l2"].items():
        for key, f in dd.items():
            g, name = key.split("|")
            if name != "unigram":
                continue
            lab = g if not dim.startswith("VPPS") else f"VPPS {g}"
            rows.append([dim.replace("_", " "), lab, n(f["n_docs"]),
                         ", ".join(f"{t} ({n(v)})" for t, v in f["top"][:5])])
    A(table(["Dimension", "Group", "ECRs", "Top 5 unigrams (ECRs)"], rows, ["---", "---", "---:", "---"]))
    A("")
    A(fig("pareto_c/by_group/C2_VPPS_L3_1_unigram.png",
          "Number of ECRs by unigram within the largest assembly, top 10."))

    A("### 3d. Word mix across groups (Equation 3)\n")
    for dim in ("Program", "VPPS_L3"):
        h = facts["heat"][f"{dim}|unigram"]
        A(fig(f"heat_tables/L2_word_mix_unigram_by_{dim}.png",
              f"Share of the ECRs of each of the top 10 {DIM_TXT[dim]} that use each of the 8 most frequent "
              f"unigrams. Range across the 10: "
              + "; ".join(f"{t} {pct(h['min'][t], 0)}–{pct(h['max'][t], 0)}" for t in h["terms"][:4]) + "."))
    A("Bigram tables and the other dimensions: `figures/heat_tables/L2_word_mix_*`.\n")

    A("### 3e. Distinctive words (Equation 4, lift)\n")
    for dim in ("Program", "VPPS_L3"):
        A(fig(f"heat_tables/L3_lift_unigram_by_{dim}.png",
              f"Five highest-lift unigrams in each of the top 10 {DIM_TXT[dim]}, among unigrams in at least 5 of the "
              "group's ECRs."))
    rows = []
    for dim in ("Program", "Supplier", "VPPS_L1", "VPPS_L3"):
        for g, lst in list(facts["lift"][f"{dim}|unigram"].items())[:3]:
            rows.append([dim.replace("_", " "), g, ", ".join(f"{w} ×{v} ({d})" for w, v, d in lst[:3])])
    A("Top 3 lifts (DF_gw in parentheses) for the 3 largest groups per dimension:\n")
    A(table(["Dimension", "Group", "Highest-lift unigrams"], rows))
    A("\n### 3f. Do the same words show up in every change? (Equation 5, ubiquity)\n")
    A(fig("ubiquity/U_unigram_by_Program.png", "Ubiquity of unigrams across lead programs with at least 25 ECRs."))
    rows = []
    for dim, u in facts["ubiquity"].items():
        rows.append([dim.replace("_", " "), n(u["groups_ranked"]),
                     ", ".join(f"{t} {pct(v, 0)}" for t, v in u["unigram"][:5]),
                     ", ".join(f"{t} {pct(v, 0)}" for t, v in u["bigram"][:3]),
                     ", ".join(f"{t} {pct(v, 0)}" for t, v in u["trigram"][:2])])
    A(table(["Dimension", "Groups ranked (≥ 25 ECRs)", "Top 5 unigrams by U_w", "Top 3 bigrams", "Top 2 trigrams"],
            rows, ["---", "---:", "---", "---", "---"]))
    A("\n**Outputs:** `step3_counts/ecr_word_counts.csv` (per ECR), `term_counts_overall.csv.gz` (Eq. 1–2), "
      "`group_term_counts_<dimension>.csv.gz` (Eq. 3–4), `ubiquity.csv` (Eq. 5), `group_sizes.csv`.\n")

    # -------- step 4 -------------------------------------------------------------------------------------
    A("## Step 4: Histograms per ECR, program, supplier and VPPS\n")
    A(f"**Literature.** Before modeling, the distributions of document length and term frequency are inspected. Term "
      f"frequencies in a collection follow Zipf's law (frequency roughly inversely proportional to rank), which "
      f"explains why a few terms are very common and most are rare {c('ZIPF', 'MRS', 'SR')}.\n")
    h = facts["hist"]
    w = h["words_per_ecr"]
    A(fig("histograms/H1_words_per_ecr.png",
          f"Number of words (unigram tokens after cleaning) per ECR, all {n(N)} ECRs. Median {w['median']:.0f}, mean "
          f"{w['mean']:.1f}, 90th percentile {w['p90']:.0f}, maximum {n(w['max'])}; {n(w['zero'])} ECRs have no "
          "word left after cleaning."))
    A(fig("histograms/H2_words_per_ecr_by_field.png",
          "Words per ECR by field: " + "; ".join(f"{FIELD_TXT[k]} median {v['median']:.0f} "
                                                 f"(90th pct. {v['p90']:.0f})" for k, v in h["by_field"].items()) + "."))
    rows = [[d.replace("_", " "), n(v["groups"]), f"{v['median']:.0f}", n(v["max"]), n(v["single_ecr"]), n(v["ge_25"])]
            for d, v in h["ecrs_per_group"].items()]
    A(table(["Dimension", "Groups", "Median ECRs per group", "Largest group", "Groups with 1 ECR", "Groups with ≥ 25 ECRs"],
            rows, ["---", "---:", "---:", "---:", "---:", "---:"]))
    A("")
    A(fig("histograms/H3_ecrs_per_Supplier.png", "Number of ECRs per supplier (log-scale bins)."))
    A(fig("histograms/H4_words_per_ecr_by_Program.png", "Words per ECR in the 3 largest lead programs."))
    A(fig("histograms/H5_rank_frequency.png",
          "Number of ECRs containing each unigram, bigram and trigram against its rank (log-log)."))
    dfu = h["df_unigrams"]
    A(fig("histograms/H6_document_frequency_unigrams.png",
          f"Document frequency of the {n(dfu['distinct'])} distinct unigrams: {n(dfu['df_1'])} appear in exactly one "
          f"ECR and {n(dfu['df_lt_5'])} in fewer than 5."))
    A("Other dimensions: `figures/histograms/H3_*`, `H4_*`.\n")

    # -------- step 5 ---------------------------------------------------------------------------------------
    A("## Step 5: Build a vocabulary of key words across all ECRs\n")
    A(f"**Literature.** The vocabulary (dictionary) is the set of terms kept as features {c('MRS')}. Removing very "
      f"infrequent terms is a common preprocessing step whose threshold should be reported and tested {c('DS')}; "
      f"terms below a lower frequency cut-off carry little significance {c('LUHN58')}. scikit-learn implements the "
      f"cut-off as `min_df` {c('SKL')}.\n")
    A(f"**Procedure.** All unigrams, bigrams and trigrams from Step 3; keep a term if it appears in at least "
      f"**{vs['min_df']} ECRs** (DF_w ≥ {vs['min_df']}). No upper cut-off is applied: the most common terms stay in "
      "the vocabulary and their spread is reported through ubiquity (Eq. 5) for review with GM.\n")
    A(table(["n-gram", "Distinct terms", "In vocabulary", "Share of terms kept", "Share of occurrences kept",
             "Terms in exactly 1 ECR"],
            [[k, n(v["distinct_terms"]), n(v["in_vocabulary"]), pct(v["share_of_terms_kept"]),
              pct(v["share_of_occurrences_kept"]), n(v["terms_in_exactly_1_ecr"])]
             for k, v in vs.items() if isinstance(v, dict)], ["---", "---:", "---:", "---:", "---:", "---:"]))
    A(f"\nVocabulary size: **{n(vs['vocabulary_size'])} terms**.\n")
    A("**Outputs:** `step5_vocabulary/vocabulary.csv` (term, n-gram, DF, TF), `all_terms_with_vocabulary_flag.csv.gz`, "
      "`vocabulary_summary.json`.\n")

    # -------- step 6 -----------------------------------------------------------------------------------------
    A("## Step 6: Calculate the term frequency for each ECR\n")
    A(f"**Literature.** The term frequency tf_{{t,d}} is the number of occurrences of term t in document d "
      f"{c('MRS')}; Luhn first used within-document frequency as a measure of a word's significance {c('LUHN57')}.\n")
    A("**Procedure.** tf_{i,w} = f_iw (raw count) for every ECR i and vocabulary term w, stored as a sparse matrix; "
      "the ECR's unigram token count is stored alongside so that relative frequency tf / length can be computed.\n")
    A(table(["Measure", "Value"], [
        ["Rows (ECRs)", n(ts["ecrs"])], ["Columns (vocabulary terms)", n(ts["vocabulary_terms"])],
        ["Non-zero cells", n(ts["nonzero_cells"])], ["Density", f"{ts['density']:.4%}"],
        ["Median distinct vocabulary terms per ECR", f"{ts['median_distinct_vocab_terms_per_ecr']:.0f}"],
        ["ECRs with no vocabulary term", n(ts["ecrs_with_no_vocabulary_term"])]], ["---", "---:"]))
    A("\n**Outputs:** `step6_term_frequency/tf_matrix.npz` (SciPy CSR), `tf_matrix_rows.csv` (row → WO_NO), "
      "`tf_matrix_columns.csv` (column → term), `tf_long.csv.gz` (WO_NO, term, n, tf, unigram tokens in ECR), "
      "`tf_summary.json`.\n")

    # -------- step 7 ----------------------------------------------------------------------------------------
    A("## Step 7: Vectorize each ECR (procedure)\n")
    A(f"**Literature.** In the vector space model each document is a vector over the vocabulary and documents are "
      f"compared by the cosine of the angle between vectors {c('SWY', 'MRS')}. Weighting each tf by inverse document "
      f"frequency, idf_w = log(N / DF_w), down-weights terms that occur in many documents {c('KSJ')}; tf-idf with "
      f"length normalization performed well across Salton and Buckley's comparison of weighting schemes {c('SB')}. "
      f"Sublinear tf (1 + log tf) and L2 normalization are standard variants {c('MRS')}. Latent semantic analysis "
      f"(truncated SVD of the weighted matrix) reduces dimension and groups co-occurring terms {c('LSA')}.\n")
    A("**Procedure for this project.**")
    A("1. Input: `tf_matrix.npz` and `tf_matrix_columns.csv` from Step 6 (no re-tokenizing).")
    A("2. Weight: w_{i,t} = (1 + log tf_{i,t}) · log(N / DF_t), then L2-normalize each row "
      f"(`TfidfTransformer(sublinear_tf=True, norm='l2')` {c('SKL')}).")
    A("3. Keep unigram-only and unigram + bigram versions, so the effect of n-grams can be checked "
      f"{c('DS')}.")
    A("4. Check: list each ECR's highest-weight terms for a random sample and compare with the masked text; compute "
      "cosine similarity between ECRs that share a title to confirm near-1 values.")
    A(f"5. Optional: truncated SVD (LSA) to 100–300 dimensions {c('LSA')} for Step 8 clustering.\n")

    # -------- step 8 ----------------------------------------------------------------------------------------
    A("## Step 8: Analyze and propose new reason codes / characterize ECRs (procedure)\n")
    A(f"**Literature.** Unsupervised methods can discover categories in text, but no method is best for every corpus "
      f"and every result must be validated against human reading {c('GS')}. Discovery and measurement should use "
      f"separate samples, so that categories found in one sample are tested on another {c('GRS')}. Arnarsson et al. "
      f"applied NLP and document clustering to Engineering Change Request databases to group related ECRs and label "
      f"the clusters {c('ARN')}. Topic models represent each document as a mixture of topics: LDA {c('LDA')}; "
      f"non-negative matrix factorization of the tf-idf matrix {c('NMF')}; the structural topic model, which adds "
      f"document covariates (here: program, supplier, VPPS) and was developed for open-ended text responses "
      f"{c('STM')}. Model fit statistics and human judgments of topic quality can disagree, so topics need human "
      f"interpretability checks {c('TEA')}. Words that separate one group from others can be ranked by weighted "
      f"log-odds or lift {c('MCQ')}.\n")
    A("**Procedure for this project** (starts after GM reviews Steps 1–6, per the 11/2 notes):")
    A("1. **Split.** Randomly split ECRs into a discovery set and a held-out measurement set, stratified by STATE "
      "and VPPS L1.")
    A("2. **Discover.** On the discovery set, run k-means on the Step 7 tf-idf/LSA vectors and NMF (and, if "
      "covariate effects are wanted, an STM with program/supplier/VPPS) for a range of K.")
    A("3. **Read and label.** For each K, list each cluster's highest-lift terms (Eq. 4) and 10 representative ECRs; "
      "the team and GM read them and name candidate reason codes and sub-reasons. Choose K by readability and "
      "stability across random seeds, not by a fit statistic alone.")
    A("4. **Write the codebook.** One definition, inclusion and exclusion rule per candidate reason, reviewed with "
      "PPCO.")
    A("5. **Measure.** Hand-label about 300 held-out ECRs with the codebook (Cohen's κ, mid-term Eq. 8), then score "
      "the automated assignment (cluster, topic or LLM) against those labels (macro-F1, mid-term Eq. 7).")
    A("6. **Characterize.** Rerun the Pareto analysis on the new reason codes, by count and by program, supplier "
      "and VPPS.\n")

    # -------- references + appendix ---------------------------------------------------------------------------
    A("## References\n")
    for i, (_, ref) in enumerate(REFS, start=1):
        A(f"[{i}] {ref}  ")
    A("\n## Appendix: reproduce and file index\n")
    A("```bash\npython run_pipeline.py --only text                 # Steps 1-6, figures, this document\n"
      "python run_pipeline.py --only text --states Closed  # same, Closed ECRs only\n```\n")
    A(table(["Folder (in `outputs/01_text/phase1/`)", "Contents"], [
        ["`step1_lemmatize/`", "Lemma merges, before/after sample"],
        ["`step2_stopwords/`", "Removal audit, top lemmas vs. stop list"],
        ["`step3_counts/`", "Per-ECR counts, Eq. 1–5 tables"],
        ["`step5_vocabulary/`", "Vocabulary and pruning summary"],
        ["`step6_term_frequency/`", "Sparse TF matrix and long table"],
        ["`figures/pareto_c/`", "Word Paretos, Level 1 (all text and by field) and Level 2 (by group)"],
        ["`figures/pareto_d/`", "ECR count by group"],
        ["`figures/heat_tables/`", "Word mix (Eq. 3) and lift (Eq. 4) tables"],
        ["`figures/ubiquity/`", "Ubiquity (Eq. 5)"],
        ["`figures/histograms/`", "Step 4 histograms"],
    ]))
    A(f"\nRun: {run_meta.get('timestamp', '')} · {n(N)} ECRs · STATE filter: {run_meta.get('states', 'all')}.")
    path = phase_dir / "Phase1_Steps1-8_Analysis.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
