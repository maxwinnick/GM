"""Steps 3, 5, 6 on a tiny hand-checkable corpus.  Run: python -m pytest -q tests"""

import numpy as np
import pandas as pd
from scipy import sparse

from ecr_text.config import ParetoConfig
from ecr_text.phase1 import Phase1Config, run_phase1, segment_ngrams


def test_segment_ngrams_stay_in_segment():
    assert segment_ngrams("a b c | d e") == ["a", "b", "c", "a b", "b c", "a b c", "d", "e", "d e"]
    assert segment_ngrams(None) == []


def _corpus(n=60):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        prog = "Program 2" if i < 30 else "Program 3"
        title = "harness update" if prog == "Program 2" else "bracket update"
        reason = "cost reduction" if i % 3 == 0 else None
        rows.append({
            "WO_NO": 7_000_000 + i, "STATE": "Closed", "REASON": "", "CREATED_ON": pd.Timestamp("2025-01-01"),
            "MASKED_LEAD_PROGRAM": prog, "MASKED_PROGRAMS": prog, "MASKED_SUPPLIER_NAME": f"Supplier {i % 2}",
            "VPPS": "80.06.01.01", "FNA_DESC": "HARNESS ASM", "TYPE": "", "SUBTYPE": "",
            "COST": float(rng.integers(1, 9)), "EXCHANGE_RATE": 1.0,
            "MASKED_TITLE": title, "TITLE_NLP": title,
            "MASKED_DESC_FOR_CHANGE": None, "DESC_FOR_CHANGE_NLP": None,
            "MASKED_REASON_FOR_CHANGE": reason, "REASON_FOR_CHANGE_NLP": reason,
            "MASKED_COST_COMMENT": None, "COST_COMMENT_NLP": None,
        })
    return pd.DataFrame(rows)


def test_phase1_counts_lift_ubiquity_and_tf(tmp_path):
    p1 = run_phase1(_corpus(), tmp_path, Phase1Config(min_df=5, min_group_ecrs=25, pareto=ParetoConfig(states=None)))
    ov = p1["overall"].set_index("term")
    assert ov.loc["update", "DF"] == 60 and ov.loc["harness", "DF"] == 30 and ov.loc["cost reduction", "DF"] == 20
    assert np.isclose(ov.loc["harness", "s_w"], 0.5)                      # Eq. 1
    t = p1["group_tables"]["Program"].set_index(["group", "term"])
    assert np.isclose(t.loc[("Program 2", "harness"), "s_gw"], 1.0)       # Eq. 3
    assert np.isclose(t.loc[("Program 2", "harness"), "lift_L_gw"], 2.0)  # Eq. 4: 1.0 / 0.5
    u = p1["ubiquity"]["Program"].set_index("term")
    assert u.loc["update", "ubiquity_U_w"] == 1.0                          # Eq. 5: top 10 in both programs
    X = sparse.load_npz(tmp_path / "step6_term_frequency" / "tf_matrix.npz")
    cols = pd.read_csv(tmp_path / "step6_term_frequency" / "tf_matrix_columns.csv")
    assert X.shape == (60, len(cols)) and set(cols.term) >= {"update", "harness update", "cost reduction"}
    assert X[:, cols.index[cols.term == "update"][0]].sum() == 60          # Step 6 raw tf
