"""Fast checks on the pieces most likely to break when the pipeline is run on GM's data."""
import pandas as pd
import pytest

from ecr_topics import common as C
from ecr_topics import compare as K
from ecr_topics import features as F


@pytest.fixture(scope="module")
def cfg(project):
    return C.load_config(project / "config.yaml")


def test_driver_normalization(cfg):
    s = pd.Series(["Engineering Group Manager (EGM)", "EGM|Problem Resolution Tracking System", None, "Problem Res"])
    assert C.normalize_driver(s, cfg).tolist() == ["EGM", "EGM;PRTS", "", "PRTS"]


def test_structured_tokens():
    d = pd.DataFrame({"DRIVER_N": ["PRTS;EGM", ""], "SUB_TRACK": ["Exp Launch", "Regular"],
                      "SUB_KIND": ["Part Chg with New Part Number", "Sourcing"]})
    t = C.structured_tokens(d, ["DRIVER", "SUBTYPE"]).tolist()
    assert t[0] == "driver_prts driver_egm track_exp_launch kind_part_chg_with_new_part_number"
    assert t[1] == "track_regular kind_sourcing"


def test_bigrams_do_not_cross_segments():
    terms = F._segments_to_terms("rear fascia bracket | cost reduction", [1, 2], {"rear"})
    assert "bracket cost" not in terms                      # no phrase across " | "
    assert "rear fascia" in terms and "rear" not in terms   # stop word only drops the single word


def test_mask_and_excel_escapes_removed():
    assert C.clean_raw_text("Hole too small_x000d_ for [Program 12] PN [12345678]") == "Hole too small for PN"


def test_partitions_cover_every_ecr_once(cfg):
    df = C.load_extract(cfg)
    masks = [C.partition_mask(df, cfg, p) for p in cfg["partitions"]]
    assert (sum(m.astype(int) for m in masks) == 1).all()


def test_quote_field_detection():
    doc = "Title: Seat foam change || Cost comment: supplier cost save || Change type: PWO - Sourcing"
    assert K.quote_fields(doc, "supplier cost save || PWO - Sourcing") == ["Cost comment", "Change type"]
