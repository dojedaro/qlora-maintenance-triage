import pandas as pd
import pytest

from maintriage.data import build_frame, extract_fault, split, top_components

LABELS = ["ACU", "Saloon Light", "Door Roller"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Full restatement after the separator is used, and the repair that follows is dropped.
        (
            "2530 TIMS showed intermittent BCU brake fault ~~~~~ "
            "2530 TIMS showed intermittent BCU brake fault. Train performance - ok.  "
            "Replaced 2530 ACU. Function check-OK.",
            "2530 TIMS showed intermittent BCU brake fault. Train performance - ok",
        ),
        # When the text after the separator is the repair itself, only the title is kept.
        (
            "SnapRep reported 2522 one saloon light n ~~~~~ "
            "#2522-B9 saloon light replaced and function tested ok.  21/11/23",
            "SnapRep reported 2522 one saloon light n",
        ),
        # Repair notes written into the fault field itself are cut too.
        (
            "1504 A7 saloon light not working. Replaced and tested ok. Done by A/B. ~~~~~ x",
            "1504 A7 saloon light not working",
        ),
        # A request to replace is the technician's diagnosis, available at triage: it stays.
        ("2613 - MPV LEAK. TO REPLACE ~~~~~ 2613 MPV replaced.", "2613 - MPV LEAK. TO REPLACE"),
    ],
)
def test_extract_fault_keeps_report_and_drops_repair(raw, expected):
    assert extract_fault(raw) == expected


def test_extract_fault_never_returns_the_repair_action():
    raw = "Car 3534 air dryer fault ~~~~~ Car 3534 air dryer fault.  Replaced air dryer."
    assert "replaced" not in extract_fault(raw).lower()


def _raw(rows):
    return pd.DataFrame(rows, columns=["text", "label"])


def test_build_frame_deduplicates_across_case_and_punctuation():
    raw = _raw(
        [
            ("Door B1 will not close ~~~~~ x", 2),
            ("door b1 will not close. ~~~~~ y", 2),
            ("Saloon light blown at car 3540 ~~~~~ z", 1),
        ]
    )
    frame = build_frame(raw, LABELS)
    assert len(frame) == 2
    assert set(frame["component"]) == {"Door Roller", "Saloon Light"}


def test_build_frame_drops_reports_too_short_to_classify():
    frame = build_frame(_raw([("2530 ~~~~~ Replaced ACU.", 0)]), LABELS)
    assert frame.empty


def test_top_components_keeps_most_frequent():
    frame = pd.DataFrame(
        {"fault": [f"report {i}" for i in range(6)], "component": ["A", "A", "A", "B", "B", "C"]}
    )
    assert set(top_components(frame, 2)["component"]) == {"A", "B"}


def test_split_is_disjoint_stratified_and_deterministic():
    frame = pd.DataFrame(
        {"fault": [f"report {i}" for i in range(200)], "component": ["A", "B", "C", "D"] * 50}
    )
    first, second = split(frame, seed=7), split(frame, seed=7)
    parts = [first.train, first.validation, first.test]
    assert sum(map(len, parts)) == len(frame)
    seen = [set(p["fault"]) for p in parts]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert all(set(p["component"]) == {"A", "B", "C", "D"} for p in parts)
    assert first.test["fault"].tolist() == second.test["fault"].tolist()
    assert first.labels == ["A", "B", "C", "D"]


def test_reports_differing_only_in_car_numbers_are_duplicates():
    raw = _raw(
        [
            ("TC reported 3709 internal smoke pre alarm ~~~~~ x", 0),
            ("TC reported 3512 internal smoke pre alarm ~~~~~ y", 0),
        ]
    )
    assert len(build_frame(raw, LABELS)) == 1


def test_reports_filed_under_conflicting_components_are_dropped():
    raw = _raw(
        [
            ("Door B1 will not close ~~~~~ x", 2),
            ("Door B1 will not close ~~~~~ y", 0),
            ("Saloon light blown at car 3540 ~~~~~ z", 1),
        ]
    )
    assert build_frame(raw, LABELS)["component"].tolist() == ["Saloon Light"]
