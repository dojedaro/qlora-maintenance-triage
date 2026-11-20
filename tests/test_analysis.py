import numpy as np
import pandas as pd

from maintriage.analysis import analyze, keyword_baseline, mcnemar, mentions_label


def test_mentions_label_needs_every_word():
    assert mentions_label("TIMS showed BCU brake fault", "BCU")
    assert mentions_label("one saloon light blown", "Saloon Light")
    assert not mentions_label("saloon door stuck", "Saloon Light")


def test_keyword_baseline_prefers_the_most_specific_name():
    faults = pd.Series(["saloon ccu fan noisy", "door stuck", "saloon ccu trip"])
    labels = ["Saloon CCU", "Saloon CCU Fan", "ACU"]
    assert keyword_baseline(faults, labels, "ACU").tolist() == [
        "Saloon CCU Fan",
        "ACU",
        "Saloon CCU",
    ]


def test_mcnemar_counts_discordant_pairs():
    gold = np.array(["a", "a", "a", "a"])
    first = np.array(["a", "a", "a", "b"])
    second = np.array(["a", "b", "b", "b"])
    result = mcnemar(gold, first, second)
    assert (result["only_first_right"], result["only_second_right"]) == (2, 0)
    assert 0 < result["p_value"] <= 1


def test_analyze_reports_leakage_and_intervals():
    frame = pd.DataFrame(
        {
            "fault": ["BCU fault", "door stuck", "light blown", "BCU trip"],
            "gold": ["BCU", "Door Roller", "Saloon Light", "BCU"],
            "tfidf_logreg": ["BCU", "Door Roller", "BCU", "BCU"],
            "qlora": ["BCU", "Door Roller", "Saloon Light", "ACU"],
            "qlora_raw": ["BCU", "Door Roller ", "saloon light", "ACU"],
        }
    )
    result = analyze(frame)
    assert result["leakage"]["share_mentioning_own_component"] == 0.5
    assert result["qlora_strict_accuracy"] == 0.5  # lowercase answer is not an exact match
    low, high = result["bootstrap"]["qlora"]["accuracy_95ci"]
    assert 0 <= low <= 0.75 <= high <= 1


def test_paired_macro_f1_difference_is_zero_for_identical_models():
    from maintriage.analysis import paired_macro_f1_difference

    gold = np.array(["a", "b", "a", "b"])
    same = np.array(["a", "a", "a", "b"])
    result = paired_macro_f1_difference(gold, same, same, rounds=50)
    assert result == {"difference": 0.0, "ci95": [0.0, 0.0]}


def test_near_duplicate_check_ignores_numbers_in_openings():
    from maintriage.analysis import near_duplicate_check

    frame = pd.DataFrame(
        {
            "fault": ["TC reported 3709 internal smoke pre alarm now", "brand new wording here ok"],
            "gold": ["A", "B"],
            "qlora": ["A", "A"],
        }
    )
    train = pd.Series(["TC reported 1234 internal smoke pre alarm again"])
    result = near_duplicate_check(frame, train)
    assert result["n_novel_opening"] == 1
    assert result["accuracy_on_novel_opening"]["qlora"] == 0.0
