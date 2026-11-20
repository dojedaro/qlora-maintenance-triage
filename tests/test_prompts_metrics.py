import pytest

from maintriage.metrics import INVALID, score
from maintriage.prompts import messages, parse_answer

LABELS = ["Saloon CCU", "Saloon CCU Fan", "ACU", "Door Roller"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Door Roller", "Door Roller"),
        ("  door roller.\n", "Door Roller"),
        ('"ACU"', "ACU"),
        ("Saloon CCU Fan", "Saloon CCU Fan"),  # never shortened to the "Saloon CCU" prefix
        ("The component is Saloon CCU Fan", "Saloon CCU Fan"),
        ("Brake pad", INVALID),  # not a listed component
        ("", INVALID),
        ("unsure\nDoor Roller", INVALID),  # only the first line counts
    ],
)
def test_parse_answer(answer, expected):
    assert parse_answer(answer, LABELS) == expected


def test_prompt_lists_every_component_and_the_fault():
    user = messages("Door B1 will not close", LABELS)[1]["content"]
    assert all(f"- {label}" in user for label in LABELS)
    assert user.endswith("Fault report: Door B1 will not close")


def test_score_counts_invalid_answers_as_wrong():
    result = score(
        ["ACU", "ACU", "Door Roller", "Door Roller"], ["ACU", INVALID, "Door Roller", "ACU"]
    )
    assert result["accuracy"] == 0.5
    assert result["invalid_rate"] == 0.25
    assert result["n"] == 4


def test_score_rejects_length_mismatch():
    with pytest.raises(ValueError):
        score(["ACU"], [])
