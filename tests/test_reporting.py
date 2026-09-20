import pandas as pd

from src.reporting import completed_participant_details


def test_completed_participant_details_are_unique_and_sorted_by_date():
    joined = pd.DataFrame(
        {
            "participant_id": ["one", "one", "two", "three"],
            "fully_completed": [True, True, True, False],
            "pretest__StartDate": [
                "2026-04-01 10:00:00",
                "2026-04-01 10:00:00",
                "2026-05-01 10:00:00",
                "2026-06-01 10:00:00",
            ],
            "pretest__City": ["City A", "City A", "City B", "City C"],
            "pretest__State": ["State A", "State A", "State B", "State C"],
            "pretest__Country": ["US", "US", "US", "US"],
            "pretest__CAge": ["8", "8", "9", "10"],
            "pretest__PEthnicity": ["A", "A", "B", "C"],
            "pretest__CEthnicity": ["D", "D", "E", "F"],
            "pretest__Referal Source": ["School", "School", "Friend", "Social media"],
        }
    )

    result = completed_participant_details(joined)

    assert result["participant_id"].tolist() == ["two", "one"]
    assert result["referral_source"].tolist() == ["Friend", "School"]
    assert result["location"].tolist() == ["City B, State B, US", "City A, State A, US"]
