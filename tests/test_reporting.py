import pandas as pd

from src.reporting import referral_sources_for_completed


def test_referral_sources_count_unique_completed_participants():
    joined = pd.DataFrame(
        {
            "participant_id": ["one", "one", "two", "three"],
            "fully_completed": [True, True, True, False],
            "pretest__Referal Source": ["School", "School", "Friend", "Social media"],
        }
    )

    result = referral_sources_for_completed(joined)

    assert result.set_index("referral_source").loc["School", "participants"] == 1
    assert result.set_index("referral_source").loc["Friend", "participants"] == 1
    assert result["participants"].sum() == 2
