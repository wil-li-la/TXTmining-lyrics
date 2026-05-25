from src.agent.ranking import score_candidate, profile_to_target

STATS = {
    "rhyme_density": {"mean": 0.3, "std": 0.1},
    "valence":       {"mean": 0.0, "std": 0.2},
}


def test_profile_to_target_signed():
    target = profile_to_target({"rhyme_density": +1.0, "valence": -0.5}, STATS)
    assert target["rhyme_density"] == 1.0
    assert target["valence"] == -0.5


def test_score_close_to_target_high():
    # rhyme +1 SD, valence -0.5 SD -- matches the profile direction
    cand_feats = {"rhyme_density": 0.4, "valence": -0.1}
    profile = {"rhyme_density": +1.0, "valence": -0.5}
    s = score_candidate(cand_feats, profile, STATS)
    assert s > 0.95


def test_score_far_from_target_low():
    # opposite direction
    cand_feats = {"rhyme_density": 0.0, "valence": +0.4}
    profile = {"rhyme_density": +1.0, "valence": -0.5}
    s = score_candidate(cand_feats, profile, STATS)
    assert s < 0.5
