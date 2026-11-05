from enkural.audio import analysis
from tests.test_audio import all_my_voice


def test_analysis_reports_pitch_and_rhythm():
    clip = all_my_voice()
    a = analysis.analyze(clip)
    assert a["pitch"]["available"]
    assert 120 < a["pitch"]["median_hz"] < 320
    assert a["timbre"]["available"] and a["timbre"]["spectral_centroid_hz"] > 500
    assert a["texture"]["available"]
    assert a["rhythm"]["available"]
    assert 2.5 < a["rhythm"]["articulation_rate_sps"] < 9.0
    assert a["pauses"]["available"] and a["pauses"]["count"] >= 1
    assert a["loudness"]["available"]
    assert isinstance(a["summary"], str) and len(a["summary"]) > 10
