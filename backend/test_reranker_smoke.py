import os

import pytest

from backend.evaluation.reranker_benchmark import run_real_reranker_smoke


@pytest.mark.skipif(
    os.getenv("RUN_REAL_RERANKER_SMOKE") != "1",
    reason="Set RUN_REAL_RERANKER_SMOKE=1 to load the real local Cross-Encoder.",
)
def test_real_cross_encoder_scores_and_reranks_candidates():
    report = run_real_reranker_smoke()
    assert report["effective"] is True
    assert report["ranking"][0] == "F0002"
    assert len(report["real_scores"]) == 3
    assert len(set(report["real_scores"])) > 1
