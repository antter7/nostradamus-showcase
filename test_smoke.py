"""One runnable check: the modules import standalone and the core math holds."""
import math

import category_classifier  # noqa: F401
import grounded_forecast  # noqa: F401
import hybrid_ml_strategy as hml
import post_count_strategy as pcs


def test_bucket_bounds():
    assert pcs.bucket_bounds("20-39") == (20.0, 39.0)
    assert pcs.bucket_bounds("500+") == (500.0, math.inf)
    assert pcs.bucket_bounds("<20")[1] == 19.0


def test_wilson_bounds_contain_observed_rate():
    lo, hi = pcs.wilson_bounds(27, 35, 1.96)
    assert 0 < lo < 27 / 35 < hi < 1


def test_psi_flags_a_shift_only():
    same = [i % 10 for i in range(500)]
    shifted = [v + 5 for v in same]
    assert hml._psi_numeric(same, same) < 0.01
    assert hml._psi_numeric(same, shifted) > hml.DRIFT_PSI_WARN
