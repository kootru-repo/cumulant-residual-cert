"""Regression: the random-Pauli snapshot estimator must be unbiased for <P>.

This is the test that catches the missing 3^|P| factor. Run it on a vacuum
state where <Z> is known exactly, with a large enough M that the unbiased
estimator concentrates well within 5%.
"""

from __future__ import annotations

import numpy as np
from cumulant_residual_cert.diagnostic import (  # private/testing helpers
    _one_shot_estimator,
    collect_shadows,
)


def test_one_shot_estimator_unbiased_for_single_Z_on_vacuum():
    n = 2
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0

    shots = collect_shadows(rho, n=n, M=20000, seed=1)
    label = ("Z", "I")
    mean = sum(_one_shot_estimator(label, basis, outcomes) for basis, outcomes in shots) / len(
        shots
    )
    # True <Z_1> on the vacuum is +1.
    assert abs(mean - 1.0) < 0.1, f"snapshot mean {mean:.4f} far from 1.0; missing 3^|P| factor?"


def test_one_shot_estimator_unbiased_for_weight_two_Pauli():
    """Z_1 Z_2 on vacuum has <ZZ> = 1 too."""
    n = 2
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0

    shots = collect_shadows(rho, n=n, M=40000, seed=2)
    label = ("Z", "Z")
    mean = sum(_one_shot_estimator(label, basis, outcomes) for basis, outcomes in shots) / len(
        shots
    )
    assert abs(mean - 1.0) < 0.15, f"snapshot mean {mean:.4f} far from 1.0"
