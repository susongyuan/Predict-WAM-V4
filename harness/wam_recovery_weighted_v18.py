"""User-authorized threefold verified-recovery action weights; unchanged reducer."""
from copy import deepcopy
import numpy as np
from harness.wam_recovery_weighted_v16 import (POLICY as PREVIOUS_POLICY,
    install_recovery_weighting, weighted_action_mse, mixed_weighted_samples as previous_samples)

POLICY = deepcopy(PREVIOUS_POLICY)
POLICY.update(schema='expert-recovery-action-weight-v18', verified_recovery_action_weight=3.0)


def mixed_weighted_samples(original, corrections, rng):
    samples = previous_samples(original, corrections, rng)
    samples[-1]['action_loss_weight'] = np.float32(3.0)
    return samples
