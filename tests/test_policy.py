import numpy as np
import pytest

from spectrum.policy import BaselinePolicy, Observation, Policy, PolicyDetector, load_policy

NUM_PRBS = 106
FFT = 1536
N_SC = NUM_PRBS * 12


def _mag(hot_prbs=(), level=28.0, hot=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    m = level * (0.8 + 0.4 * rng.random(FFT)).astype(np.float32)
    for p in hot_prbs:
        m[p * 12:(p + 1) * 12] = hot
    return m


def _detector(policy):
    return PolicyDetector(policy, fft_size=FFT, num_prbs=NUM_PRBS)


class Echo(Policy):
    def __init__(self, out=None, history_len=4):
        super().__init__(history_len)
        self.out = set(out or [])
        self.seen = []

    def update(self, obs):
        self.seen.append(obs)
        return self.out


def test_observation_layout_and_levels():
    p = Echo()
    d = _detector(p)
    ready, blocked, power_db, nf = d.update(_mag([72]), 5.0, None)
    obs = p.seen[0]
    assert ready and nf is None
    assert obs.power_db.shape == (N_SC,) and obs.prb_power_db.shape == (NUM_PRBS,) and obs.valid.all()
    assert obs.t == 5.0
    assert obs.prb_power_db[72] == pytest.approx(60.0, abs=0.1)
    assert np.nanmedian(obs.prb_power_db) == pytest.approx(20 * np.log10(28.0), abs=1.0)
    assert not blocked.any()


def test_returned_prbs_become_subcarrier_mask():
    d = _detector(Echo({3, 72}))
    _, blocked, _, _ = d.update(_mag(), 0.0, None)
    assert blocked.shape == (FFT,)
    assert set(np.where(blocked)[0] // 12) == {3, 72}
    assert blocked.sum() == 24


def test_valid_mask_excludes_unmeasured_columns():
    p = Echo()
    d = _detector(p)
    valid = np.ones(FFT, dtype=bool)
    valid[10 * 12:11 * 12] = False
    d.update(_mag([10]), 0.0, valid)
    obs = p.seen[0]
    assert np.isnan(obs.prb_power_db[10])
    assert not obs.valid[10 * 12:11 * 12].any()


def test_history_keeps_the_last_observations_with_the_current_one_last():
    p = Echo(history_len=3)
    d = _detector(p)
    for i in range(5):
        d.update(_mag(), float(i), None)
    assert [o.t for o in p.history] == [2.0, 3.0, 4.0]


def test_out_of_range_prb_is_rejected():
    d = _detector(Echo({NUM_PRBS}))
    with pytest.raises(ValueError):
        d.update(_mag(), 0.0, None)


def test_none_return_means_release_everything():
    class Silent(Policy):
        def update(self, obs):
            return None

    _, blocked, _, _ = _detector(Silent()).update(_mag(), 0.0, None)
    assert not blocked.any()


def test_baseline_blocks_incumbent_and_releases_after_hold():
    d = _detector(BaselinePolicy(margin_db=12.0, hold_s=1.0))
    t = 0.0
    for _ in range(5):
        _, b, _, _ = d.update(_mag(), t, None)
        assert not b.any()
        t += 0.02
    for _ in range(5):
        _, b, _, _ = d.update(_mag([72, 73]), t, None)
        assert set(np.where(b)[0] // 12) == {72, 73}
        t += 0.02
    _, b, _, _ = d.update(_mag(), t, None)            # still inside the hold time
    assert set(np.where(b)[0] // 12) == {72, 73}
    _, b, _, _ = d.update(_mag(), t + 1.2, None)      # hold expired
    assert not b.any()


def test_baseline_does_not_need_calibration_across_noise_levels():
    for level in (5.0, 28.0, 150.0, 800.0):
        d = _detector(BaselinePolicy())
        _, b, _, _ = d.update(_mag([40], level=level, hot=level * 60), 0.0, None)
        assert set(np.where(b)[0] // 12) == {40}, level
        _, b, _, _ = _detector(BaselinePolicy()).update(_mag(level=level), 0.0, None)
        assert not b.any(), level


def test_load_policy():
    assert isinstance(load_policy("spectrum.policy:BaselinePolicy"), BaselinePolicy)
    with pytest.raises(ValueError):
        load_policy("nocolon")
    with pytest.raises(TypeError):
        load_policy("collections:OrderedDict")
