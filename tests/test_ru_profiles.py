import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectrum.ru_profiles import (  # noqa: E402
    DEFAULT_RU,
    RU_CHOICES,
    RU_PROFILES,
    canonical_ru,
    default_max_samples_per_file,
    resolve_ru_settings,
    uncalibrated_keys,
)


def test_default_profile_is_usrp():
    assert DEFAULT_RU == "usrp"
    assert resolve_ru_settings(DEFAULT_RU) == {
        "num_prbs": 106,
        "num_subcarrier_spacing": 30,
        "center_freq": 3.6192e9,
        "noise_floor_threshold": 20,
        "fp16_beta": 1.0 / 128.0,
        "front_end_sample_rate": 46.08e6,
        "max_samples_per_file": 38_160_000,
    }


def test_alias_resolves_to_usrp():
    assert canonical_ru("x310") == "usrp"
    assert resolve_ru_settings("x310") == resolve_ru_settings("usrp")


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="unknown --ru profile"):
        resolve_ru_settings("nonexistent")


def test_unknown_setting_rejected():
    with pytest.raises(ValueError, match="unknown settings"):
        resolve_ru_settings("usrp", {"ota": True})


def test_explicit_flags_win_over_profile():
    s = resolve_ru_settings("usrp", {
        "noise_floor_threshold": 53,
        "num_prbs": 273,
        "center_freq": 3.75e9,
        "fp16_beta": 1.0 / 2048.0,
        "front_end_sample_rate": 61.44e6,
        "max_samples_per_file": 1000,
    })
    assert s["noise_floor_threshold"] == 53
    assert s["num_prbs"] == 273
    assert s["center_freq"] == 3.75e9
    assert s["fp16_beta"] == 1.0 / 2048.0
    assert s["front_end_sample_rate"] == 61.44e6
    assert s["max_samples_per_file"] == 1000


def test_none_means_not_passed():
    assert resolve_ru_settings("foxconn", {"center_freq": None}) == resolve_ru_settings("foxconn")


def test_zero_threshold_is_an_explicit_value():
    assert resolve_ru_settings("usrp", {"noise_floor_threshold": 0})["noise_floor_threshold"] == 0


def test_max_samples_follows_overridden_geometry():
    s = resolve_ru_settings("usrp", {"num_prbs": 273})
    assert s["max_samples_per_file"] == default_max_samples_per_file(273, 30) == 98_280_000


def test_uncalibrated_profile_falls_back_to_generic_defaults():
    s = resolve_ru_settings("benetel")
    assert s["num_prbs"] == 273
    assert s["noise_floor_threshold"] == 53
    assert s["center_freq"] == 3.6192e9
    assert s["front_end_sample_rate"] is None
    assert set(uncalibrated_keys("benetel")) == {"center_freq", "noise_floor_threshold"}
    assert uncalibrated_keys("usrp") == []


def test_every_profile_resolves():
    for name in RU_CHOICES:
        s = resolve_ru_settings(name)
        assert s["num_prbs"] in (106, 273)
        assert s["fp16_beta"] in (1.0 / 128.0, 1.0 / 2048.0)


def test_usrp_family_uses_small_beta():
    for name in ("usrp", "x410", "x310-3p58"):
        assert RU_PROFILES[name]["fp16_beta"] == 1.0 / 128.0
    assert RU_PROFILES["x310-3p58"]["center_freq"] == 3.58002e9
    assert RU_PROFILES["foxconn"]["center_freq"] == 3.75e9
