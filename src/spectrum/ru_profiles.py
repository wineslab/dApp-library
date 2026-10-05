"""Radio-unit profiles: operator-supplied defaults for deployment-dependent
dApp settings that the RAN does not (yet) send over E3.

A profile only seeds options the operator did not pass; an explicit value
always wins. A key missing from a profile means "not calibrated / not known":
the generic fallback in ``FALLBACK_DEFAULTS`` applies instead.
"""

from typing import Any, Mapping

SETTING_KEYS = (
    "num_prbs",
    "num_subcarrier_spacing",
    "center_freq",
    "noise_floor_threshold",
    "fp16_beta",
    "front_end_sample_rate",
    "max_samples_per_file",
)

DEFAULT_RU = "usrp"

# Target wall-clock span of one SigMF segment at the dApp's write rate.
SEGMENT_SECONDS = 1.0

FALLBACK_DEFAULTS: dict[str, Any] = {
    "num_prbs": 106,
    "num_subcarrier_spacing": 30,
    "center_freq": 3.6192e9,
    "noise_floor_threshold": 53,
    "fp16_beta": 1.0 / 2048.0,
    "front_end_sample_rate": None,
}

_X300_106 = {
    "num_prbs": 106,
    "num_subcarrier_spacing": 30,
    "center_freq": 3.6192e9,
    "noise_floor_threshold": 20,
    "fp16_beta": 1.0 / 128.0,
    "front_end_sample_rate": 46.08e6,
}

_FOXCONN_106 = {
    "num_prbs": 106,
    "num_subcarrier_spacing": 30,
    "center_freq": 3.75e9,
    "noise_floor_threshold": 53,
    "fp16_beta": 1.0 / 2048.0,
}

RU_PROFILES: dict[str, dict[str, Any]] = {
    "usrp": dict(_X300_106),
    "x410": dict(_X300_106),
    "x310-3p58": {**_X300_106, "center_freq": 3.58002e9},
    "colosseum": {
        "num_prbs": 106,
        "num_subcarrier_spacing": 30,
        "center_freq": 3.6192e9,
        "noise_floor_threshold": 53,
        "front_end_sample_rate": 46.08e6,
    },
    "foxconn": dict(_FOXCONN_106),
    "foxconn-273": {**_FOXCONN_106, "num_prbs": 273},
    "benetel": {
        "num_prbs": 273,
        "num_subcarrier_spacing": 30,
        "fp16_beta": 1.0 / 2048.0,
    },
    "rfsim": {
        "num_prbs": 106,
        "num_subcarrier_spacing": 30,
        "center_freq": 3.6192e9,
        "fp16_beta": 1.0 / 2048.0,
        "front_end_sample_rate": 46.08e6,
    },
}

RU_ALIASES = {"x310": "usrp"}

RU_CHOICES = tuple(RU_PROFILES) + tuple(RU_ALIASES)


def canonical_ru(ru: str) -> str:
    name = RU_ALIASES.get(ru, ru)
    if name not in RU_PROFILES:
        raise ValueError(
            f"unknown --ru profile {ru!r}; choose one of {', '.join(RU_CHOICES)}"
        )
    return name


def default_max_samples_per_file(num_prbs: int, num_subcarrier_spacing_khz: int,
                                 seconds: float = SEGMENT_SECONDS) -> int:
    """True IQ samples written in ``seconds`` at the dApp's write rate
    (``num_prbs * 12 * scs`` samples/s)."""
    return int(round(num_prbs * 12 * num_subcarrier_spacing_khz * 1e3 * seconds))


def resolve_ru_settings(ru: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge explicit settings (``None`` = not passed), the ``ru`` profile and
    ``FALLBACK_DEFAULTS``, in that precedence order."""
    profile = RU_PROFILES[canonical_ru(ru)]
    overrides = overrides or {}
    unknown = set(overrides) - set(SETTING_KEYS)
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")

    resolved: dict[str, Any] = {}
    for key in SETTING_KEYS:
        if key == "max_samples_per_file":
            continue
        value = overrides.get(key)
        if value is None:
            value = profile.get(key, FALLBACK_DEFAULTS[key])
        resolved[key] = value

    max_samples = overrides.get("max_samples_per_file")
    if max_samples is None:
        max_samples = default_max_samples_per_file(
            resolved["num_prbs"], resolved["num_subcarrier_spacing"]
        )
    resolved["max_samples_per_file"] = max_samples
    return resolved


def uncalibrated_keys(ru: str) -> list[str]:
    """Settings the profile leaves unset, i.e. served by the generic fallback."""
    profile = RU_PROFILES[canonical_ru(ru)]
    return [k for k in FALLBACK_DEFAULTS if k not in profile and k != "front_end_sample_rate"]
