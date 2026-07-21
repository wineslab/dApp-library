"""Reader for the OAI-L2-KPM Service Model indication payload.

The OAI gNB ships, per scheduled UL slot, the time-frequency footprint
of every sensing-PUSCH the MAC scheduler injected into that slot. The
dApp correlates these MAC-level KPMs with the L1-KPM SM's IQ samples
by matching ``(sfn, slot)`` — L1 ships the IQ, L2 tells the dApp
which ``(symbol, PRB)`` cells of that IQ carry the sensing signal.

Two wire formats are accepted depending on which libe3 channel the
dApp registered on:

  - JSON channel: inline JSON object
        ``{"timestamp": ..., "sfn": ..., "slot": ..., "beam": ...,
          "sensing_ranges": [{"start_symbol": ..., "num_symbols": ...,
                              "rb_start": ..., "rb_size": ...}, ...]}``

  - ASN.1 channel: APER-encoded ``L2KPM-Indication`` (schema in
        ``defs/e3sm_oai_l2_kpm.asn``)

:class:`L2KpmIndication` exposes one self-contained record with
``from_bytes`` auto-dispatching on the first byte (``{`` → JSON,
anything else → APER).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import asn1tools


# ----------------------------------------------------------------------------
# OAI-L2-KPM ASN.1 schema. asn1tools' compile_files is heavy (~50 ms on
# cold disk); cache the compiler at module scope.
# ----------------------------------------------------------------------------

_OAI_L2_KPM_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "defs", "e3sm_oai_l2_kpm.asn"
)
_oai_l2_kpm_compiler = None


def _get_oai_l2_kpm_compiler():
    """Return a cached asn1tools compiler for L2KPM-Indication."""
    global _oai_l2_kpm_compiler
    if _oai_l2_kpm_compiler is None:
        _oai_l2_kpm_compiler = asn1tools.compile_files(
            _OAI_L2_KPM_SCHEMA_PATH, codec="per"
        )
    return _oai_l2_kpm_compiler


@dataclass(frozen=True)
class SensingRange:
    """One sensing-PUSCH injection's time-frequency footprint."""

    start_symbol: int
    num_symbols: int
    rb_start: int
    rb_size: int


@dataclass(frozen=True)
class L2KpmIndication:
    """Parsed L2-KPM indication payload.

    ``timestamp_ns`` is CLOCK_MONOTONIC nanoseconds at publish time on
    the gNB. ``sfn`` and ``slot`` are the keys for correlating with
    the L1 IQ indication. ``beam`` defaults to 0 when absent on the
    wire (today the gNB always emits beam 0).
    """

    timestamp_ns: int
    sfn: int
    slot: int
    beam: int
    sensing_ranges: List[SensingRange] = field(default_factory=list)

    # ------------------------------------------------------------------ JSON
    @classmethod
    def from_json_bytes(cls, data: bytes) -> Optional["L2KpmIndication"]:
        """Parse the JSON-encoded payload. Returns ``None`` on malformed input."""
        try:
            obj = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        try:
            ranges = []
            for r in obj.get("sensing_ranges", []) or []:
                ranges.append(
                    SensingRange(
                        start_symbol=int(r["start_symbol"]),
                        num_symbols=int(r["num_symbols"]),
                        rb_start=int(r["rb_start"]),
                        rb_size=int(r["rb_size"]),
                    )
                )
            return cls(
                timestamp_ns=int(obj.get("timestamp", 0)),
                sfn=int(obj["sfn"]),
                slot=int(obj["slot"]),
                beam=int(obj.get("beam", 0)),
                sensing_ranges=ranges,
            )
        except (KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ APER
    @classmethod
    def from_asn1_bytes(cls, data: bytes) -> Optional["L2KpmIndication"]:
        """Parse the APER-encoded payload. Returns ``None`` on malformed input.

        Schema: ``L2KPM-Indication`` in ``defs/e3sm_oai_l2_kpm.asn`` —
        kept in lockstep with the OAI APER encoder.
        """
        try:
            decoded = _get_oai_l2_kpm_compiler().decode("L2KPM-Indication", data)
        except Exception:
            return None

        try:
            ranges = []
            for r in decoded.get("sensingRanges", []) or []:
                ranges.append(
                    SensingRange(
                        start_symbol=int(r["startSymbol"]),
                        num_symbols=int(r["numSymbols"]),
                        rb_start=int(r["rbStart"]),
                        rb_size=int(r["rbSize"]),
                    )
                )
            return cls(
                timestamp_ns=int(decoded.get("timestamp", 0)),
                sfn=int(decoded["sfn"]),
                slot=int(decoded["slot"]),
                beam=int(decoded.get("beam") or 0),
                sensing_ranges=ranges,
            )
        except (KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------- dispatch
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["L2KpmIndication"]:
        """Auto-detect JSON vs APER and dispatch to the matching parser.

        Discriminator is the first byte: JSON payloads start with
        ``{`` (0x7B); APER payloads start with the SEQUENCE OPTIONAL
        preamble byte for this schema (one of 0x00 / 0x80 — only the
        ``beam`` field is OPTIONAL). 0x7B is not a valid APER
        preamble byte for L2KPM-Indication, so a one-byte sniff is
        sufficient.
        """
        if not data:
            return None
        if data[0] == 0x7B:  # '{'
            return cls.from_json_bytes(data)
        return cls.from_asn1_bytes(data)
