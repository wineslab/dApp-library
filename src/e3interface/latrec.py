"""latrec stamping for the dApp, through libe3's own recorder.

libe3 owns the ring format, the stage catalog and the on/off decision (a build
option, LIBE3_ENABLE_LATREC; there is no runtime switch). This module only binds
the per-thread stamping calls and turns them into plain no-ops when the installed
libe3py was built without the recorder. Stage ids come from libe3py, never from a
local table, so they cannot drift from the catalog.

Rings are per thread: a thread must ``open_ring`` before its stamps and
``ctx_set`` calls take effect. Threads that never open one stamp nothing.
"""

try:
    import libe3py as _l
except ImportError:  # pragma: no cover - environment dependent
    _l = None


def _recorder_built_in() -> bool:
    # A build without the recorder compiles latrec_seq_next() to a constant 0.
    if _l is None or not hasattr(_l, "latrec_tstamp_py"):
        return False
    try:
        return _l.latrec_seq_next_py() != 0
    except Exception:
        return False


ENABLED = _recorder_built_in()

_U64 = 0xFFFFFFFFFFFFFFFF


def _id(name: str) -> int:
    return getattr(_l, name, 0)


DECODE_E3SM_BEGIN = _id("LATREC_DECODE_E3SM_BEGIN")
DECODE_E3SM_DONE = _id("LATREC_DECODE_E3SM_DONE")
PROCESS_DONE = _id("LATREC_PROCESS_DONE")
CREATE_OUTPUT = _id("LATREC_CREATE_OUTPUT")
ENCODE_E3SM_BEGIN = _id("LATREC_ENCODE_E3SM_BEGIN")
ENCODE_E3SM_DONE = _id("LATREC_ENCODE_E3SM_DONE")
APPLY_POLICY_DONE = _id("LATREC_APPLY_POLICY_DONE")

PDU_INDICATION = _id("PduType_INDICATION_MESSAGE")
PDU_XAPP_CONTROL = _id("PduType_XAPP_CONTROL_ACTION")
PDU_CONTROL = _id("PduType_DAPP_CONTROL_ACTION")
PDU_REPORT = _id("PduType_DAPP_REPORT")


def _noop_stamp(seq, stage, aux=0, aux2=0):
    return None


def _noop(_arg):
    return None


def _zero():
    return 0


if ENABLED:
    _tstamp = _l.latrec_tstamp_py

    def stamp(seq, stage, aux=0, aux2=0):
        _tstamp(seq, stage, int(aux) & _U64, int(aux2) & _U64)

    open_ring = _l.latrec_tls_open_as_py
    ctx_set = _l.latrec_ctx_set_py
    ctx = _l.latrec_ctx_py
    seq_next = _l.latrec_seq_next_py
    set_output_dir = _l.latrec_set_output_dir_py
else:
    stamp = _noop_stamp
    open_ring = ctx_set = set_output_dir = _noop
    ctx = seq_next = _zero
