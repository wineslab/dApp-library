"""A report always carries a non-zero correlation id: E3-SequenceID is 1..4294967295 on the wire."""
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from e3interface.e3_interface import E3Interface


def _iface():
    iface = object.__new__(E3Interface)
    iface.outbound_queue = queue.Queue()
    iface._report_seq = 0
    iface._report_seq_lock = threading.Lock()
    return iface


def test_a_report_gets_the_next_non_zero_id():
    iface = _iface()
    ids = [iface.schedule_report(1, 1, b"x") for _ in range(3)]
    assert ids == [1, 2, 3]
    queued = [iface.outbound_queue.get_nowait()[1]["sequenceId"] for _ in range(3)]
    assert queued == ids


def test_an_id_given_by_the_caller_is_kept():
    iface = _iface()
    assert iface.schedule_report(1, 1, b"x", sequenceId=77) == 77
    assert iface.outbound_queue.get_nowait()[1]["sequenceId"] == 77


def test_the_id_wraps_to_1_and_never_reaches_0():
    iface = _iface()
    iface._report_seq = 0xFFFFFFFE
    assert iface.schedule_report(1, 1, b"x") == 0xFFFFFFFF
    assert iface.schedule_report(1, 1, b"x") == 1


def test_ids_are_unique_across_threads():
    iface = _iface()
    out = []

    def work():
        out.extend(iface.schedule_report(1, 1, b"x") for _ in range(500))

    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(out)) == 2000
