"""A send returns the assigned message id on success; only a negative value is an error."""
import logging
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from e3interface import e3_interface
from e3interface.e3_interface import E3Interface


class _Agent:
    def __init__(self, rc):
        self.rc = rc

    def send_control(self, *args):
        return self.rc

    def send_report(self, *args):
        return self.rc


def _drain(rc, msg, data, caplog):
    iface = object.__new__(E3Interface)
    iface.agent = _Agent(rc)
    iface.outbound_queue = queue.Queue()
    iface.stop_event = threading.Event()
    iface.outbound_queue.put((msg, data))
    t = threading.Thread(target=iface._outbound_connection)
    with caplog.at_level(logging.ERROR, logger=e3_interface.e3_logger.name):
        t.start()
        while not iface.outbound_queue.empty():
            pass
        iface.stop_event.set()
        t.join(timeout=5)
    iface.stop_event.set()
    return [r for r in caplog.records if "send failed" in r.getMessage()]


CONTROL = {"ranFunctionId": 1, "controlId": 1, "actionData": b"x"}
REPORT = {"ranFunctionId": 1, "reportData": b"x"}


@pytest.mark.parametrize("msg,data", [("control", CONTROL), ("report", REPORT)])
def test_a_positive_message_id_is_not_an_error(caplog, msg, data):
    assert _drain(7, msg, data, caplog) == []


@pytest.mark.parametrize("msg,data", [("control", CONTROL), ("report", REPORT)])
def test_a_negative_error_code_is_logged(caplog, msg, data):
    assert len(_drain(-3, msg, data, caplog)) == 1
