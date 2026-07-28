"""E3 link/transport enumerations.

These are kept purely for CLI/argument convenience in the example dApps (valid
``--link`` / ``--transport`` choices). The actual transport is implemented by
libe3; the dApp only passes the selected strings through to
:class:`e3interface.e3_interface.E3Interface`.

Valid link/transport combinations (enforced by libe3): (zmq, ipc), (zmq, tcp),
(posix, tcp), (posix, sctp), (posix, ipc).
"""

from enum import Enum


class E3LinkLayer(Enum):
    ZMQ = "zmq"
    POSIX = "posix"


class E3TransportLayer(Enum):
    SCTP = "sctp"
    TCP = "tcp"
    IPC = "ipc"
