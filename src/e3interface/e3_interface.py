import queue
import threading
import time

from .e3_logging import e3_logger
from .libe3_agent import (
    Libe3Agent,
    SUCCESS,
    EVENT_INDICATION,
    EVENT_XAPP_CONTROL,
    EVENT_SUBSCRIPTION_RESPONSE,
    EVENT_SETUP_RESPONSE,
    EVENT_MESSAGE_ACK,
)

# Inbound drain tuning. poll_events blocks up to POLL_TIMEOUT_MS for the first
# event (so stop_event is checked regularly) then sweeps up to POLL_MAX_BATCH
# already-queued events in one call — amortising the GIL acquire + Python<->C++
# crossing across the batch, which is what sustains the sub-ms/high-throughput
# E3AP rate. See libe3 swig/e3_dapp_session.hpp.
POLL_MAX_BATCH = 256
POLL_TIMEOUT_MS = 100
SETUP_WAIT_MS = 6000


class E3Interface:
    """Singleton E3AP interface, backed by the libe3 dApp session (``libe3py``).

    All E3AP operations (transport, setup handshake, subscribe, indication/
    control framing, wire encoding) are handled by libe3. This class keeps the
    dApp-facing orchestration: the setup call, the outbound schedule queue, the
    batched inbound drain, and the callback dispatch model (unchanged, preserving
    the concurrency fixes for issues #56/#57/#83). Service-model payloads are
    opaque ``bytes`` here and are encoded/decoded in the dApp subclasses.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(E3Interface, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, link: str = "zmq", transport: str = "ipc",
                 encoding: str = "asn1", **kwargs):
        """Initialise the interface.

        Args:
            link: link layer ("zmq" or "posix").
            transport: transport layer ("ipc", "tcp", "sctp").
            encoding: E3AP wire encoding ("asn1", "json", "protobuf"); must match
                the gNB E3Configuration.
        """
        if not hasattr(self, "initialized"):
            self.indication_callbacks = {}   # key: (dAppId, subscriptionId) -> list(callbacks)
            self.subscription_callbacks = {} # key: dAppId -> list(callbacks)
            self.xapp_control_callbacks = {} # key: dAppId -> list(callbacks)
            # Guards the three callback dicts above: caller-thread add_/remove_*
            # vs. inbound-thread iteration in the _handle_* dispatchers.
            self._callback_lock = threading.Lock()
            self.stop_event = threading.Event()
            # Subscription-response correlation. The wire is fire-and-forget, so a
            # queued request is NOT an accepted one — a queue put() succeeding
            # says nothing about the gNB. libe3 now returns the assigned request
            # id from subscribe() (a positive int) and the RAN echoes it in the
            # SubscriptionResponse, so we map response -> RF by request id
            # (recorded when the outbound thread actually sends). This is robust
            # to a dropped/reordered response, unlike the former FIFO pairing
            # which desynced permanently on the first drop. Callers block on the
            # recorded verdict until the RAN accepts/rejects (or the wait times
            # out, which drops the pending entry so it can't desync later).
            self._sub_cv = threading.Condition()
            self._sub_reqid_to_rf = {}   # request_id -> ranFunctionId (in flight)
            self._sub_pending_rfs = set()  # RFs queued and awaiting a verdict
            self._sub_results = {}       # ranFunctionId -> bool (positive?)

            self._link = link
            self._transport = transport
            self._encoding = encoding
            # The libe3 agent is created lazily in send_setup_request(), once the
            # dApp identity (name/version/vendor) is known — libe3 needs it in the
            # E3Config before start().
            self.agent: Libe3Agent | None = None

            self.outbound_queue = queue.Queue()
            e3_logger.info(
                "E3Interface configured (link=%s transport=%s encoding=%s)",
                link, transport, encoding,
            )
            # Set LAST: __init__ runs unlocked (only __new__ holds _lock), and
            # the re-init guard keys on `initialized`. Assigning it only after
            # agent/outbound_queue/_link etc. are in place prevents a second
            # thread from seeing a truthy `initialized` and using a
            # partially-constructed singleton.
            self.initialized = True

    def send_setup_request(self, e3apProtocolVersion: str = "0.0.0", dAppName: str = "",
                           dAppVersion: str = "0.0.0", vendor: str = "") -> tuple[bool, dict | None]:
        """Create the libe3 agent, start it, and complete the setup handshake.

        Returns (positive, setupResponseDict). The dict shape matches what the
        dApp base class and examples consume (dAppIdentifier, responseCode,
        ranFunctionList[{ranFunctionIdentifier, telemetryIdentifierList,
        controlIdentifierList, ranFunctionData}]).
        """
        e3_logger.info(
            "Send setup request for dApp '%s' version %s (vendor=%s, e3ap_version=%s)",
            dAppName, dAppVersion, vendor, e3apProtocolVersion,
        )
        try:
            self.agent = Libe3Agent(
                link=self._link,
                transport=self._transport,
                encoding=self._encoding,
                dapp_name=dAppName,
                dapp_version=dAppVersion,
                vendor=vendor,
                e3ap_version=e3apProtocolVersion,
            )
            rc = self.agent.start()
            if rc != SUCCESS:
                e3_logger.error("libe3 agent failed to start (ErrorCode=%s)", rc)
                return False, None

            rc = self.agent.wait_for_setup(SETUP_WAIT_MS)
            if rc != SUCCESS:
                e3_logger.error("E3 setup handshake failed (ErrorCode=%s)", rc)
                return False, None
        except ImportError as exc:
            # Deterministic (libe3py not installed): re-raise with the install
            # hint so the caller's retry loop does not spin 3x on it and the
            # real cause isn't buried under a generic "setup failed" message.
            e3_logger.error("Cannot start the E3 agent: %s", exc)
            raise
        except Exception:
            e3_logger.exception("Unable to establish the E3 setup with the RAN")
            return False, None

        response = self.agent.setup_response_dict()
        e3_logger.info("Setup response received: %s", response)
        return self.agent.setup_positive(), response

    def send_subscription_request(self, dappId: int, ranFunctionId: int, telemetryIds: list[int],
                                  controlIds: list[int], subscriptionTime: int | None = None,
                                  periodicity: int | None = None) -> bool:
        """Queue a subscription request (sent by the outbound thread via libe3).

        The queue put() succeeding says nothing about the gNB, so we mark the RF
        as pending here; the outbound thread records the request-id->RF mapping
        once libe3 assigns it, and wait_for_subscription_result blocks on the
        verdict _handle_subscription_response records for that RF.
        """
        e3_logger.info("Queue subscription request for RAN function %s", ranFunctionId)
        with self._sub_cv:
            self._sub_pending_rfs.add(ranFunctionId)
            self._sub_results.pop(ranFunctionId, None)
        try:
            self.outbound_queue.put(('subscription', {
                'ranFunctionId': ranFunctionId,
                'telemetryIds': telemetryIds,
                'controlIds': controlIds,
                'subscriptionTime': subscriptionTime,
                'periodicity': periodicity,
            }))
            return True
        except Exception as e:
            with self._sub_cv:
                self._sub_pending_rfs.discard(ranFunctionId)
            e3_logger.error("Failed to queue subscription request: %s", e)
            return False

    def wait_for_subscription_result(self, ran_function_id: int, timeout: float = 5.0):
        """Block until the gNB's SubscriptionResponse for ``ran_function_id``
        arrives. Returns True (accepted), False (rejected), or None (no response
        within ``timeout``). This is the only authoritative signal that a
        subscription was accepted — enqueueing the request is not."""
        deadline = time.monotonic() + timeout
        with self._sub_cv:
            while ran_function_id not in self._sub_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Timed out: stop tracking this RF and drop any request-id
                    # mapping pointing at it, so a late/duplicate response can't
                    # resurrect a stale verdict or desync later correlations.
                    self._sub_pending_rfs.discard(ran_function_id)
                    for rid in [r for r, rf in self._sub_reqid_to_rf.items()
                                if rf == ran_function_id]:
                        del self._sub_reqid_to_rf[rid]
                    return None
                self._sub_cv.wait(remaining)
            return self._sub_results[ran_function_id]

    def send_message_ack(self, requestId: int, responseCode: str = "positive"):
        """Queue a message acknowledgment."""
        self.outbound_queue.put(('ack', {
            'requestId': requestId,
            'positive': responseCode == "positive",
        }))
        e3_logger.debug("Message ACK queued for request %s", requestId)
        return True

    def send_release_message(self, dappId: int):
        """Queue a release message to end interactions with the RAN."""
        self.outbound_queue.put(('release', {}))
        e3_logger.debug("Release message queued for dApp %s", dappId)
        return True

    def setup_connections(self):
        # Two worker threads: inbound drains libe3's event queue, outbound sends
        # scheduled controls/reports/etc through libe3.
        self.inbound_thread = threading.Thread(target=self._inbound_connection)
        self.outbound_thread = threading.Thread(target=self._outbound_connection)
        self.inbound_thread.start()
        self.outbound_thread.start()

    def _inbound_connection(self):
        """Batched drain of libe3 inbound events into the callback dispatchers."""
        e3_logger.info("Start inbound loop")
        last_dropped = 0
        try:
            while not self.stop_event.is_set():
                batch = self.agent.poll_events(POLL_MAX_BATCH, POLL_TIMEOUT_MS)
                for ev in batch:
                    # Per-event resilience: one malformed/unknown event must not
                    # tear down the loop (mirrors the pre-libe3 per-PDU isolation).
                    try:
                        kind = ev.kind
                        if kind == EVENT_INDICATION:
                            self._handle_indication_data(
                                ev.dapp_id, ev.ran_function_id, ev.get_payload())
                        elif kind == EVENT_XAPP_CONTROL:
                            self._handle_xapp_control_data(
                                ev.dapp_id, ev.ran_function_id, ev.get_payload())
                        elif kind == EVENT_SUBSCRIPTION_RESPONSE:
                            self._handle_subscription_response({
                                'dAppIdentifier': ev.dapp_id,
                                'responseCode': 'positive' if ev.response_code == 0 else 'negative',
                                'subscriptionId': ev.subscription_id,
                                'requestId': ev.request_id,
                            })
                        elif kind == EVENT_MESSAGE_ACK:
                            e3_logger.debug(
                                "Received message ACK: request=%s rc=%s",
                                ev.request_id, ev.response_code)
                        elif kind == EVENT_SETUP_RESPONSE:
                            # Consumed synchronously in send_setup_request(); ignore.
                            pass
                        else:
                            e3_logger.warning("Unrecognized event kind %r — dropping", kind)
                    except Exception:
                        e3_logger.exception(
                            "Inbound event dispatch failed (kind=%r); dropping and continuing",
                            getattr(ev, "kind", None))
                        continue

                dropped = self.agent.dropped_events()
                if dropped != last_dropped:
                    e3_logger.warning(
                        "libe3 inbound ring dropped %d event(s) total (backpressure)", dropped)
                    last_dropped = dropped
        except Exception:
            e3_logger.exception("Fatal error in inbound thread")
            self.stop_event.set()
        finally:
            e3_logger.info("Close inbound connection")

    def _outbound_connection(self):
        """Drain the outbound queue and forward each message through libe3."""
        e3_logger.info("Start outbound loop")
        try:
            while not self.stop_event.is_set():
                try:
                    msg, data = self.outbound_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Per-message resilience: a single bad send (e.g. a control with
                # non-bytes actionData that raises in libe3) must not set
                # stop_event and tear down both the outbound AND inbound planes.
                # Mirror the inbound loop's per-event isolation.
                try:
                    e3_logger.debug("Outbound queue has got '%s', %s", msg, data)
                    rc = SUCCESS
                    match msg:
                        case "control":
                            rc = self.agent.send_control(
                                data["ranFunctionId"], data["controlId"], data["actionData"])
                        case "subscription":
                            self._send_subscription(data)
                        case "ack":
                            rc = self.agent.send_message_ack(data["requestId"], data["positive"])
                        case "report":
                            rc = self.agent.send_report(data["ranFunctionId"], data["reportData"])
                        case "release":
                            rc = self.agent.release()
                        case _:
                            e3_logger.error("Unrecognized outbound message: %r; dropping", msg)
                            continue

                    if rc != SUCCESS:
                        e3_logger.error("libe3 %s send failed (ErrorCode=%s)", msg, rc)
                except Exception:
                    e3_logger.exception(
                        "Outbound message %r failed; dropping and continuing", msg)
                    continue
        except Exception:
            e3_logger.exception("Fatal error in outbound thread")
            self.stop_event.set()
        finally:
            e3_logger.info("Close outbound connection")

    def _send_subscription(self, data):
        """Send a subscribe via libe3 and record the request-id -> RF mapping.

        libe3's subscribe() returns the assigned request id (a positive int) on
        success, or a negative ErrorCode on failure. We hold ``_sub_cv`` across
        the send + mapping-record so the mapping is in place before the inbound
        thread can process the (later, post-round-trip) SubscriptionResponse.
        """
        rf = data["ranFunctionId"]
        with self._sub_cv:
            ret = self.agent.subscribe(
                rf, data["telemetryIds"], data["controlIds"],
                data.get("subscriptionTime"), data.get("periodicity"))
            if ret > 0:
                self._sub_reqid_to_rf[ret] = rf
            else:
                # Send failed: record a negative verdict so a waiter doesn't
                # block for the full timeout, and stop tracking the RF.
                self._sub_pending_rfs.discard(rf)
                self._sub_results[rf] = False
                self._sub_cv.notify_all()
                e3_logger.error("libe3 subscribe send failed for RF %s (ErrorCode=%s)", rf, ret)

    def _handle_subscription_response(self, data):
        dapp_id = data['dAppIdentifier']
        # Correlate by request id: the RAN echoes the request's id in the
        # response, and the outbound thread recorded request_id -> RF when it
        # sent the subscribe. Robust to a dropped/reordered response (the former
        # FIFO pairing blamed the wrong RF and stayed off-by-one on a drop).
        request_id = data.get('requestId')
        positive = data.get('responseCode') == 'positive'
        with self._sub_cv:
            rfid = self._sub_reqid_to_rf.pop(request_id, None)
            if rfid is not None:
                self._sub_pending_rfs.discard(rfid)
                self._sub_results[rfid] = positive
                self._sub_cv.notify_all()
            else:
                e3_logger.warning(
                    "SubscriptionResponse for unknown/stale requestId=%s; ignoring",
                    request_id)
        with self._callback_lock:
            e3_logger.debug(f"DApp ID requested {dapp_id}, map status {self.subscription_callbacks}")
            callbacks = list(self.subscription_callbacks.get(dapp_id, []))
        if callbacks:
            e3_logger.debug(f"Launch {len(callbacks)} subscription callback(s) for dApp {dapp_id}")
            for callback in callbacks:
                callback(data)
        else:
            e3_logger.warning(f"No subscription callback registered for dApp {dapp_id}")

    def _handle_indication_data(self, dapp_identifier, ran_function_id, data):
        # Snapshot the matching callbacks under the lock (#83: never iterate the
        # live dict while it may be mutated from another thread), then invoke
        # outside the lock. De-duplicate while preserving order: the same
        # callback can be registered under multiple (dAppId, subscriptionId)
        # keys (e.g. the base DApp registers _handle_indication under
        # subscriptionId 0), and must fire at most once per indication.
        with self._callback_lock:
            snapshot = [
                callback
                for key, cbs in self.indication_callbacks.items()
                if key[0] == dapp_identifier
                for callback in cbs
            ]
        seen = set()
        callbacks = []
        for callback in snapshot:
            if callback in seen:
                continue
            seen.add(callback)
            callbacks.append(callback)
        if callbacks:
            e3_logger.debug(f"Launch {len(callbacks)} unique callback(s) for dApp {dapp_identifier}")
            for callback in callbacks:
                callback(dapp_identifier, ran_function_id, data)
        else:
            e3_logger.warning(f"No indication callback registered for dApp {dapp_identifier}")

    def _handle_xapp_control_data(self, dapp_identifier, ran_function_id, xapp_control_data):
        e3_logger.debug(
            "Received xAppControlAction: dApp=%s, ranFunc=%s, payload=%d bytes",
            dapp_identifier, ran_function_id, len(xapp_control_data),
        )
        with self._callback_lock:
            callbacks = list(self.xapp_control_callbacks.get(dapp_identifier, []))
        if callbacks:
            e3_logger.debug(f"Launch {len(callbacks)} xApp control callback(s) for dApp {dapp_identifier}")
            for callback in callbacks:
                callback(dapp_identifier, xapp_control_data)
        else:
            e3_logger.warning(f"No xApp control callback registered for dApp {dapp_identifier}")

    def schedule_control(self, dappId: int, ranFunctionId: int, controlId: int, actionData: bytes = b""):
        self.outbound_queue.put(('control', {
            'ranFunctionId': ranFunctionId,
            'controlId': controlId,
            'actionData': actionData,
        }))

    def schedule_report(self, dappId: int, ranFunctionId: int, reportData: bytes):
        self.outbound_queue.put(('report', {
            'ranFunctionId': ranFunctionId,
            'reportData': reportData,
        }))

    def add_subscription_callback(self, dapp_id: int, callback):
        with self._callback_lock:
            if dapp_id not in self.subscription_callbacks:
                e3_logger.debug(f"Add first subscription callback for dApp {dapp_id}")
                self.subscription_callbacks[dapp_id] = [callback]
            else:
                callbacks = list(self.subscription_callbacks[dapp_id])
                if callback not in callbacks:
                    e3_logger.debug(f"Add additional subscription callback for dApp {dapp_id}")
                    callbacks.append(callback)
                    self.subscription_callbacks[dapp_id] = callbacks
                else:
                    e3_logger.warning(f"Subscription callback already registered for dApp {dapp_id}, skipping")

    def remove_subscription_callback(self, dapp_id: int, callback=None):
        with self._callback_lock:
            if dapp_id in self.subscription_callbacks:
                if callback is None:
                    e3_logger.debug(f"Remove all subscription callbacks for dApp {dapp_id}")
                    del self.subscription_callbacks[dapp_id]
                else:
                    callbacks = list(self.subscription_callbacks[dapp_id])
                    if callback in callbacks:
                        e3_logger.debug(f"Remove specific subscription callback for dApp {dapp_id}")
                        callbacks.remove(callback)
                        if callbacks:
                            self.subscription_callbacks[dapp_id] = callbacks
                        else:
                            del self.subscription_callbacks[dapp_id]
                    else:
                        e3_logger.warning(f"Specific subscription callback not found for dApp {dapp_id}")
            else:
                e3_logger.warning(f"No subscription callbacks found for dApp {dapp_id}")

    def add_indication_callback(self, dapp_id: int, subscription_id: int, callback):
        key = (dapp_id, subscription_id)
        with self._callback_lock:
            if key not in self.indication_callbacks:
                e3_logger.debug(f"Add first indication callback for dApp {dapp_id}, subscription {subscription_id}")
                self.indication_callbacks[key] = [callback]
            else:
                callbacks = list(self.indication_callbacks[key])
                if callback not in callbacks:
                    e3_logger.debug(f"Add additional indication callback for dApp {dapp_id}, subscription {subscription_id}")
                    callbacks.append(callback)
                    self.indication_callbacks[key] = callbacks
                else:
                    e3_logger.warning(
                        f"Indication callback already registered for dApp {dapp_id}, subscription {subscription_id}, skipping"
                    )

    def remove_indication_callback(self, dapp_id: int, subscription_id: int | None = None, callback=None):
        with self._callback_lock:
            if subscription_id is None:
                # Remove all entries for this dapp_id
                keys_to_remove = [key for key in self.indication_callbacks if key[0] == dapp_id]
                if keys_to_remove:
                    e3_logger.debug(f"Remove all indication callbacks for dApp {dapp_id}")
                    for key in keys_to_remove:
                        del self.indication_callbacks[key]
                else:
                    e3_logger.warning(f"No indication callbacks found for dApp {dapp_id}")

            elif callback is not None:
                # Remove specific callback from any key matching dapp_id
                found = False
                keys_to_check = [key for key in self.indication_callbacks if key[0] == dapp_id]
                for key in keys_to_check:
                    callbacks = list(self.indication_callbacks[key])
                    if callback in callbacks:
                        e3_logger.debug(f"Remove specific callback for dApp {dapp_id}, subscription {key[1]}")
                        callbacks.remove(callback)
                        if callbacks:
                            self.indication_callbacks[key] = callbacks
                        else:
                            del self.indication_callbacks[key]
                        found = True
                        break
                if not found:
                    e3_logger.warning(f"Specific callback not found for dApp {dapp_id}")

            else:
                # subscription_id is present, callback is None: remove the specific key
                key = (dapp_id, subscription_id)
                if key in self.indication_callbacks:
                    e3_logger.debug(f"Remove all callbacks for dApp {dapp_id}, subscription {subscription_id}")
                    del self.indication_callbacks[key]
                else:
                    e3_logger.warning(f"No indication callbacks found for dApp {dapp_id}, subscription {subscription_id}")

    def add_xapp_control_callback(self, dapp_id: int, subscription_id: int, callback):
        with self._callback_lock:
            if dapp_id not in self.xapp_control_callbacks:
                e3_logger.debug(f"Add first xApp control callback for dApp {dapp_id}")
                self.xapp_control_callbacks[dapp_id] = [callback]
            else:
                callbacks = list(self.xapp_control_callbacks[dapp_id])
                if callback not in callbacks:
                    e3_logger.debug(f"Add additional xApp control callback for dApp {dapp_id}")
                    callbacks.append(callback)
                    self.xapp_control_callbacks[dapp_id] = callbacks
                else:
                    e3_logger.warning(f"xApp control callback already registered for dApp {dapp_id}, skipping")

    def remove_xapp_control_callback(self, dapp_id: int, subscription_id: int | None = None, callback=None):
        with self._callback_lock:
            if dapp_id in self.xapp_control_callbacks:
                if callback is None:
                    e3_logger.debug(f"Remove all xApp control callbacks for dApp {dapp_id}")
                    del self.xapp_control_callbacks[dapp_id]
                else:
                    callbacks = list(self.xapp_control_callbacks[dapp_id])
                    if callback in callbacks:
                        e3_logger.debug(f"Remove specific xApp control callback for dApp {dapp_id}")
                        callbacks.remove(callback)
                        if callbacks:
                            self.xapp_control_callbacks[dapp_id] = callbacks
                        else:
                            del self.xapp_control_callbacks[dapp_id]
                    else:
                        e3_logger.warning(f"Specific xApp control callback not found for dApp {dapp_id}")
            else:
                e3_logger.warning(f"No xApp control callbacks found for dApp {dapp_id}")

    def terminate_connections(self):
        e3_logger.info("Stop event")
        self.stop_event.set()

        # Release + stop the libe3 agent early to unblock any blocked poll_events.
        if self.agent is not None:
            try:
                self.agent.stop()
            except Exception:
                e3_logger.debug("Error stopping libe3 agent during shutdown")

        if hasattr(self, "inbound_thread") and self.inbound_thread.is_alive():
            self.inbound_thread.join(timeout=2)
            if self.inbound_thread.is_alive():
                e3_logger.warning("Inbound thread did not terminate gracefully")

        if hasattr(self, "outbound_thread") and self.outbound_thread.is_alive():
            self.outbound_thread.join(timeout=2)
            if self.outbound_thread.is_alive():
                e3_logger.warning("Outbound thread did not terminate gracefully")

    def __del__(self):
        if not self.stop_event.is_set():
            self.terminate_connections()
