import logging
import time
from datetime import datetime, timezone
from threading import Event, Lock, RLock, Thread


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PublicVersionSynchronizer:
    def __init__(
        self,
        *,
        get_active_version,
        activate_version,
        is_loaded_ready,
        event_source,
        event_channel: str,
        interval_seconds: float,
        loaded_version_id: str | None = None,
    ):
        self.get_active_version = get_active_version
        self.activate_version = activate_version
        self.is_loaded_ready = is_loaded_ready
        self.event_source = event_source
        self.event_channel = event_channel
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.state_lock = RLock()
        self.switch_lock = Lock()
        self.stop_event = Event()
        self.wake_event = Event()
        self.started = False
        self.threads: list[Thread] = []
        self.loaded_version_id = loaded_version_id
        self.remote_active_version_id: str | None = None
        self.sync_status = "initializing"
        self.last_checked_at = ""
        self.last_success_at = ""
        self.last_error = ""
        self.next_check_at = 0.0

    def ensure_current(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        with self.state_lock:
            if not force and now < self.next_check_at:
                return False
            self.next_check_at = now + self.interval_seconds
            self.sync_status = "checking"

        try:
            remote_version_id = self.get_active_version()
            with self.state_lock:
                self.remote_active_version_id = remote_version_id
                self.last_checked_at = utc_now()

            if not remote_version_id:
                with self.state_lock:
                    self.sync_status = (
                        "empty" if not self.loaded_version_id else "degraded"
                    )
                    self.last_error = (
                        ""
                        if not self.loaded_version_id
                        else "Remote active version is missing."
                    )
                return False

            with self.state_lock:
                loaded_version_id = self.loaded_version_id
            if (
                loaded_version_id == remote_version_id
                and self.is_loaded_ready()
            ):
                self._mark_synchronized(remote_version_id)
                return False

            with self.switch_lock:
                latest_version_id = self.get_active_version()
                with self.state_lock:
                    self.remote_active_version_id = latest_version_id
                    self.last_checked_at = utc_now()
                if not latest_version_id:
                    raise RuntimeError("Remote active version is missing.")

                with self.state_lock:
                    loaded_version_id = self.loaded_version_id
                if (
                    loaded_version_id == latest_version_id
                    and self.is_loaded_ready()
                ):
                    self._mark_synchronized(latest_version_id)
                    return False

                self.activate_version(latest_version_id)
                self._mark_synchronized(latest_version_id)
                logger.info(
                    "public_knowledge_base_version_synchronized",
                    extra={"version_id": latest_version_id},
                )
                return True
        except Exception as exc:
            with self.state_lock:
                self.sync_status = "degraded"
                self.last_checked_at = utc_now()
                self.last_error = str(exc) or exc.__class__.__name__
            logger.exception("public_knowledge_base_version_sync_failed")
            return False

    def _mark_synchronized(self, version_id: str) -> None:
        with self.state_lock:
            self.loaded_version_id = version_id
            self.remote_active_version_id = version_id
            self.sync_status = "synchronized"
            self.last_success_at = utc_now()
            self.last_error = ""

    def mark_loaded(self, version_id: str) -> None:
        self._mark_synchronized(version_id)

    def notify(self, version_id: str | None = None) -> None:
        with self.state_lock:
            if version_id:
                self.remote_active_version_id = version_id
                if version_id != self.loaded_version_id:
                    self.sync_status = "stale"
            self.next_check_at = 0.0
        self.wake_event.set()

    def status(self) -> dict:
        with self.state_lock:
            return {
                "status": self.sync_status,
                "remote_active_version": (
                    self.remote_active_version_id or ""
                ),
                "loaded_version": self.loaded_version_id or "",
                "last_checked_at": self.last_checked_at,
                "last_success_at": self.last_success_at,
                "last_error": self.last_error,
            }

    def start(self) -> None:
        with self.state_lock:
            if self.started:
                return
            self.started = True
            self.stop_event.clear()

        self.ensure_current(force=True)
        polling_thread = Thread(
            target=self._poll,
            name="public-version-poller",
            daemon=True,
        )
        self.threads = [polling_thread]
        polling_thread.start()

        if getattr(self.event_source, "backend_name", "") == "redis":
            event_thread = Thread(
                target=self._listen,
                name="public-version-events",
                daemon=True,
            )
            self.threads.append(event_thread)
            event_thread.start()

    def stop(self) -> None:
        with self.state_lock:
            if not self.started:
                return
            self.started = False
        self.stop_event.set()
        self.wake_event.set()
        for thread in self.threads:
            thread.join(timeout=2)
        self.threads = []

    def _poll(self) -> None:
        while not self.stop_event.is_set():
            self.wake_event.wait(self.interval_seconds)
            self.wake_event.clear()
            if not self.stop_event.is_set():
                self.ensure_current(force=True)

    def _listen(self) -> None:
        try:
            self.event_source.listen_events(
                self.event_channel,
                self.stop_event,
                self.notify,
            )
        except Exception:
            logger.exception("public_knowledge_base_version_event_listener_failed")
