import unittest

from backend.version_sync import PublicVersionSynchronizer


class EventSource:
    backend_name = "memory"


class PublicVersionSynchronizerTests(unittest.TestCase):
    def create_synchronizer(
        self,
        state,
        activated,
        *,
        loaded_version_id="version-1",
    ):
        return PublicVersionSynchronizer(
            get_active_version=lambda: state["active"],
            activate_version=lambda version_id: activated.append(version_id),
            is_loaded_ready=lambda: state["ready"],
            event_source=EventSource(),
            event_channel="versions",
            interval_seconds=60,
            loaded_version_id=loaded_version_id,
        )

    def test_two_instances_converge_on_the_same_remote_version(self):
        state = {"active": "version-2", "ready": True}
        first_activated = []
        second_activated = []
        first = self.create_synchronizer(state, first_activated)
        second = self.create_synchronizer(state, second_activated)

        self.assertTrue(first.ensure_current(force=True))
        self.assertTrue(second.ensure_current(force=True))

        self.assertEqual(first_activated, ["version-2"])
        self.assertEqual(second_activated, ["version-2"])
        self.assertEqual(first.status()["loaded_version"], "version-2")
        self.assertEqual(second.status()["loaded_version"], "version-2")
        self.assertEqual(first.status()["status"], "synchronized")
        self.assertEqual(second.status()["status"], "synchronized")

    def test_failed_switch_keeps_the_previous_loaded_version(self):
        state = {"active": "version-2", "ready": True}

        def fail_activation(_version_id):
            raise RuntimeError("download failed")

        synchronizer = PublicVersionSynchronizer(
            get_active_version=lambda: state["active"],
            activate_version=fail_activation,
            is_loaded_ready=lambda: state["ready"],
            event_source=EventSource(),
            event_channel="versions",
            interval_seconds=60,
            loaded_version_id="version-1",
        )

        self.assertFalse(synchronizer.ensure_current(force=True))

        status = synchronizer.status()
        self.assertEqual(status["status"], "degraded")
        self.assertEqual(status["loaded_version"], "version-1")
        self.assertEqual(status["remote_active_version"], "version-2")
        self.assertEqual(status["last_error"], "download failed")

    def test_event_invalidates_the_poll_cache(self):
        state = {"active": "version-1", "ready": True}
        activated = []
        synchronizer = self.create_synchronizer(state, activated)
        synchronizer.ensure_current(force=True)
        state["active"] = "version-2"

        synchronizer.notify("version-2")
        self.assertEqual(synchronizer.status()["status"], "stale")
        self.assertTrue(synchronizer.ensure_current())

        self.assertEqual(activated, ["version-2"])
        self.assertEqual(
            synchronizer.status()["remote_active_version"],
            "version-2",
        )


if __name__ == "__main__":
    unittest.main()
