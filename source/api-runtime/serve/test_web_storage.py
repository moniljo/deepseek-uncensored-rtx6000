"""Exercise browser chat storage with real IndexedDB and a mock HTTP origin.

No Strata server, GPU, live browser profile, or credentials are used. Requires
Node, Playwright, and a local Chromium/Edge executable; skips if unavailable.
Run with ``python -m unittest serve.test_web_storage -v``.
"""
from __future__ import annotations

import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "serve"
HELPER = Path(__file__).with_suffix(".js")
DEPENDENCIES = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"


class MockWeb(http.server.SimpleHTTPRequestHandler):
    """Serve the actual frontend, with inert health/monitor responses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE), **kwargs)

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = urlsplit(self.path).path
        payloads = {
            "/health": {"service": "strata", "model": "storage-test-only", "loaded": True,
                        "images": True, "max_context": 1048576, "image_generation": False},
            "/metrics": {"live": {"state": "idle"}, "engine": {"model": "storage-test-only"}},
            "/mcp": {"servers": [], "tools": 0},
            "/settings": {"shared": False},
            "/config": {},
        }
        if path in payloads:
            data = json.dumps(payloads[path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        elif path == "/":
            self.path = "/web/index.html"
            super().do_GET()
        elif path.startswith("/web/"):
            super().do_GET()
        else:
            self.send_error(404)

    def do_POST(self):
        # All chat requests must be intercepted by Playwright's inert SSE mock.
        self.send_error(501, "No backend or inference is available in this test")


class BrowserChatStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = os.environ.get("STRATA_TEST_NODE") or shutil.which("node")
        if not cls.node:
            candidate = DEPENDENCIES / "bin/node.exe"
            cls.node = str(candidate) if candidate.is_file() else None
        if not cls.node:
            raise unittest.SkipTest("Node is not installed")
        env = dict(os.environ)
        node_modules = DEPENDENCIES / "node_modules"
        if node_modules.is_dir():
            env["NODE_PATH"] = os.pathsep.join(filter(None, (str(node_modules), env.get("NODE_PATH"))))
        probe = subprocess.run(
            [cls.node, str(HELPER), "--probe"], capture_output=True, text=True,
            env=env, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if probe.returncode:
            raise unittest.SkipTest(probe.stderr.strip() or "Playwright/browser unavailable")
        cls.env = env
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockWeb)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.httpd.server_port}"
        # Keep ephemeral Edge profiles in the ignored private test area.
        profile_root = ROOT / ".strata-mcp" / "tuning"
        profile_root.mkdir(parents=True, exist_ok=True)
        cls.profiles = tempfile.TemporaryDirectory(prefix="web-storage-tests-", dir=profile_root)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls.profiles.cleanup()

    def check_scenario(self, scenario):
        profile = Path(self.profiles.name) / scenario
        result = subprocess.run(
            [self.node, str(HELPER), scenario, self.origin, str(profile)],
            capture_output=True, text=True, env=self.env, timeout=100,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["scenario"], scenario)
        self.assertTrue(receipt["passed"])

    def test_multi_megabyte_attachment_survives_reload_and_followup(self):
        self.check_scenario("attachment-roundtrip")

    def test_chat_and_file_commit_before_a_long_network_request(self):
        self.check_scenario("save-before-network")

    def test_legacy_chat_migrates_after_durable_commit(self):
        self.check_scenario("legacy-migration")

    def test_aborted_migration_keeps_legacy_history(self):
        self.check_scenario("migration-abort")

    def test_namespaces_and_immutable_queued_snapshots(self):
        self.check_scenario("namespace-snapshots")

    def test_write_failure_is_visible(self):
        self.check_scenario("visible-failure")

    def test_failed_load_cannot_silently_overwrite_unseen_history(self):
        self.check_scenario("failed-load")

    def test_shortcut_send_waits_for_delayed_history(self):
        self.check_scenario("delayed-send")

    def test_clear_and_undo_preserve_queued_history(self):
        self.check_scenario("clear-undo")

    def test_prior_undo_cannot_replace_a_fresh_chat_during_inference(self):
        self.check_scenario("undo-while-busy")

    def test_text_attachment_size_boundary(self):
        self.check_scenario("attachment-boundary")

    def test_export_waits_for_history_and_includes_file_text(self):
        self.check_scenario("delayed-export")

    def test_large_backtick_heavy_file_is_safe(self):
        self.check_scenario("backtick-file")

    def test_repeated_metrics_render_errors_are_visible_and_recover(self):
        self.check_scenario("metrics-render-error")

    def test_invalid_metrics_json_is_visible_and_recovers(self):
        self.check_scenario("metrics-invalid-json")

    def test_hung_metrics_requests_time_out_and_automatically_recover(self):
        self.check_scenario("metrics-timeout")

    def test_chat_eof_without_a_terminal_event_is_an_incomplete_error(self):
        self.check_scenario("sse-incomplete")

    def test_chat_finish_reason_without_done_is_accepted(self):
        self.check_scenario("sse-finish-only")

    def test_chat_done_without_finish_reason_is_accepted(self):
        self.check_scenario("sse-done-only")

    def test_user_abort_is_stopped_without_an_incomplete_error(self):
        self.check_scenario("sse-user-abort")


if __name__ == "__main__":
    unittest.main()
