"""The custom web app uses edited keys immediately and validates Apply before saving.

Runs the page's storage and server-access code with Node, without a GPU or server:
    python -m unittest serve.test_web_api_key -v
"""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

APP = Path(__file__).with_name("web") / "app.js"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebApiKey(unittest.TestCase):
    def run_js(self, actions, *, stored=None, blocked=False, quota=False, status=200, session=""):
        src = APP.read_text(encoding="utf-8")
        store = src[src.index("const store ="):src.index("// ------------------------------------------------------------------ toasts")]
        access = src[src.index("// ------------------------------------------------------------------ server access"):src.index("let health =")]
        js = f"""
(async () => {{
const data = new Map(Object.entries({json.dumps(stored or {})}));
const elements = new Map();
const $ = (id) => {{
  if (!elements.has(id)) elements.set(id, {{value: "", textContent: "", type: "password", disabled: false,
    attributes: {{}}, setAttribute(k, v) {{ this.attributes[k] = String(v); }} }});
  return elements.get(id);
}};
const field = $("api-key");
const notices = [], calls = [];
const sessionId = {json.dumps(session)};
const storePrefix = sessionId ? `strata.session.${{sessionId}}.` : "strata.";
const window = {{}};
const location = {{pathname: "/"}};
let keyWarned = true;
const toast = (...args) => notices.push(args);
const fetch = async (url, options = {{}}) => {{
  calls.push({{url, ...options}});
  return {{ok: {status} >= 200 && {status} < 300, status: {status}}};
}};
const localStorage = {{
  getItem(k) {{
    if ({json.dumps(blocked)}) throw new Error("Storage blocked");
    return data.has(k) ? data.get(k) : null;
  }},
  setItem(k, v) {{
    if ({json.dumps(blocked or quota)}) throw new Error("Storage unavailable");
    data.set(k, v);
  }},
}};
function edit(value, event = "input") {{
  field.value = value;
  const handler = field["on" + event];
  if (handler) handler();
}}
{store}
{access}
{actions}
}})().catch(error => {{ console.error(error.stack); process.exitCode = 1; }});
"""
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_input_is_used_without_blur_or_premature_save(self):
        got = self.run_js('''
edit("  test-key  ");
console.log(JSON.stringify({headers: headers(true), saved: data.get("strata.apikey") ?? null,
  notices: notices.length, checks: calls.length}));
''')
        self.assertEqual(got, {"headers": {"X-API-Key": "test-key", "Content-Type": "application/json"},
                               "saved": None, "notices": 0, "checks": 0})

    def test_saved_key_is_restored_on_reload(self):
        got = self.run_js('console.log(JSON.stringify({value: field.value, headers: headers()}));',
                          stored={"strata.apikey": '"saved-key"'})
        self.assertEqual(got, {"value": "saved-key", "headers": {"X-API-Key": "saved-key"}})

    def test_blocked_storage_still_uses_key_in_memory(self):
        got = self.run_js('edit("test-key"); await applyApiKey(); console.log(JSON.stringify(headers()));', blocked=True)
        self.assertEqual(got, {"X-API-Key": "test-key"})

    def test_failed_write_does_not_reuse_old_key(self):
        got = self.run_js('''edit("new-key"); await applyApiKey();
console.log(JSON.stringify({headers: headers(), saved: data.get("strata.apikey")}));''',
                          stored={"strata.apikey": '"old-key"'}, quota=True)
        self.assertEqual(got, {"headers": {"X-API-Key": "new-key"}, "saved": '"old-key"'})

    def test_clearing_key_removes_authorization(self):
        for quota in (False, True):
            with self.subTest(quota=quota):
                got = self.run_js('edit("  "); console.log(JSON.stringify(headers()));',
                                  stored={"strata.apikey": '"old-key"'}, quota=quota)
                self.assertEqual(got, {})

    def test_successful_apply_validates_and_notifies_after_typing(self):
        got = self.run_js('''
edit("test-key");
const before = notices.length;
await $("api-key-apply").onclick();
console.log(JSON.stringify({before, kinds: notices.map(n => n[0]), calls,
  saved: data.get("strata.apikey"), status: $("api-key-status").textContent,
  disabled: $("api-key-apply").disabled, warned: keyWarned}));
''')
        self.assertEqual(got["before"], 0)
        self.assertEqual(got["kinds"], ["success"])
        self.assertEqual(got["calls"], [{"url": "v1/models", "cache": "no-store", "headers": {"X-API-Key": "test-key"}}])
        self.assertEqual(got["saved"], '"test-key"')
        self.assertIn("accepted and saved", got["status"])
        self.assertFalse(got["disabled"])
        self.assertFalse(got["warned"])

    def test_failed_save_truthfully_reports_tab_only_acceptance(self):
        got = self.run_js('''edit("test-key"); await applyApiKey();
console.log(JSON.stringify({status: $("api-key-status").textContent, notices,
  saved: data.get("strata.apikey") ?? null}));''',
                          blocked=True)
        self.assertIsNone(got["saved"])
        self.assertIn("accepted for this tab", got["status"])
        self.assertIn("storage is unavailable", got["status"])
        self.assertNotIn("and saved", got["status"])
        self.assertTrue(all("saved" not in " ".join(map(str, notice)).lower() for notice in got["notices"]))

    def test_rejected_apply_preserves_the_last_stored_key(self):
        got = self.run_js('''edit("rejected-key"); await applyApiKey();
console.log(JSON.stringify({saved: data.get("strata.apikey"), headers: headers(),
  status: $("api-key-status").textContent, notices, disabled: $("api-key-apply").disabled}));''',
                          stored={"strata.apikey": '"previously-accepted-key"'}, status=401)
        self.assertEqual(got["saved"], '"previously-accepted-key"')
        self.assertEqual(got["headers"], {"X-API-Key": "rejected-key"})
        self.assertIn("Key not accepted", got["status"])
        self.assertEqual(got["notices"], [])
        self.assertFalse(got["disabled"])

    def test_session_key_is_restored_and_saved_separately(self):
        got = self.run_js('''const restored = field.value;
edit("new-macbook-key"); await applyApiKey();
console.log(JSON.stringify({restored, headers: headers(), original: data.get("strata.apikey"),
  saved: data.get("strata.session.macbook.apikey")}));''',
                          stored={"strata.apikey": '"original-key"',
                                  "strata.session.macbook.apikey": '"macbook-key"'}, session="macbook")
        self.assertEqual(got, {"restored": "macbook-key", "headers": {"X-API-Key": "new-macbook-key"},
                               "original": '"original-key"', "saved": '"new-macbook-key"'})

    def test_server_without_key_keeps_json_header_only(self):
        got = self.run_js('console.log(JSON.stringify(headers(true)));')
        self.assertEqual(got, {"Content-Type": "application/json"})


if __name__ == "__main__":
    unittest.main()
