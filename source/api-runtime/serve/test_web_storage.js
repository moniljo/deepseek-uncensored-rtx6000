// Real IndexedDB tests for the unmodified web app, isolated from production.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

function runtime() {
  let playwright;
  try { playwright = require("playwright"); }
  catch (_) { throw new Error("Playwright is not installed; no dependencies were downloaded"); }
  const candidates = [process.env.STRATA_TEST_BROWSER, playwright.chromium.executablePath(),
    process.env["ProgramFiles(x86)"] && path.join(process.env["ProgramFiles(x86)"], "Microsoft/Edge/Application/msedge.exe"),
    process.env.ProgramFiles && path.join(process.env.ProgramFiles, "Microsoft/Edge/Application/msedge.exe"),
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"].filter(Boolean);
  const executablePath = candidates.find((p) => fs.existsSync(p));
  if (!executablePath) throw new Error("No installed Chromium/Edge executable; no browser was downloaded");
  return {playwright, executablePath};
}

// Seed fake user history before app startup. This contains no credentials.
const history = [{role: "user", text: "An earlier question", time: 1,
  files: [{name: "source.txt", text: "Retained attachment: amber lantern"}], images: []},
  {role: "assistant", text: "An earlier answer", time: 2}];

async function main() {
  const {playwright, executablePath} = runtime();
  const [scenario, origin, profile] = process.argv.slice(2);
  if (scenario === "--probe") { console.log(JSON.stringify({executablePath})); return; }
  const context = await playwright.chromium.launchPersistentContext(profile, {
    executablePath, headless: true, acceptDownloads: true,
    args: ["--disable-gpu", "--disable-background-networking", "--no-first-run", "--disable-sync"],
  });
  const requests = [], errors = [];
  let metricsMode = "normal", metricsInjected = 0;
  const heldMetrics = new Set();
  let releaseReply;
  const replyGate = ["save-before-network", "undo-while-busy"].includes(scenario) ?
    new Promise((resolve) => { releaseReply = resolve; }) : null;
  let page;
  try {
    await context.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.origin !== origin) { await route.abort(); return; }
      if (url.pathname === "/metrics" && metricsMode === "invalid-json") {
        metricsInjected++;
        await route.fulfill({status: 200, contentType: "application/json", body: "{not valid JSON"});
      } else if (url.pathname === "/metrics" && metricsMode === "hang") {
        metricsInjected++;
        await new Promise((resolve) => heldMetrics.add(resolve));
        // The browser may already have aborted this intercepted request.
        try { await route.abort(); } catch (_) { /* request already closed */ }
      } else if (url.pathname === "/v1/chat/completions") {
        requests.push(route.request().postDataJSON());
        if (replyGate) await replyGate;           // leave this inert request pending until durability is checked
        const data = {choices: [{delta: {content: "Storage test response."}}]};
        const ending = scenario === "sse-incomplete" ? "" : scenario === "sse-finish-only" ?
          `data: ${JSON.stringify({choices: [{delta: {}, finish_reason: "stop"}]})}` :
          scenario === "sse-done-only" ? "data: [DONE]" : "data: [DONE]\n\n";
        await route.fulfill({status: 200, contentType: "text/event-stream",
          body: `data: ${JSON.stringify(data)}\n\n${ending}`});
      } else await route.continue();
    });
    page = context.pages()[0] || await context.newPage();
    page.on("pageerror", (e) => errors.push(e.stack || String(e)));
    const ready = async () => {
      await page.waitForFunction(() => typeof createChatStorage === "function" && typeof chatReady !== "undefined");
      await page.evaluate(() => chatReady);
    };
    const open = async (suffix = "") => { await page.goto(origin + "/" + suffix); await ready(); };
    const stored = async (prefix = "strata.", id = "current") => page.evaluate(
      async ([p, i]) => createChatStorage(p, i).load(), [prefix, id]);
    const waitRequests = async (n) => {
      const until = Date.now() + 10000;
      while (requests.length < n && Date.now() < until) await new Promise((r) => setTimeout(r, 20));
      assert.equal(requests.length, n, "Expected inert mocked chat request");
      await page.waitForFunction(() => busy === null);
    };
    const seed = async (extra = {}) => context.addInitScript(({history, extra}) => {
      if (!sessionStorage.getItem("storage-test-seeded")) {
        localStorage.setItem("strata.chat", JSON.stringify(history));
        sessionStorage.setItem("storage-test-seeded", "yes");
      }
      Object.assign(window, extra);
    }, {history, extra});
    const delayOpen = async () => context.addInitScript(() => {
      const original = IDBFactory.prototype.open;
      IDBFactory.prototype.open = function (...args) {
        const native = original.apply(this, args), delayed = {};
        for (const key of ["result", "error", "readyState", "transaction"])
          Object.defineProperty(delayed, key, {get: () => native[key]});
        native.onupgradeneeded = (event) => delayed.onupgradeneeded?.call(delayed, event);
        native.onerror = (event) => delayed.onerror?.call(delayed, event);
        native.onblocked = (event) => delayed.onblocked?.call(delayed, event);
        native.onsuccess = (event) => setTimeout(() => delayed.onsuccess?.call(delayed, event), 700);
        return delayed;
      };
    });

    if (scenario === "metrics-render-error") {
      await open();
      await page.waitForFunction(() => lastMetrics !== null);
      await page.evaluate(() => {
        window.__workingRender = render;
        window.__lastGoodMetrics = lastMetrics;
        window.__renderFailures = 0;
        render = () => { window.__renderFailures++; throw new Error("Synthetic transient renderer failure"); };
      });
      await page.waitForFunction(() => window.__renderFailures >= 3 &&
        document.getElementById("pill-text").textContent === "Metrics display error", null, {timeout: 10000});
      assert((await page.evaluate(() => window.__renderFailures)) >= 3,
        "Repeated HTTP 200 responses must not reset the consecutive rendering failure count");
      assert((await page.evaluate(() => metricsFailures)) >= 3, "Renderer failures must accumulate in the real retry counter");
      assert.equal(await page.evaluate(() => lastMetrics === window.__lastGoodMetrics), true,
        "A failed rendering pass must not publish its metrics snapshot");
      await page.evaluate(() => { render = window.__workingRender; });
      await page.waitForFunction(() => document.getElementById("pill-text").textContent === "Idle", null, {timeout: 5000});
      assert.equal(await page.evaluate(() => metricsFailures), 0, "Successful rendering must reset the retry counter");
    } else if (scenario === "metrics-invalid-json") {
      await open();
      await page.waitForFunction(() => lastMetrics !== null);
      await page.evaluate(() => { window.__lastGoodMetrics = lastMetrics; });
      metricsMode = "invalid-json";
      await page.waitForFunction(() => document.getElementById("pill-text").textContent === "Metrics data error", null, {timeout: 10000});
      const until = Date.now() + 10000;
      while (metricsInjected < 3 && Date.now() < until) await new Promise((resolve) => setTimeout(resolve, 20));
      assert(metricsInjected >= 3, "Invalid HTTP 200 JSON must accumulate consecutive data failures");
      assert((await page.evaluate(() => metricsFailures)) >= 3);
      assert.equal(await page.evaluate(() => lastMetrics === window.__lastGoodMetrics), true);
      metricsMode = "normal";
      await page.waitForFunction(() => document.getElementById("pill-text").textContent === "Idle", null, {timeout: 5000});
      assert.equal(await page.evaluate(() => metricsFailures), 0);
    } else if (scenario === "metrics-timeout") {
      await context.addInitScript(() => {
        const nativeTimeout = window.setTimeout;
        window.setTimeout = (callback, ms, ...args) => nativeTimeout(callback,
          ms === 10000 && typeof callback === "function" && /\.abort\(/.test(String(callback)) ? 100 : ms, ...args);
      });
      await open();
      await page.waitForFunction(() => lastMetrics !== null);
      await page.evaluate(() => {
        window.__metricsAborts = 0;
        window.__lastGoodMetrics = lastMetrics;
        const nativeFetch = window.fetch;
        window.fetch = function (input, options) {
          if (new URL(input, location.href).pathname === "/metrics" && options?.signal)
            options.signal.addEventListener("abort", () => window.__metricsAborts++, {once: true});
          return nativeFetch.call(this, input, options);
        };
      });
      metricsMode = "hang";
      await page.waitForFunction(() => window.__metricsAborts >= 3, null, {timeout: 10000});
      assert(metricsInjected >= 3, "Each hung request must be aborted and retried automatically");
      assert((await page.evaluate(() => metricsFailures)) >= 3);
      assert.equal(await page.evaluate(() => lastMetrics === window.__lastGoodMetrics), true);
      const failure = await page.locator("#pill-text").innerText();
      assert.match(failure, /metrics|server|timed|reachable/i);
      assert.notEqual(failure, "Idle");
      metricsMode = "normal";
      for (const release of heldMetrics) release();
      heldMetrics.clear();
      await page.waitForFunction(() => document.getElementById("pill-text").textContent === "Idle", null, {timeout: 5000});
      assert.equal(await page.evaluate(() => metricsFailures), 0);
    } else if (["sse-incomplete", "sse-finish-only", "sse-done-only"].includes(scenario)) {
      await open();
      await page.locator("#input").fill("Synthetic stream completion check");
      await page.locator("#send-btn").click();
      await waitRequests(1);
      await page.waitForFunction(() => !document.getElementById("chat-storage-status").textContent);
      const answer = await page.evaluate(() => messages.at(-1));
      assert.equal(answer.text, "Storage test response.");
      assert.equal(!!answer.stopped, false);
      if (scenario === "sse-incomplete") {
        assert.match(answer.error || "", /incomplete|ended|finish|interrupt|complete/i,
          "A bare EOF after partial content must be reported as an incomplete answer");
        assert((await page.locator(".st-msg--assistant").innerText()).includes(answer.error));
        assert((await page.locator(".st-msg--assistant").innerText()).includes(answer.text));
        assert.equal((await stored()).at(-1).error, answer.error, "Incomplete error must survive persistence");
        await page.reload(); await ready();
        const restored = await page.evaluate(() => messages.at(-1));
        assert.equal(restored.error, answer.error);
        assert.equal(restored.text, answer.text);
        assert((await page.locator(".st-msg--assistant").innerText()).includes(answer.error));
        assert((await page.locator(".st-msg--assistant").innerText()).includes(answer.text));
      } else {
        assert.equal(answer.error, undefined, "Either a finish reason or DONE is a valid stream terminal signal");
        assert.equal((await stored()).at(-1).text, answer.text);
      }
    } else if (scenario === "sse-user-abort") {
      await open();
      await page.evaluate(() => {
        const nativeFetch = window.fetch;
        window.__mockChatPosts = 0;
        window.__mockChatAborted = false;
        window.fetch = function (input, options) {
          if (new URL(input, location.href).pathname !== "/v1/chat/completions")
            return nativeFetch.call(this, input, options);
          window.__mockChatPosts++;
          const body = new ReadableStream({start(controller) {
            controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"A partial answer before Stop."}}]}\n\n'));
            options.signal.addEventListener("abort", () => {
              window.__mockChatAborted = true;
              controller.error(new DOMException("Synthetic user abort", "AbortError"));
            }, {once: true});
          }});
          return Promise.resolve(new Response(body, {status: 200, headers: {"Content-Type": "text/event-stream"}}));
        };
      });
      await page.locator("#input").fill("Synthetic user cancellation check");
      await page.locator("#send-btn").click();
      await page.waitForFunction(() => busy !== null && messages.at(-1).text === "A partial answer before Stop.");
      await page.locator("#stop-btn").click();
      await page.waitForFunction(() => busy === null && !document.getElementById("chat-storage-status").textContent);
      const answer = await page.evaluate(() => messages.at(-1));
      assert.equal(await page.evaluate(() => window.__mockChatPosts), 1);
      assert.equal(await page.evaluate(() => window.__mockChatAborted), true);
      assert.equal(answer.stopped, true);
      assert.equal(answer.error, undefined, "A deliberate user Stop must not become an incomplete-stream error");
      assert.equal(answer.text, "A partial answer before Stop.");
      assert.equal((await stored()).at(-1).stopped, true);
    } else if (scenario === "attachment-roundtrip") {
      await open();
      const length = 3 * 1024 * 1024;
      await page.evaluate((n) => addFiles([new File(["DATA:" + "x".repeat(n) + ":END"], "large.txt", {type: "text/plain"})]), length);
      await page.waitForFunction(() => attachments.length === 1);
      await page.locator("#input").fill("Keep this source for my next question");
      await page.locator("#send-btn").click();
      await waitRequests(1);
      assert(requests[0].messages[0].content.includes("DATA:" + "x".repeat(100)));
      const saved = await stored();
      assert.equal(saved[0].files[0].text.length, length + 9);
      await page.reload(); await ready();
      assert.equal(await page.evaluate(() => messages[0].files[0].text.length), length + 9);
      await page.locator("#input").fill("What did the source say?");
      await page.locator("#send-btn").click();
      await waitRequests(2);
      assert(requests[1].messages[0].content.endsWith(":END\n```"));
      assert.equal((await stored())[0].files[0].text.length, length + 9);
    } else if (scenario === "save-before-network") {
      await open();
      await page.evaluate(() => {
        window.__storageEvents = [];
        const transaction = IDBDatabase.prototype.transaction;
        IDBDatabase.prototype.transaction = function (...args) {
          const tx = transaction.apply(this, args);
          if (args[1] === "readwrite") tx.addEventListener("complete", () => window.__storageEvents.push("write committed"));
          return tx;
        };
        const nativeFetch = window.fetch;
        window.fetch = function (input, options) {
          if (new URL(input, location.href).pathname === "/v1/chat/completions") {
            window.__eventsAtFetch = window.__storageEvents.slice();
            window.__storageEvents.push("fetch started");
          }
          return nativeFetch.call(this, input, options);
        };
        addFiles([new File(["DURABLE-SOURCE:" + "q".repeat(1024 * 1024)], "pending-source.txt", {type: "text/plain"})]);
      });
      await page.waitForFunction(() => attachments.length === 1);
      await page.locator("#input").fill("A long request must keep its source even before the answer arrives");
      await page.locator("#send-btn").click();
      await page.waitForFunction(() => !!window.__eventsAtFetch);
      assert.deepEqual(await page.evaluate(() => window.__eventsAtFetch), ["write committed"],
        "The actual IndexedDB transaction must complete before chat fetch begins");
      const saved = await stored();
      assert.equal(saved[0].text, "A long request must keep its source even before the answer arrives");
      assert.equal(saved[0].files[0].name, "pending-source.txt");
      assert.equal(saved[0].files[0].text, "DURABLE-SOURCE:" + "q".repeat(1024 * 1024));
      assert.equal(await page.evaluate(() => busy !== null && messages.at(-1).text === ""), true,
        "Durability is checked while inference is still pending, not after answer completion");
      releaseReply();
      await waitRequests(1);
    } else if (scenario === "legacy-migration") {
      const fakeKey = "fake-storage-test-key";
      const sampling = {thinking: "low", temperature: 0.25, top_p: 0.8, top_k: 17,
        max: "64", seed: "123", show: false, esp: false, mcp: false};
      await seed();
      await context.addInitScript(({fakeKey, sampling}) => {
        if (!sessionStorage.getItem("storage-test-settings-seeded")) {
          localStorage.setItem("strata.apikey", JSON.stringify(fakeKey));
          localStorage.setItem("strata.sampling", JSON.stringify(sampling));
          sessionStorage.setItem("storage-test-settings-seeded", "yes");
        }
      }, {fakeKey, sampling});
      await open();
      assert.deepEqual(await stored(), history);
      assert.equal(await page.evaluate(() => localStorage.getItem("strata.chat")), null);
      await page.reload(); await ready();
      assert.equal(await page.evaluate(() => messages[0].files[0].text), history[0].files[0].text);
      assert.equal(await page.locator("#api-key").inputValue(), fakeKey);
      assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem("strata.apikey"))), fakeKey);
      assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem("strata.sampling"))), sampling);
      assert.deepEqual(await page.evaluate(() => settings), sampling);
    } else if (scenario === "migration-abort") {
      await seed();
      await context.addInitScript(() => {
        const put = IDBObjectStore.prototype.put;
        IDBObjectStore.prototype.put = function (...args) {
          const request = put.apply(this, args), transaction = this.transaction;
          request.addEventListener("success", () => transaction.abort(), {once: true});
          return request;
        };
      });
      await open();
      assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem("strata.chat"))), history,
        "An aborted transaction must retain the recoverable legacy copy");
      assert.equal(await page.evaluate(() => messages[0].files[0].text), history[0].files[0].text);
      assert(await page.locator("#chat-storage-status").isVisible());
    } else if (scenario === "namespace-snapshots") {
      await open();
      const result = await page.evaluate(async () => {
        const a = createChatStorage("strata.session.alpha."), b = createChatStorage("strata.session.beta.");
        await Promise.all([a.load(), b.load()]);
        const input = [{role: "user", text: "alpha", files: [{name: "a.txt", text: "source-alpha"}]}];
        const first = a.save(input);
        input[0].files[0].text = "mutated after enqueue";
        await first;
        await b.save([{role: "user", text: "beta"}]);
        const snapshot = await a.load();
        await Promise.all([a.save([{role: "user", text: "older queued save"}]), a.reset([]), a.reset(snapshot)]);
        const prefix = "strata.session.same-prefix.";
        const legacy = [{role: "user", text: "legacy current chat"}];
        localStorage.setItem(prefix + "chat", JSON.stringify(legacy));
        const archive = createChatStorage(prefix, "archive"), draft = createChatStorage(prefix, "draft");
        const emptyArchive = await archive.load();
        const legacyAfterRead = JSON.parse(localStorage.getItem(prefix + "chat"));
        await archive.save([{role: "user", text: "archived conversation"}]);
        await draft.save([{role: "user", text: "draft conversation"}]);
        const legacyAfterWrite = JSON.parse(localStorage.getItem(prefix + "chat"));
        const current = await createChatStorage(prefix, "current").load();
        return {a: await a.load(), b: await b.load(), emptyArchive, legacy, legacyAfterRead, legacyAfterWrite,
          archive: await archive.load(), draft: await draft.load(), current,
          legacyAfterCurrentMigration: localStorage.getItem(prefix + "chat")};
      });
      assert.equal(result.a[0].files[0].text, "source-alpha");
      assert.equal(result.b[0].text, "beta");
      assert.deepEqual(result.emptyArchive, [], "A noncurrent chat must not inherit the legacy current chat");
      assert.deepEqual(result.legacyAfterRead, result.legacy);
      assert.deepEqual(result.legacyAfterWrite, result.legacy, "A noncurrent save must not delete the legacy current chat");
      assert.equal(result.archive[0].text, "archived conversation");
      assert.equal(result.draft[0].text, "draft conversation");
      assert.deepEqual(result.current, result.legacy);
      assert.equal(result.legacyAfterCurrentMigration, null);
      assert.deepEqual(await stored(), []);
    } else if (scenario === "visible-failure") {
      await open();
      await page.evaluate(() => {
        IDBObjectStore.prototype.put = function () { throw new DOMException("Simulated full storage", "QuotaExceededError"); };
      });
      await page.locator("#input").fill("A message that cannot be saved");
      await page.locator("#send-btn").click();
      await page.waitForFunction(() => !document.getElementById("chat-storage-status").hidden &&
        document.getElementById("chat-storage-status").textContent.trim().length > 0);
      assert(await page.locator("#chat-storage-status").isVisible());
      assert.match(await page.locator("#chat-storage-status").innerText(), /sav|stor|export|reload/i);
    } else if (scenario === "failed-load") {
      await open();
      const result = await page.evaluate(async () => {
        const prefix = "strata.session.read-failure.";
        const initial = createChatStorage(prefix);
        await initial.load(); await initial.save([{role: "user", text: "unseen saved history"}]);
        const get = IDBObjectStore.prototype.get;
        IDBObjectStore.prototype.get = function () { throw new DOMException("Simulated read error", "UnknownError"); };
        const broken = createChatStorage(prefix);
        try { await broken.load(); } catch (_) { /* Either rejection or warned empty load is acceptable. */ }
        IDBObjectStore.prototype.get = get;
        let rejected = false;
        try { await broken.save([{role: "user", text: "would clobber history"}]); } catch (_) { rejected = true; }
        const preserved = await createChatStorage(prefix).load();
        await broken.reset([]);
        return {rejected, preserved, cleared: await createChatStorage(prefix).load()};
      });
      assert.equal(result.rejected, true);
      assert.equal(result.preserved[0].text, "unseen saved history");
      assert.deepEqual(result.cleared, []);
    } else if (scenario === "delayed-send") {
      await seed(); await delayOpen();
      await page.goto(origin + "/?q=A%20fresh%20question", {waitUntil: "domcontentloaded"});
      await page.waitForTimeout(120);
      assert.equal(requests.length, 0, "Shortcut must not run before persisted history is loaded");
      await ready(); await waitRequests(1);
      assert.equal(requests[0].messages[0].content.includes("amber lantern"), true);
      assert.equal(requests[0].messages.at(-1).content, "A fresh question");
    } else if (scenario === "clear-undo") {
      await seed(); await open();
      await page.evaluate(() => { saveChat(); document.getElementById("new-btn").click(); });
      await page.getByRole("button", {name: "Undo", exact: true}).click();
      await page.waitForFunction(() => messages.length === 2);
      assert.deepEqual(await stored(), history);
      await page.reload(); await ready();
      assert.deepEqual(await page.evaluate(() => messages), history);
    } else if (scenario === "undo-while-busy") {
      await seed(); await open();
      await page.locator("#new-btn").click();
      await page.waitForFunction(() => messages.length === 0);
      await page.evaluate(() => addFiles([new File(["Fresh source survives an old Undo action"], "fresh.txt", {type: "text/plain"})]));
      await page.waitForFunction(() => attachments.length === 1);
      await page.locator("#input").fill("This is the fresh conversation");
      // The New chat toast overlaps the Send button at this viewport; Enter
      // starts the request while the real Undo action is still available.
      await page.locator("#input").press("Enter");
      await page.waitForFunction(() => busy !== null && !document.getElementById("chat-storage-status").textContent);
      const fresh = await page.evaluate(() => messages);
      assert.equal(fresh[0].text, "This is the fresh conversation");
      assert.equal(fresh[1].text, "");
      assert.deepEqual(await stored(), fresh);
      await page.getByRole("button", {name: "Undo", exact: true}).click();
      assert((await page.locator("#toasts").innerText()).includes("Stop the answer first."));
      assert.equal(await page.evaluate(() => busy !== null), true);
      assert.deepEqual(await page.evaluate(() => messages), fresh,
        "An old Undo action must not replace the fresh conversation while its answer is pending");
      assert.deepEqual(await stored(), fresh, "The rejected Undo must not overwrite the durable fresh chat");
      releaseReply();
      await waitRequests(1);
      await page.waitForFunction(() => !document.getElementById("chat-storage-status").textContent);
      const completed = await stored();
      assert.equal(completed.length, 2);
      assert.equal(completed[0].text, "This is the fresh conversation");
      assert.equal(completed[0].files[0].text, "Fresh source survives an old Undo action");
      assert.equal(completed[1].text, "Storage test response.");
    } else if (scenario === "attachment-boundary") {
      await open();
      await page.evaluate(() => addFiles([new File(["x".repeat(16 * 1024 * 1024)], "boundary.txt", {type: "text/plain"})]));
      await page.waitForFunction(() => attachments.length === 1);
      assert.equal(await page.evaluate(() => attachments[0].text.length), 16 * 1024 * 1024);
      await page.evaluate(() => addFiles([new File(["x".repeat(16 * 1024 * 1024 + 1)], "oversize.txt", {type: "text/plain"})]));
      assert.equal(await page.evaluate(() => attachments.length), 1);
      assert.match(await page.locator("#toasts").innerText(), /oversize\.txt/);
    } else if (scenario === "delayed-export") {
      await seed(); await delayOpen();
      await context.addInitScript(() => {
        const create = URL.createObjectURL;
        URL.createObjectURL = (blob) => { window.__exportBlob = blob; return create.call(URL, blob); };
        const click = HTMLAnchorElement.prototype.click;
        HTMLAnchorElement.prototype.click = function () { if (!this.download) return click.call(this); };
      });
      await page.goto(origin + "/", {waitUntil: "domcontentloaded"});
      await page.locator("#export-btn").click();
      await page.waitForFunction(() => !!window.__exportBlob);
      const exported = await page.evaluate(() => window.__exportBlob.text());
      assert(exported.includes("source.txt"));
      assert(exported.includes("Retained attachment: amber lantern"));
      assert(exported.includes("An earlier answer"));
    } else if (scenario === "backtick-file") {
      await open();
      const result = await page.evaluate(() => {
        const text = "`x".repeat(400000) + "\n````\nlast line";
        return {textLength: text.length, block: fileBlock({name: "ticks.txt", text})};
      });
      assert(result.block.startsWith("File: ticks.txt\n`````\n"));
      assert(result.block.endsWith("last line\n`````"));
      assert(result.block.length > result.textLength);
    } else throw new Error(`Unknown test scenario: ${scenario}`);
    assert.deepEqual(errors, [], "The app must not raise uncaught browser errors");
    console.log(JSON.stringify({scenario, passed: true}));
  } finally {
    releaseReply?.();
    for (const release of heldMetrics) release();
    await context.close();
  }
}

main().catch((error) => { console.error(error.stack || String(error)); process.exitCode = 1; });
