import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_CORS_ORIGINS,
  KeyedAsyncQueue,
  isLoopbackHost,
  resolveCorsOrigins,
  validateBody,
} from "../core.mjs";

test("resolveCorsOrigins uses defaults and env override", () => {
  assert.deepEqual(resolveCorsOrigins(""), DEFAULT_CORS_ORIGINS);
  assert.deepEqual(resolveCorsOrigins("http://a.test, http://b.test "), [
    "http://a.test",
    "http://b.test",
  ]);
});

test("isLoopbackHost recognizes loopback variants", () => {
  assert.equal(isLoopbackHost("127.0.0.1"), true);
  assert.equal(isLoopbackHost("localhost"), true);
  assert.equal(isLoopbackHost("::1"), true);
  assert.equal(isLoopbackHost("0.0.0.0"), false);
});

test("validateBody rejects missing required fields and unknown fields", () => {
  assert.throws(
    () => validateBody("jobSave", {}),
    (err) => err?.statusCode === 400 && err?.field === "name",
  );
  assert.throws(
    () => validateBody("targetSave", { name: "local", extra: true }),
    (err) => err?.statusCode === 400 && err?.field === "extra",
  );
  assert.throws(
    () => validateBody("connectorTest", { params: [] }),
    (err) => err?.statusCode === 400 && err?.field === "params",
  );
  assert.throws(
    () => validateBody("registryUrlsPut", { urls: ["", "http://ok.test"] }),
    (err) => err?.statusCode === 400 && err?.field === "urls",
  );
});

test("KeyedAsyncQueue serializes work for the same key", async () => {
  const queue = new KeyedAsyncQueue();
  const events = [];
  await Promise.all([
    queue.run("project", async () => {
      events.push("start-1");
      await new Promise((resolve) => setTimeout(resolve, 30));
      events.push("end-1");
    }),
    queue.run("project", async () => {
      events.push("start-2");
      await new Promise((resolve) => setTimeout(resolve, 5));
      events.push("end-2");
    }),
  ]);
  assert.deepEqual(events, ["start-1", "end-1", "start-2", "end-2"]);
});
