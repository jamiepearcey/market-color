import test from "node:test";
import assert from "node:assert/strict";
import { classifyEngineFailure } from "../engine-errors.mjs";

test("classifyEngineFailure maps config validation stderr", () => {
  const err = classifyEngineFailure({ stderr: "validation failed: missing required field host" });
  assert.equal(err.code, "CONFIG_INVALID");
  assert.equal(err.statusCode, 400);
});

test("classifyEngineFailure maps missing plugin stderr", () => {
  const err = classifyEngineFailure({ stderr: "unknown connector source-demo" });
  assert.equal(err.code, "CONNECTOR_NOT_FOUND");
  assert.equal(err.statusCode, 404);
});

test("classifyEngineFailure maps not-installed stderr", () => {
  const err = classifyEngineFailure({ stderr: "plugin source-demo is not installed; run celeritas add source-demo" });
  assert.equal(err.code, "NEEDS_INSTALL");
  assert.equal(err.statusCode, 409);
});

test("classifyEngineFailure maps connectivity stderr", () => {
  const err = classifyEngineFailure({ stderr: "dial tcp 127.0.0.1:5432: connection refused" });
  assert.equal(err.code, "CONNECTIVITY");
  assert.equal(err.statusCode, 502);
});

test("classifyEngineFailure maps auth stderr", () => {
  const err = classifyEngineFailure({ stderr: "authentication failed for user postgres" });
  assert.equal(err.code, "AUTH_FAILED");
  assert.equal(err.statusCode, 401);
});

test("classifyEngineFailure falls back to ENGINE_FAILED", () => {
  const err = classifyEngineFailure({ stderr: "unexpected boom" });
  assert.equal(err.code, "ENGINE_FAILED");
  assert.equal(err.statusCode, 502);
});
