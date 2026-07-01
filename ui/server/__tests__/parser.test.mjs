import test from "node:test";
import assert from "node:assert/strict";
import { parseRunSummary } from "../parser.mjs";

test("parseRunSummary handles a single stream summary", () => {
  const fixture = `
[info] run starting
Records:
  users: 12
Total records: 12
`;
  assert.deepEqual(parseRunSummary(fixture), {
    streams: [{ stream: "users", count: 12 }],
    total: 12,
  });
});

test("parseRunSummary handles multiple streams", () => {
  const fixture = `
Records:
  accounts: 7
  contacts: 3
Total records: 10
`;
  assert.deepEqual(parseRunSummary(fixture), {
    streams: [
      { stream: "accounts", count: 7 },
      { stream: "contacts", count: 3 },
    ],
    total: 10,
  });
});

test("parseRunSummary handles zero-record runs", () => {
  const fixture = `
Records:
  (none)
Total records: 0
`;
  assert.deepEqual(parseRunSummary(fixture), {
    streams: [],
    total: 0,
  });
});

test("parseRunSummary ignores failed-run stderr noise", () => {
  const fixture = `
[error] loader failed
connection reset by peer
`;
  assert.deepEqual(parseRunSummary(fixture), {
    streams: [],
    total: 0,
  });
});
