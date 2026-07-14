import test from "node:test";
import assert from "node:assert/strict";
import {
  buildGraphFromFacts,
  dateToOrdinal,
  entityFacts,
  getGraph,
  _resetGraphCache,
} from "../graph.mjs";

const fact = (entities, extra = {}) => ({ entities, ...extra });

test("dateToOrdinal matches Python date.toordinal()", () => {
  assert.equal(dateToOrdinal("2026-06-30"), 739797); // value observed in the corpus payloads
  assert.equal(dateToOrdinal("1970-01-01"), 719163);
});

test("graph builder counts nodes and weights co-occurrence edges", () => {
  const { nodes, edges } = buildGraphFromFacts([
    fact(["oil", "hormuz"]),
    fact(["oil", "hormuz", "lng"]),
    fact(["oil"]),
  ]);
  assert.deepEqual(
    nodes,
    [
      { id: "oil", count: 3 },
      { id: "hormuz", count: 2 },
      { id: "lng", count: 1 },
    ],
  );
  const key = (e) => `${e.source}->${e.target}`;
  const byKey = Object.fromEntries(edges.map((e) => [key(e), e]));
  assert.equal(byKey["hormuz->oil"].weight, 2);
  assert.equal(byKey["hormuz->oil"].causal, false);
  assert.equal(byKey["hormuz->lng"].weight, 1);
  assert.equal(byKey["lng->oil"].weight, 1);
  assert.equal(edges.length, 3);
});

test("min_weight trims weak co-occurrence edges and max_nodes keeps the top entities", () => {
  const facts = [
    fact(["oil", "hormuz"]),
    fact(["oil", "hormuz"]),
    fact(["oil", "lng"]),
    fact(["lng", "pakistan"]),
  ];
  const g = buildGraphFromFacts(facts, { minWeight: 2, maxNodes: 3 });
  assert.deepEqual(
    g.nodes.map((n) => n.id),
    ["oil", "hormuz", "lng"], // pakistan (count 1) trimmed by max_nodes
  );
  assert.deepEqual(g.edges, [{ source: "hormuz", target: "oil", weight: 2, causal: false }]);
});

test("cause_entities is feature-detected and emits directed causal edges to the subject-ish entity", () => {
  const g = buildGraphFromFacts([
    fact(["brent forecast", "iran"], {
      subject: "Brent forecast",
      cause_entities: ["us-iran mou"],
    }),
    // no cause_entities → co-occurrence only (backward compatible)
    fact(["oil", "hormuz"]),
  ]);
  const causal = g.edges.filter((e) => e.causal);
  assert.deepEqual(causal, [
    { source: "us-iran mou", target: "brent forecast", weight: 1, causal: true },
  ]);
  // the cause entity becomes a node even though it never appears in `entities`
  assert.ok(g.nodes.some((n) => n.id === "us-iran mou"));
});

function stubScroll(pages, calls) {
  return async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    const page = pages[Math.min(calls.length - 1, pages.length - 1)];
    return {
      ok: true,
      json: async () => ({ result: page }),
      text: async () => "",
    };
  };
}

test("getGraph scrolls Qdrant with the desk/date filter, paginates, and caches for 60s", async (t) => {
  _resetGraphCache();
  const calls = [];
  const pages = [
    {
      points: [{ payload: fact(["oil", "hormuz"]) }],
      next_page_offset: "cursor-1",
    },
    {
      points: [{ payload: fact(["oil", "lng"]) }],
      next_page_offset: null,
    },
  ];
  const realFetch = globalThis.fetch;
  globalThis.fetch = stubScroll(pages, calls);
  t.after(() => {
    globalThis.fetch = realFetch;
    _resetGraphCache();
  });

  const g = await getGraph({ desk: "energy", since: "2026-06-27", until: "2026-07-01" });
  assert.equal(calls.length, 2);
  assert.match(calls[0].url, /\/collections\/market_facts\/points\/scroll$/);
  assert.deepEqual(calls[0].body.filter, {
    must: [
      { key: "desk", match: { value: "energy" } },
      { key: "published_ordinal", range: { gte: dateToOrdinal("2026-06-27"), lte: dateToOrdinal("2026-07-01") } },
    ],
  });
  assert.equal(calls[0].body.offset, undefined);
  assert.equal(calls[1].body.offset, "cursor-1"); // pagination cursor passed back
  assert.deepEqual(
    g.nodes.map((n) => n.id),
    ["oil", "hormuz", "lng"],
  );

  // second identical call is served from the in-process cache (no new fetches)
  const again = await getGraph({ desk: "energy", since: "2026-06-27", until: "2026-07-01" });
  assert.equal(calls.length, 2);
  assert.deepEqual(again, g);
});

test("entityFacts filters on entities MatchAny and returns newest-first, capped at 50", async (t) => {
  const calls = [];
  const points = Array.from({ length: 60 }, (_, i) => ({
    payload: { claim: `fact ${i}`, entities: ["oil"], published_epoch: 1000 + i },
  }));
  const realFetch = globalThis.fetch;
  globalThis.fetch = stubScroll([{ points, next_page_offset: null }], calls);
  t.after(() => {
    globalThis.fetch = realFetch;
  });

  const res = await entityFacts("Oil", "energy");
  assert.equal(res.entity, "oil");
  assert.deepEqual(calls[0].body.filter, {
    must: [
      { key: "desk", match: { value: "energy" } },
      { key: "entities", match: { any: ["oil"] } },
    ],
  });
  assert.equal(res.facts.length, 50);
  assert.equal(res.facts[0].claim, "fact 59"); // newest first
  assert.equal(res.facts[49].claim, "fact 10");
});
