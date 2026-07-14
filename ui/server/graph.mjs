// Transmission-map graph built from the Qdrant `market_facts` collection.
//
// Nodes are fact entities; undirected edges are entity co-occurrence within a
// fact (weight = shared-fact count); when a fact carries `cause_entities`
// (feature-detected — later loads may add it) we ALSO emit directed causal
// edges cause → subject-ish entity. Everything is read via the Qdrant REST
// scroll API with a payload filter (desk / published_ordinal day range) —
// Node built-ins + fetch only, matching the api.mjs style.

const QDRANT_URL = () => process.env.QDRANT_URL || "http://localhost:6333";
const COLLECTION = () => process.env.MARKET_FACTS_COLLECTION || "market_facts";

const SCROLL_PAGE = 512;
const MAX_POINTS = 20000; // safety cap — corpus is ~hundreds of facts per desk today

/** Python date.toordinal() equivalent for a YYYY-MM-DD string (the corpus
 *  indexes `published_ordinal` this way: days since 0001-01-01, 1-based). */
export function dateToOrdinal(yyyyMmDd) {
  const [y, m, d] = yyyyMmDd.split("-").map(Number);
  return Math.floor(Date.UTC(y, m - 1, d) / 86400000) + 719163;
}

function qdrantFilter({ desk, since, until, entity }) {
  const must = [];
  if (desk) must.push({ key: "desk", match: { value: desk } });
  if (entity) must.push({ key: "entities", match: { any: [entity] } });
  if (since || until) {
    const range = {};
    if (since) range.gte = dateToOrdinal(since);
    if (until) range.lte = dateToOrdinal(until);
    must.push({ key: "published_ordinal", range });
  }
  return must.length ? { must } : undefined;
}

/** Scroll every matching point's payload out of Qdrant (paginated). */
export async function scrollFacts(filter, { maxPoints = MAX_POINTS } = {}) {
  const url = `${QDRANT_URL()}/collections/${COLLECTION()}/points/scroll`;
  const payloads = [];
  let offset = null;
  do {
    const body = {
      limit: Math.min(SCROLL_PAGE, maxPoints - payloads.length),
      with_payload: true,
      with_vector: false,
    };
    if (filter) body.filter = filter;
    if (offset !== null) body.offset = offset;
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`qdrant scroll failed: HTTP ${res.status} ${await res.text()}`);
    const data = await res.json();
    const points = data?.result?.points ?? [];
    for (const p of points) if (p?.payload) payloads.push(p.payload);
    offset = data?.result?.next_page_offset ?? null;
  } while (offset !== null && payloads.length < maxPoints);
  return payloads;
}

function normEntity(s) {
  return String(s).trim().toLowerCase();
}

/** Pure graph builder from fact payloads (exported for tests). */
export function buildGraphFromFacts(facts, { minWeight = 1, maxNodes = 150 } = {}) {
  const counts = new Map(); // entity -> #facts mentioning it
  const cooc = new Map(); // "ab" (a<b) -> weight
  const causal = new Map(); // "causeeffect" -> weight

  for (const f of facts) {
    const entities = [
      ...new Set((Array.isArray(f?.entities) ? f.entities : []).map(normEntity).filter(Boolean)),
    ];
    for (const e of entities) counts.set(e, (counts.get(e) ?? 0) + 1);

    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const [a, b] = entities[i] < entities[j] ? [entities[i], entities[j]] : [entities[j], entities[i]];
        const k = `${a}${b}`;
        cooc.set(k, (cooc.get(k) ?? 0) + 1);
      }
    }

    // Feature-detect `cause_entities` (arrives with later fact loads): emit
    // directed causal edges cause → the subject-ish entity of the fact.
    const causes = Array.isArray(f?.cause_entities)
      ? [...new Set(f.cause_entities.map(normEntity).filter(Boolean))]
      : [];
    if (causes.length && entities.length) {
      const subject = f?.subject ? normEntity(f.subject) : null;
      const causeSet = new Set(causes);
      const target =
        (subject && entities.find((e) => e === subject || e.includes(subject) || subject.includes(e))) ||
        entities.find((e) => !causeSet.has(e)) ||
        null;
      if (target) {
        for (const c of causes) {
          if (c === target) continue;
          if (!counts.has(c)) counts.set(c, counts.get(c) ?? 0); // cause may not be in entities
          const k = `${c}${target}`;
          causal.set(k, (causal.get(k) ?? 0) + 1);
        }
      }
    }
  }

  const kept = new Set(
    [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
      .slice(0, maxNodes)
      .map(([e]) => e),
  );

  const edges = [];
  for (const [k, weight] of cooc) {
    if (weight < minWeight) continue;
    const [source, target] = k.split("");
    if (kept.has(source) && kept.has(target)) edges.push({ source, target, weight, causal: false });
  }
  // Causal edges are the interesting ones — keep them regardless of min_weight.
  for (const [k, weight] of causal) {
    const [source, target] = k.split("");
    if (kept.has(source) && kept.has(target)) edges.push({ source, target, weight, causal: true });
  }
  edges.sort((a, b) => b.weight - a.weight);

  const nodes = [...kept].map((id) => ({ id, count: counts.get(id) ?? 0 }));
  nodes.sort((a, b) => b.count - a.count || (a.id < b.id ? -1 : 1));
  return { nodes, edges };
}

// --- 60s in-process cache keyed by the query --------------------------------
const CACHE_TTL_MS = 60_000;
const cache = new Map(); // key -> { t, data }

export function _resetGraphCache() {
  cache.clear();
}

export async function getGraph({ desk = null, since = null, until = null, minWeight = 1, maxNodes = 150 } = {}) {
  const key = JSON.stringify({ desk, since, until, minWeight, maxNodes });
  const hit = cache.get(key);
  if (hit && Date.now() - hit.t < CACHE_TTL_MS) return hit.data;

  const facts = await scrollFacts(qdrantFilter({ desk, since, until }));
  const data = buildGraphFromFacts(facts, { minWeight, maxNodes });
  cache.set(key, { t: Date.now(), data });
  // opportunistic prune so the map never grows unbounded
  if (cache.size > 64) {
    for (const [k, v] of cache) if (Date.now() - v.t >= CACHE_TTL_MS) cache.delete(k);
  }
  return data;
}

/** Facts mentioning one entity (newest first, capped at 50). */
export async function entityFacts(name, desk = null) {
  const entity = normEntity(name);
  const facts = await scrollFacts(qdrantFilter({ desk, entity }), { maxPoints: 1000 });
  facts.sort(
    (a, b) =>
      (b?.published_epoch ?? b?.published_ordinal ?? 0) - (a?.published_epoch ?? a?.published_ordinal ?? 0),
  );
  return { entity, facts: facts.slice(0, 50) };
}
