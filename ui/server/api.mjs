// market-color REST API — the /api surface.
//
// Pure-ish router: handleApi(method, pathname, body) -> {status, json}. The
// standalone server adapts req/res to it. Everything reads/writes Postgres
// (config + state); runs and reports are kicked off async and polled.

import crypto from "node:crypto";
import {
  addDesk,
  addAdhoc,
  addSource,
  createReport,
  dbReady,
  deleteDesk,
  deleteSource,
  getReport,
  getRun,
  listDesks,
  listReports,
  listRuns,
  listSources,
  updateSource,
} from "./db.mjs";
import { ingestAdhoc, triggerDailyRun } from "./orchestrator.mjs";
import { generateReport } from "./reports.mjs";
import { chatReady } from "./chat-core.mjs";
import { entityFacts, getGraph } from "./graph.mjs";
import { listBriefs, readBrief } from "./briefs.mjs";

async function qdrantReady() {
  const url = process.env.QDRANT_URL || "http://localhost:6333";
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 1500);
    const res = await fetch(`${url}/collections`, { signal: ctrl.signal });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

const ok = (json, status = 200) => ({ status, json });
const err = (message, status = 400) => ({ status, json: { error: message } });

export async function handleApi(method, pathname, body) {
  const url = new URL(pathname, "http://api.local"); // pathname may carry ?query
  const q = url.searchParams;
  const parts = url.pathname.replace(/^\/+|\/+$/g, "").split("/").filter(Boolean); // e.g. ["desks","5"]
  const [resource, id, sub] = parts;

  try {
    // --- health ---
    if (resource === "health") {
      const [db, qdrant] = await Promise.all([dbReady(), qdrantReady()]);
      return ok({ db, qdrant, anthropic: chatReady() });
    }

    // --- desks ---
    if (resource === "desks") {
      if (method === "GET" && !id) return ok(await listDesks());
      if (method === "POST" && !id) {
        if (!body?.key?.trim() || !body?.label?.trim()) return err("key and label are required");
        return ok(await addDesk({ key: body.key.trim(), label: body.label.trim(), anchor: body.anchor ?? "" }));
      }
      if (method === "DELETE" && id) return ok(await deleteDesk(Number(id)));
    }

    // --- sources ---
    if (resource === "sources") {
      if (method === "GET" && !id) return ok(await listSources());
      if (method === "POST" && id === "upload") {
        if (!body?.title?.trim() || !body?.body?.trim()) return err("title and body are required");
        const publishedDate = (body.published_date || new Date().toISOString().slice(0, 10)).slice(0, 10);
        const docId =
          "adhoc:" +
          crypto
            .createHash("sha1")
            .update(`${body.title}|${publishedDate}|${String(body.body).slice(0, 256)}`)
            .digest("hex")
            .slice(0, 16);
        const document = await addAdhoc({
          doc_id: docId,
          title: body.title.trim(),
          body: body.body,
          url: body.url ?? null,
          published_date: publishedDate,
          desks: Array.isArray(body.desks) ? body.desks : [],
        });
        const run = await ingestAdhoc(document.doc_id);
        return ok({ document: { doc_id: document.doc_id }, run });
      }
      if (method === "POST" && !id) {
        if (!body?.name?.trim() || !body?.url?.trim()) return err("name and url are required");
        return ok(await addSource({ ...body, kind: "rss" }));
      }
      if (method === "PATCH" && id) return ok(await updateSource(Number(id), body ?? {}));
      if (method === "DELETE" && id) return ok(await deleteSource(Number(id)));
    }

    // --- runs ---
    if (resource === "runs") {
      if (method === "GET" && !id) return ok(await listRuns());
      if (method === "GET" && id) {
        const run = await getRun(Number(id));
        return run ? ok(run) : err("not found", 404);
      }
      if (method === "POST" && !id) return ok(await triggerDailyRun(body?.date || null));
    }

    // --- reports ---
    if (resource === "reports") {
      if (method === "GET" && !id) return ok(await listReports());
      if (method === "GET" && id) {
        const report = await getReport(Number(id));
        return report ? ok(report) : err("not found", 404);
      }
      if (method === "POST" && !id) {
        if (!body?.prompt?.trim()) return err("prompt is required");
        const report = await createReport({
          kind: "adhoc",
          run_date: body.run_date ?? null,
          title: body.title?.trim() || "Ad-hoc report",
          prompt: body.prompt.trim(),
        });
        void generateReport(report.id); // background; client polls
        return ok(report);
      }
    }

    // --- graph (transmission map from the Qdrant facts) ---
    if (resource === "graph") {
      if (method === "GET" && id === "entity") {
        const name = (q.get("name") || "").trim();
        if (!name) return err("name is required");
        return ok(await entityFacts(name, q.get("desk")?.trim() || null));
      }
      if (method === "GET" && !id) {
        const since = q.get("since")?.trim() || null;
        const until = q.get("until")?.trim() || null;
        for (const [k, v] of [["since", since], ["until", until]]) {
          if (v && !/^\d{4}-\d{2}-\d{2}$/.test(v)) return err(`${k} must be YYYY-MM-DD`);
        }
        const num = (v, dflt) => (v && Number.isFinite(Number(v)) ? Number(v) : dflt);
        return ok(
          await getGraph({
            desk: q.get("desk")?.trim() || null,
            since,
            until,
            minWeight: num(q.get("min_weight"), 1),
            maxNodes: num(q.get("max_nodes"), 150),
          }),
        );
      }
    }

    // --- briefs (rendered per-desk daily notes under data/briefs/) ---
    if (resource === "briefs") {
      if (method === "GET" && !id) return ok(listBriefs());
      if (method === "GET" && id && sub) {
        const brief = readBrief(id, sub);
        return brief ? ok(brief) : err("not found", 404);
      }
    }

    return err(`no route for ${method} /api/${parts.join("/")}`, 404);
  } catch (e) {
    return err(String(e?.message || e), 500);
  }
}
