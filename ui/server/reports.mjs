// market-color report generation.
//
// A report is a grounded markdown write-up produced from the indexed facts —
// either automatically after a daily run (kind='daily') or on demand from a
// prompt (kind='adhoc'). Both go through the same Claude+MCP loop as the chat,
// with a report-oriented system prompt, and are persisted to Postgres with the
// facts they were grounded on.

import { runOnce, DEFAULT_MODEL } from "./chat-core.mjs";
import { getReport, getSetting, updateReport } from "./db.mjs";

const REPORT_SYSTEM = `You are Market Color, writing a desk-ready market intelligence report from a
corpus of structured facts (recent financial news, decomposed into atomic claims).

- Use the search_market_facts tool to gather the facts the report needs — search each
  desk/theme the brief calls for, more than once where useful. Filter by desk and by the
  'since' date when the brief is time-bounded.
- Write in markdown: a one-paragraph executive summary, then sections grouped by desk
  (## Energy, ## Rates, ## FX, …) covering only desks with material facts.
- Ground every statement in a retrieved fact and cite the source inline, e.g.
  "(Reuters, 2026-06-30)". Attribute numbers to the fact that carried them.
- Do not invent facts or fill gaps from memory. If a desk has no material facts, omit it.
  If the corpus is empty for the brief, say so.`;

export function buildDailyPrompt(runDate) {
  return (
    `Write today's market color report for ${runDate}. Cover the day's most material ` +
    `developments across every desk that has news, grounded in facts published on or ` +
    `around ${runDate} (use the 'since' filter = ${runDate}). Lead with the cross-asset ` +
    `takeaway, then per-desk sections.`
  );
}

/** Run the grounded generation for an existing (pending) report row and persist
 *  the result. Returns the updated report. Awaitable — the orchestrator waits on
 *  it; the API fires it in the background. */
export async function generateReport(reportId) {
  const report = await getReport(reportId);
  if (!report) return null;
  const model = (await getSetting("default_model", DEFAULT_MODEL)) || DEFAULT_MODEL;
  const prompt = report.prompt || buildDailyPrompt(report.run_date ?? "the latest date");
  try {
    const out = await runOnce(prompt, { model, system: REPORT_SYSTEM });
    if (out.error && !out.reply) {
      return updateReport(reportId, { status: "error", error: out.error });
    }
    return updateReport(reportId, {
      body: out.reply || "",
      facts: out.facts || [],
      status: "done",
      error: null,
    });
  } catch (e) {
    return updateReport(reportId, { status: "error", error: String(e) });
  }
}
