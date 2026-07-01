// market-color chat/report core — Claude grounded on Qdrant facts via MCP.
//
// One turn = a manual Anthropic agentic loop in which Claude is given a single
// tool, `search_market_facts`, served by our Python MCP server
// (mcp/qdrant_facts_server.py). We are the MCP *client*: we spawn that server
// over stdio, list its tools, hand them to Claude, execute the tool calls
// Claude makes, and feed the structured facts back until Claude produces a
// grounded answer. The facts retrieved this turn are returned alongside the
// prose so callers (chat UI, report generator) can show provenance.
//
// Two entry points share the loop:
//   runChat  — multi-turn, thread-persisted, NDJSON-streamed (the chat UI).
//   runOnce  — single grounded turn with a custom system prompt (reports).

import path from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(HERE, "../.."); // research/market-color
const MCP_SERVER = path.join(PROJECT_ROOT, "mcp", "qdrant_facts_server.py");

export const DEFAULT_MODEL = "claude-opus-4-8";
const MAX_TOOL_ROUNDS = 6;

export const CHAT_SYSTEM = `You are Market Color, a markets-desk assistant. You answer questions about
markets, companies, commodities, central banks, and macro/geopolitical events using a
corpus of structured, source-attributed facts decomposed from recent financial news, plus
an entity graph over those facts.

Tools — retrieve AND explore the fact graph (do not answer market questions from your own
memory; it is stale and ungrounded):
- search_market_facts(query, desk?, since?, expand?) — semantic search. Returns "seed" facts
  plus graph-connected "graph-neighbor" facts (personalized-PageRank over shared entities),
  each with its causal driver (cause), direction, entities and source.
- facts_for_entity(entity) — everything the corpus says about one entity.
- neighbors(entity) — the entities most connected to it (pick what to explore next).
- trace_causes(entity) — what DRIVES an entity and what IT DRIVES (build transmission chains).
- entity_path(entity_a, entity_b) — shortest path between two entities, with the connecting facts.

How to reason — explore the graph, don't stop at the first search:
1. search_market_facts to find the seed facts and the key entities.
2. Follow the graph with neighbors / trace_causes / entity_path to gather the connected facts —
   especially causal drivers and which assets are affected. Take a few hops for transmission or
   "what's exposed" questions, not just one search.
3. Assemble the chain: driver -> affected asset -> second-order effect, each step tied to a fact.
4. Weigh it: note corroboration (independent sources) and any contradictions.

Grounding — not optional:
- Assert only claims and causal links that a returned fact supports. Never invent an edge the
  graph didn't give you. If retrieval is empty or weak, say so ("I don't have indexed facts on
  that") and abstain rather than guess.
- Cite sources inline, e.g. "(Reuters, 2026-06-30)"; attribute figures to the fact that carried them.
- Be concise and desk-ready: lead with the takeaway, then the supporting chain.`;

let anthropic = null;
let mcpClient = null;
let mcpTools = null; // Anthropic-shaped tool definitions
let mcpConnecting = null;
const threads = new Map(); // threadId -> Anthropic.MessageParam[]
let threadSeq = 0;

export function chatReady() {
  return Boolean(process.env.ANTHROPIC_API_KEY);
}

function getAnthropic() {
  if (!anthropic) anthropic = new Anthropic(); // reads ANTHROPIC_API_KEY from env
  return anthropic;
}

async function connectMcp() {
  if (mcpClient) return;
  if (mcpConnecting) return mcpConnecting;
  mcpConnecting = (async () => {
    const transport = new StdioClientTransport({
      command: process.env.MARKET_MCP_COMMAND || "uv",
      args: process.env.MARKET_MCP_COMMAND
        ? (process.env.MARKET_MCP_ARGS || "").split(" ").filter(Boolean)
        : ["run", "--with", "mcp", "--with", "qdrant-client>=1.15", "--with", "fastembed", "python", MCP_SERVER],
      env: {
        ...process.env,
        QDRANT_URL: process.env.QDRANT_URL || "http://localhost:6333",
        MARKET_FACTS_COLLECTION: process.env.MARKET_FACTS_COLLECTION || "market_facts",
      },
    });
    const client = new Client({ name: "market-color-chat", version: "0.1.0" }, { capabilities: {} });
    await client.connect(transport);
    const { tools } = await client.listTools();
    mcpClient = client;
    mcpTools = tools.map((t) => ({
      name: t.name,
      description: t.description ?? "",
      input_schema: t.inputSchema ?? { type: "object", properties: {} },
    }));
  })();
  try {
    await mcpConnecting;
  } finally {
    mcpConnecting = null;
  }
}

function readToolResult(res) {
  let text = "";
  if (Array.isArray(res?.content)) {
    text = res.content
      .filter((b) => b?.type === "text")
      .map((b) => b.text)
      .join("\n")
      .trim();
  }
  let facts = [];
  const fromStructured = res?.structuredContent?.result ?? res?.structuredContent;
  if (Array.isArray(fromStructured)) {
    facts = fromStructured;
  } else if (text) {
    try {
      const parsed = JSON.parse(text);
      facts = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.result) ? parsed.result : [];
    } catch {
      facts = [];
    }
  }
  if (!text) text = JSON.stringify(facts);
  return { text, facts };
}

/** The shared agentic loop. Mutates `messages` in place (so callers can persist
 *  a thread) and returns the prose + the facts/toolCalls gathered this turn. */
async function agenticTurn(messages, model, system, emit) {
  const facts = [];
  const factKeys = new Set();
  const toolCalls = [];
  const client = getAnthropic();
  let reply = "";

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    emit?.({ kind: "thinking", label: round === 0 ? "Reading the question…" : "Synthesising the facts…" });
    const resp = await client.messages.create({
      model: model || DEFAULT_MODEL,
      max_tokens: 8000,
      system,
      thinking: { type: "adaptive" },
      tools: mcpTools,
      messages,
    });

    if (resp.stop_reason === "refusal") {
      return { reply: "", facts, toolCalls, error: "the model declined to answer" };
    }

    messages.push({ role: "assistant", content: resp.content });

    const toolUses = resp.content.filter((b) => b.type === "tool_use");
    const textNow = resp.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("");
    if (textNow) reply = textNow;

    if (resp.stop_reason !== "tool_use" || toolUses.length === 0) break;

    const toolResults = [];
    for (const tu of toolUses) {
      const argStr =
        tu.input && typeof tu.input === "object"
          ? Object.entries(tu.input)
              .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
              .join(" ")
          : "";
      emit?.({ kind: "tool", label: `${tu.name}(${argStr})` });
      toolCalls.push({ tool: tu.name, args: tu.input ?? {}, status: "ok" });
      try {
        const res = await mcpClient.callTool({ name: tu.name, arguments: tu.input ?? {} });
        const { text, facts: got } = readToolResult(res);
        for (const f of got) {
          const key = f.fact_id || `${f.source_name}:${f.claim}`;
          if (!factKeys.has(key)) {
            factKeys.add(key);
            facts.push(f);
          }
        }
        toolResults.push({ type: "tool_result", tool_use_id: tu.id, content: text || "[]" });
      } catch (e) {
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: `tool error: ${String(e)}`,
          is_error: true,
        });
      }
    }
    messages.push({ role: "user", content: toolResults });
  }

  return { reply, facts, toolCalls };
}

/** Multi-turn chat. Streams activity via onEvent; resolves to the turn result. */
export async function runChat(prompt, threadId, model, onEvent) {
  const emit = (e) => {
    try {
      onEvent?.(e);
    } catch {
      /* ignore */
    }
  };
  if (!process.env.ANTHROPIC_API_KEY) {
    return { threadId, reply: "", facts: [], toolCalls: [], error: "ANTHROPIC_API_KEY is not set" };
  }
  try {
    emit({ kind: "thinking", label: "Connecting to the market-color corpus…" });
    await connectMcp();
  } catch (e) {
    return {
      threadId,
      reply: "",
      facts: [],
      toolCalls: [],
      error: `could not start the Qdrant MCP server (is uv/python + Qdrant available?): ${String(e)}`,
    };
  }

  const tid = threadId || `thr-${Date.now().toString(36)}-${++threadSeq}`;
  const messages = threads.get(tid) ?? [];
  messages.push({ role: "user", content: prompt });
  try {
    const out = await agenticTurn(messages, model, CHAT_SYSTEM, emit);
    threads.set(tid, messages);
    return { threadId: tid, ...out };
  } catch (e) {
    threads.set(tid, messages);
    return { threadId: tid, reply: "", facts: [], toolCalls: [], error: String(e) };
  }
}

/** Single grounded turn with a caller-supplied system prompt (report generation).
 *  No thread persistence. */
export async function runOnce(prompt, { model, system, emit } = {}) {
  if (!process.env.ANTHROPIC_API_KEY) {
    return { reply: "", facts: [], toolCalls: [], error: "ANTHROPIC_API_KEY is not set" };
  }
  try {
    await connectMcp();
  } catch (e) {
    return { reply: "", facts: [], toolCalls: [], error: `MCP server unavailable: ${String(e)}` };
  }
  const messages = [{ role: "user", content: prompt }];
  try {
    return await agenticTurn(messages, model, system || CHAT_SYSTEM, emit);
  } catch (e) {
    return { reply: "", facts: [], toolCalls: [], error: String(e) };
  }
}

export async function shutdown() {
  try {
    await mcpClient?.close();
  } catch {
    /* ignore */
  }
  mcpClient = null;
  mcpTools = null;
}
