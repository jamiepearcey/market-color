// market-color chat transport — the same NDJSON streaming contract the
// quant-algos brain uses, pointed at our Claude+MCP backend.
//
// POST /chat streams newline-delimited JSON: live {type:"event"} activity lines
// while Claude searches the corpus and reasons, then a single {type:"result"}
// line. GET /chat/health reports whether the backend is configured.

import type { Fact, ToolCall } from "./model";
import { apiBase } from "../api-base";

export interface ChatReply {
  threadId: string | null;
  reply: string;
  facts: Fact[];
  toolCalls: ToolCall[];
  error?: string;
}

export interface ChatActivity {
  kind: "thinking" | "tool" | "shell" | "delta";
  label: string;
}

/** Is the chat backend reachable and configured (ANTHROPIC_API_KEY present)? */
export async function chatAvailable(): Promise<boolean> {
  try {
    const res = await fetch(`${apiBase()}/chat/health`);
    if (!res.ok) return false;
    return ((await res.json()) as { available?: boolean }).available === true;
  } catch {
    return false;
  }
}

/** Send one turn. Streams activity via onActivity, resolves to the final reply. */
export async function sendChat(
  prompt: string,
  threadId: string | null,
  onActivity?: (a: ChatActivity) => void,
  model?: string | null,
): Promise<ChatReply> {
  let res: Response;
  try {
    res = await fetch(`${apiBase()}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, threadId, model: model || null }),
    });
  } catch (e) {
    return { threadId, reply: "", facts: [], toolCalls: [], error: `chat bridge: ${String(e)}` };
  }
  if (!res.ok || !res.body) {
    return { threadId, reply: "", facts: [], toolCalls: [], error: `chat bridge: HTTP ${res.status}` };
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let result: ChatReply | null = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line) as
          | { type: "event"; event: ChatActivity }
          | { type: "result"; result: ChatReply };
        if (msg.type === "event") onActivity?.(msg.event);
        else result = msg.result;
      } catch {
        // partial/garbled line — ignore
      }
    }
  }
  return result ?? { threadId, reply: "", facts: [], toolCalls: [], error: "stream ended without a result" };
}
