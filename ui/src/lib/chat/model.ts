// market-color chat model — a thin record of the conversation plus the
// structured facts each answer was grounded on.
//
// Unlike the quant-algos brain (which replays an engine), here the "evidence"
// is the set of facts the LLM retrieved from Qdrant via MCP this turn. We keep
// them on the assistant message so the UI can render source cards that link
// straight back to the article each fact came from.

/** One structured fact returned by the search_market_facts MCP tool. Mirrors
 *  the payload shape in mcp/qdrant_facts_server.py. */
export interface Fact {
  fact_id?: string;
  claim: string;
  desk?: string;
  direction?: "bullish" | "bearish" | "neutral" | string;
  metric?: string;
  entities?: string[];
  source_name?: string;
  published_date?: string;
  url?: string;
  title?: string;
  score?: number;
}

/** A tool call the model made this turn (for the activity / audit line). */
export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  status: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  at: number;
  /** Facts the answer was grounded on (assistant turns). */
  facts?: Fact[];
  toolCalls?: ToolCall[];
  /** Still streaming? */
  pending?: boolean;
  /** Backend error for this turn, if any. */
  error?: string;
  /** Wall-clock of the turn that produced an assistant message. */
  durationMs?: number;
}

let seq = 0;
export function mintId(prefix: string): string {
  seq += 1;
  return `${prefix}-${Date.now().toString(36)}-${seq}`;
}

export function newChatMessage(role: ChatMessage["role"], text: string): ChatMessage {
  return { id: mintId("msg"), role, text, at: Date.now() };
}

/** Model options offered in the composer. Defaults to the project default. */
export const MODEL_OPTIONS = [
  { id: "claude-opus-4-8", label: "Opus 4.8" },
  { id: "claude-sonnet-4-6", label: "Sonnet 4.6" },
  { id: "claude-haiku-4-5", label: "Haiku 4.5" },
] as const;

export const DEFAULT_MODEL = "claude-opus-4-8";
