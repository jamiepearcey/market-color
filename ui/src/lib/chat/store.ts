// market-color chat store — valtio state + the send orchestration.
//
// Lean by design: the conversation lives here, the backend owns the thread
// (history + MCP loop) keyed by threadId. We push the user message, add a
// pending assistant message, stream activity into it, then settle it with the
// reply + the facts the answer was grounded on.

import { proxy } from "valtio";
import { chatAvailable, sendChat, type ChatActivity } from "./chat";
import { DEFAULT_MODEL, mintId, newChatMessage, type ChatMessage } from "./model";

const MODEL_KEY = "market-color:model";

interface ChatStore {
  messages: ChatMessage[];
  /** null = unknown (probing), true/false = backend reachable. */
  ready: boolean | null;
  /** A turn is in flight (we serialize turns). */
  sending: boolean;
  /** One-line description of what the backend is doing right now. */
  activity: string | null;
  /** Whether the current activity is a tool (corpus) call vs. thinking. */
  activityIsTool: boolean;
  model: string;
  threadId: string | null;
}

const initialModel =
  (typeof localStorage !== "undefined" && localStorage.getItem(MODEL_KEY)) || DEFAULT_MODEL;

export const chatStore = proxy<ChatStore>({
  messages: [],
  ready: null,
  sending: false,
  activity: null,
  activityIsTool: false,
  model: initialModel,
  threadId: null,
});

export async function probeReady(): Promise<void> {
  chatStore.ready = await chatAvailable();
}

export function setModel(model: string): void {
  chatStore.model = model;
  try {
    localStorage.setItem(MODEL_KEY, model);
  } catch {
    /* ignore */
  }
}

export function clearChat(): void {
  chatStore.messages = [];
  chatStore.threadId = null;
  chatStore.activity = null;
}

export async function sendMessage(prompt: string): Promise<void> {
  const text = prompt.trim();
  if (!text || chatStore.sending) return;

  chatStore.sending = true;
  chatStore.activity = null;
  chatStore.activityIsTool = false;
  chatStore.messages.push(newChatMessage("user", text));

  const assistant: ChatMessage = {
    id: mintId("msg"),
    role: "assistant",
    text: "",
    at: Date.now(),
    pending: true,
  };
  chatStore.messages.push(assistant);
  const startedAt = Date.now();

  const onActivity = (a: ChatActivity) => {
    chatStore.activity = a.label;
    chatStore.activityIsTool = a.kind === "tool";
  };

  try {
    const result = await sendChat(text, chatStore.threadId, onActivity, chatStore.model);
    assistant.pending = false;
    assistant.text = result.reply;
    assistant.facts = result.facts ?? [];
    assistant.toolCalls = result.toolCalls ?? [];
    if (result.error) assistant.error = result.error;
    assistant.durationMs = Date.now() - startedAt;
    if (result.threadId) chatStore.threadId = result.threadId;
    if (result.error && !result.reply) chatStore.ready = await chatAvailable();
  } catch (e) {
    assistant.pending = false;
    assistant.error = String(e);
  } finally {
    chatStore.sending = false;
    chatStore.activity = null;
    chatStore.activityIsTool = false;
  }
}
