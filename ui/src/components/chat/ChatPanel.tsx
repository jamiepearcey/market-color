// ChatPanel — the market-color chat surface.
//
// Same visual language as the quant-algos brain chat (user/assistant bubbles,
// marked+DOMPurify prose, a composer with a model picker, a live activity line
// while the turn runs), repurposed for grounded market Q&A: each answer renders
// the facts it was retrieved-and-reasoned over as a FactsCard underneath.

import { useEffect, useMemo, useRef } from "react";
import { proxy, useSnapshot } from "valtio";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { ArrowUp, Bot, Loader2, Newspaper, Sparkles, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  chatStore,
  clearChat,
  probeReady,
  sendMessage,
  setModel,
} from "@/lib/chat/store";
import { MODEL_OPTIONS, type ChatMessage } from "@/lib/chat/model";
import { FactsCard } from "@/components/chat/FactsCard";

const PRESETS = [
  "What's the latest on OPEC+ supply and crude oil?",
  "Summarize today's macro and central-bank color.",
  "What's moving in metals right now?",
  "Any notable geopolitical risk in the corpus this week?",
];

export function ChatPanel() {
  const snap = useSnapshot(chatStore);
  const local = useRef(proxy({ draft: "" })).current;
  const ui = useSnapshot(local);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void probeReady();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [snap.messages.length, snap.activity]);

  const send = () => {
    const text = local.draft.trim();
    if (!text || snap.sending || snap.ready === false) return;
    local.draft = "";
    void sendMessage(text);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-[920px] flex-col gap-3 px-4 py-4">
          {snap.messages.length === 0 && (
            <ChatEmptyState
              ready={snap.ready}
              onPick={(p) => {
                local.draft = p;
              }}
            />
          )}
          {snap.messages.map((m) => (
            <MessageView key={m.id} m={m as ChatMessage} />
          ))}
          {snap.sending && <PendingActivity activity={snap.activity} isTool={snap.activityIsTool} />}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-outline-subtle bg-surface-header px-4 py-3">
        <div className="mx-auto w-full max-w-[920px] rounded-xl border border-outline-subtle bg-surface-panel/70 p-2.5">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="gap-1">
              <Newspaper className="size-3" /> Market Color
            </Badge>
            <span className="text-[10.5px] text-muted-foreground">
              Answers are grounded in structured facts retrieved from the news corpus via MCP.
            </span>
            <label className="ml-auto flex items-center gap-1 text-[10.5px] text-muted-foreground">
              <span className="opacity-60">model</span>
              <select
                value={snap.model}
                onChange={(e) => setModel(e.target.value)}
                className="rounded border border-outline-subtle bg-surface-input px-1.5 py-0.5 text-[10.5px]"
              >
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            {snap.messages.length > 0 && (
              <button
                onClick={() => clearChat()}
                className="text-[10.5px] text-muted-foreground hover:text-foreground"
              >
                clear
              </button>
            )}
          </div>

          <div className="flex items-end gap-2">
            <textarea
              value={ui.draft}
              onChange={(e) => (local.draft = e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder={
                snap.ready === false
                  ? "Chat backend unavailable — set ANTHROPIC_API_KEY and restart the dev server."
                  : "Ask about markets — OPEC, the Fed, gold, equities, geopolitics…"
              }
              rows={Math.min(6, Math.max(2, ui.draft.split("\n").length))}
              disabled={snap.ready === false}
              className="min-h-[72px] flex-1 resize-none rounded-lg border border-outline-subtle bg-surface-input px-3 py-2.5 text-[13px] leading-relaxed text-foreground/90 placeholder:text-muted-foreground/60 disabled:opacity-60"
            />
            <Button
              onClick={send}
              disabled={!ui.draft.trim() || snap.sending || snap.ready === false}
              className="h-[72px] min-w-[72px] shrink-0 rounded-lg px-3"
            >
              {snap.sending ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  <span>Sending</span>
                </>
              ) : (
                <>
                  <ArrowUp className="size-4" />
                  <span>Send</span>
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageView({ m }: { m: ChatMessage }) {
  if (m.role === "user") {
    return (
      <div className="ml-auto flex max-w-[85%] items-start gap-2">
        <div className="whitespace-pre-wrap rounded-md border border-primary/30 bg-primary/[0.08] px-3 py-2 text-[13px] leading-relaxed text-foreground/95">
          {m.text}
        </div>
        <span className="mt-1 grid size-6 shrink-0 place-items-center rounded border border-outline-subtle bg-surface-toolbar text-muted-foreground">
          <User className="size-3.5" />
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2">
      <span className="mt-1 grid size-6 shrink-0 place-items-center rounded border border-transparent bg-primary/[0.14] text-icon-tile-foreground">
        <Bot className="size-3.5" />
      </span>
      <div className="min-w-0 flex-1 space-y-2">
        {m.text && <Prose text={m.text} />}
        {m.error && (
          <div className="rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
            {m.error}
          </div>
        )}
        {m.facts && m.facts.length > 0 && <FactsCard facts={m.facts} />}
        {!m.pending && !m.text && !m.error && (
          <div className="rounded-md border border-outline-subtle bg-surface-panel px-3 py-2 text-[12px] text-muted-foreground">
            No answer was produced.
          </div>
        )}
      </div>
    </div>
  );
}

function PendingActivity({ activity, isTool }: { activity: string | null; isTool: boolean }) {
  return (
    <div className="flex items-start gap-2">
      <span className="mt-1 grid size-6 shrink-0 place-items-center rounded border border-transparent bg-primary/[0.14] text-icon-tile-foreground">
        <Bot className="size-3.5" />
      </span>
      <div className="min-w-0 flex-1 rounded-md border border-outline-subtle bg-surface-panel px-3 py-2.5">
        <div className="flex items-center gap-2 text-[12.5px] text-muted-foreground">
          <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" />
          <span className="shrink-0">{activity ? "Working" : "Starting turn"}</span>
          {activity && (
            <span
              className={cn(
                "min-w-0 truncate font-mono text-[11px]",
                isTool ? "text-emerald-300" : "text-foreground/75",
              )}
            >
              {activity}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function Prose({ text }: { text: string }) {
  const html = useMemo(
    () => DOMPurify.sanitize(marked.parse(text, { async: false })),
    [text],
  );
  return (
    <div
      className={cn(
        "prose-chat rounded-md border border-outline-subtle bg-surface-panel px-3 py-2.5 text-[13px] leading-relaxed text-foreground/90",
        "[&_code]:rounded [&_code]:bg-surface-input [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[11.5px]",
        "[&_p]:my-1.5 [&_ul]:my-1.5 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-1.5 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5",
        "[&_h1]:text-[15px] [&_h2]:text-[14px] [&_h3]:text-[13px] [&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold",
        "[&_pre]:overflow-x-auto [&_pre]:rounded [&_pre]:border [&_pre]:border-outline-subtle [&_pre]:bg-surface-input [&_pre]:p-2",
        "[&_a]:text-primary [&_a]:underline",
        "[&_table]:my-2 [&_th]:px-2 [&_th]:py-1 [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_td]:border [&_th]:border-outline-subtle [&_td]:border-outline-subtle",
      )}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function ChatEmptyState({
  ready,
  onPick,
}: {
  ready: boolean | null;
  onPick: (p: string) => void;
}) {
  return (
    <div className="mt-10 flex flex-col items-center gap-4 text-center">
      <span className="grid size-12 place-items-center rounded-xl bg-gradient-to-br from-amber-400 to-orange-600 shadow-[0_0_18px_-4px] shadow-amber-500/50">
        <Sparkles className="size-6 text-white" />
      </span>
      <div>
        <div className="text-[15px] font-semibold">Ask the market-color corpus</div>
        <div className="mt-1 max-w-[460px] text-[12.5px] text-muted-foreground">
          Questions are answered from structured facts retrieved out of the crawled news
          corpus via Qdrant + MCP — every claim links back to its source.
        </div>
      </div>
      {ready === false && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/[0.08] px-3 py-2 text-[12px] text-amber-200">
          Backend unavailable. Set <code className="font-mono">ANTHROPIC_API_KEY</code> and make sure
          Qdrant is running, then restart the dev server.
        </div>
      )}
      <div className="flex max-w-[560px] flex-wrap justify-center gap-2">
        {PRESETS.map((p) => (
          <button
            key={p}
            onClick={() => onPick(p)}
            className="rounded-md border border-outline-subtle bg-surface-panel px-3 py-1.5 text-[12px] text-foreground/80 transition-colors hover:border-primary/40 hover:bg-primary/[0.06]"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}
