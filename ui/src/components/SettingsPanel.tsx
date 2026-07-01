// SettingsPanel — API endpoint config + backend health + how retrieval works.
//
// The UI is a static client; this is where you point it at the API server
// (which may live on another host) and see whether Postgres, Qdrant, and the
// chat backend are reachable.

import { useEffect, useState } from "react";
import { useSnapshot } from "valtio";
import { CheckCircle2, XCircle, RefreshCw, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { chatStore, probeReady, setModel } from "@/lib/chat/store";
import { MODEL_OPTIONS } from "@/lib/chat/model";
import { getApiHealth, type ApiHealth } from "@/lib/mc-api";
import { getApiBaseRaw, setApiBase } from "@/lib/api-base";

export function SettingsPanel() {
  const snap = useSnapshot(chatStore);
  const [base, setBase] = useState(getApiBaseRaw());
  const [saved, setSaved] = useState(false);
  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [checking, setChecking] = useState(false);

  const recheck = async () => {
    setChecking(true);
    try {
      setHealth(await getApiHealth());
    } catch {
      setHealth({ db: false, qdrant: false, anthropic: false });
    }
    await probeReady();
    setChecking(false);
  };

  useEffect(() => {
    void recheck();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const saveBase = () => {
    setApiBase(base);
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
    void recheck();
  };

  return (
    <div className="mx-auto w-full max-w-[720px] px-6 py-8">
      <h1 className="text-[18px] font-semibold">Settings</h1>
      <p className="mt-1 text-[12.5px] text-muted-foreground">
        Market Color is a static client for the Market Color API. Point it at the API server
        and confirm the pipeline is reachable.
      </p>

      <section className="mt-6 rounded-md border border-outline-subtle bg-surface-panel p-4">
        <div className="text-[13px] font-semibold">API endpoint</div>
        <p className="mt-1 text-[11.5px] text-muted-foreground">
          Base URL of the API server. Leave blank for same-origin (dev proxy, or when the API
          serves this build). Set it to e.g. <code className="font-mono">https://api.example.com</code>{" "}
          when running the desktop client against a remote API.
        </p>
        <div className="mt-2 flex items-center gap-2">
          <input
            value={base}
            onChange={(e) => setBase(e.target.value)}
            placeholder="(same origin)"
            className="flex-1 rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[12.5px] font-mono"
          />
          <Button size="sm" onClick={saveBase}>
            <Save className="size-3.5" /> {saved ? "Saved" : "Save"}
          </Button>
        </div>
      </section>

      <section className="mt-4 rounded-md border border-outline-subtle bg-surface-panel p-4">
        <div className="flex items-center justify-between">
          <div className="text-[13px] font-semibold">Backend health</div>
          <Button size="sm" variant="outline" onClick={recheck} disabled={checking}>
            <RefreshCw className={`size-3.5 ${checking ? "animate-spin" : ""}`} /> Re-check
          </Button>
        </div>
        <StatusRow ok={!!health?.db} label="Postgres" detail="config + state store" />
        <StatusRow ok={!!health?.qdrant} label="Qdrant" detail="fact vector index (market_facts)" />
        <StatusRow
          ok={!!health?.anthropic}
          label="Anthropic (chat + reports)"
          detail={health?.anthropic ? "ANTHROPIC_API_KEY set on the API server" : "set ANTHROPIC_API_KEY on the API server"}
        />
        <StatusRow ok={snap.ready === true} label="Chat bridge" detail="/chat reachable from this client" />
        <div className="mt-3 flex items-center gap-2 text-[12px]">
          <span className="text-muted-foreground">Chat model</span>
          <select
            value={snap.model}
            onChange={(e) => setModel(e.target.value)}
            className="rounded border border-outline-subtle bg-surface-input px-2 py-1 text-[12px]"
          >
            {MODEL_OPTIONS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="mt-4 rounded-md border border-outline-subtle bg-surface-panel p-4 text-[12.5px] leading-relaxed text-muted-foreground">
        <div className="text-[13px] font-semibold text-foreground">How it works</div>
        <ol className="mt-2 list-decimal space-y-1 pl-5">
          <li>A daily run scrapes the configured RSS sources into a dated partition.</li>
          <li>Articles are decomposed into atomic, desk-tagged facts with provenance.</li>
          <li>Facts are embedded into the <code className="font-mono">market_facts</code> Qdrant collection.</li>
          <li>A report is generated, and chat answers ad-hoc questions — both grounded via the <code className="font-mono">search_market_facts</code> MCP tool.</li>
        </ol>
      </section>
    </div>
  );
}

function StatusRow({ ok, label, detail }: { ok: boolean; label: string; detail: string }) {
  return (
    <div className="mt-3 flex items-start gap-2">
      {ok ? (
        <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-400" />
      ) : (
        <XCircle className="mt-0.5 size-4 shrink-0 text-amber-400" />
      )}
      <div>
        <div className="text-[12.5px] font-medium text-foreground">{label}</div>
        <div className="text-[11.5px] text-muted-foreground">{detail}</div>
      </div>
    </div>
  );
}
