import { useRef, useState } from "react";
import { proxy, useSnapshot } from "valtio";
import { Settings, KeyRound, Eye, EyeOff, Save, Loader2, CheckCircle2, ShieldCheck, Plug, CircleX, Server, Palette, Plus, Trash2, Database, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PageHeader } from "@/components/ui/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { actions, store } from "@/lib/store";
import { isDesktop } from "@/lib/api";

const desktop = isDesktop();

export function SettingsView() {
  const snap = useSnapshot(store);
  const { openaiKey, openaiDraft, busy, message, serverUrl, serverUrlDraft, serverOk, serverMessage, theme } = snap.settings;
  // Local UI state per project rule (Valtio over state hooks): reveal toggle for
  // the masked key field.
  const local = useRef(proxy({ reveal: false })).current;
  const ui = useSnapshot(local);

  const dirty = openaiDraft.trim() !== openaiKey;
  const configured = openaiKey.length > 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader icon={<Settings className="size-4 text-icon-tile-foreground" />} title="Settings" />
      <ScrollArea className="flex-1">
        <div className="mx-auto w-full max-w-2xl px-6 py-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="size-4 text-icon-tile-foreground" /> Sidecar connection
              </CardTitle>
              <CardDescription>
                Configure the Celeritas sidecar base URL used by the browser client.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <label htmlFor="server-url" className="text-xs font-medium text-muted-foreground">
                  Base URL
                </label>
                <Input
                  id="server-url"
                  value={serverUrlDraft}
                  autoComplete="off"
                  spellCheck={false}
                  placeholder="http://127.0.0.1:8787"
                  className="font-mono"
                  onChange={(e) => actions.setServerUrlDraft(e.target.value)}
                />
                <p className="text-[11px] text-muted-foreground">
                  Current: <span className="font-mono text-foreground/80">{serverUrl || "not set"}</span>
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={() => void actions.saveSettings()} disabled={busy}>
                  {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
                  Save settings
                </Button>
                <Button variant="outline" onClick={() => void actions.testServerConnection()} disabled={busy}>
                  <Plug className="size-3.5" /> Test connection
                </Button>
                {serverOk != null && (
                  <span className={`flex items-center gap-1.5 text-xs ${serverOk ? "text-emerald-400" : "text-amber-200/90"}`}>
                    {serverOk ? <CheckCircle2 className="size-3.5" /> : <CircleX className="size-3.5" />}
                    {serverMessage}
                  </span>
                )}
              </div>
            </CardContent>
          </Card>

          <div className="mt-4">
            <ThemeCard theme={theme} />
          </div>

          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <KeyRound className="size-4 text-icon-tile-foreground" /> OpenAI API key
              </CardTitle>
              <CardDescription>
                Powers the AI assistant. The key is stored locally on this machine and sent only to OpenAI when you chat.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <label htmlFor="openai-key" className="text-xs font-medium text-muted-foreground">
                  Secret key
                </label>
                <div className="relative">
                  <Input
                    id="openai-key"
                    type={ui.reveal ? "text" : "password"}
                    value={openaiDraft}
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="sk-…"
                    className="pr-9 font-mono"
                    onChange={(e) => actions.setOpenaiDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && dirty && !busy) void actions.saveSettings();
                    }}
                  />
                  <button
                    type="button"
                    title={ui.reveal ? "Hide key" : "Show key"}
                    aria-label={ui.reveal ? "Hide key" : "Show key"}
                    onClick={() => (local.reveal = !local.reveal)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {ui.reveal ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                  </button>
                </div>
                <p className="text-[11px] text-muted-foreground">
                  Get a key from{" "}
                  <span className="font-mono text-foreground/80">platform.openai.com/api-keys</span>.
                </p>
                <p className="text-[11px] text-muted-foreground">
                  Or store a reference like{" "}
                  <span className="font-mono text-foreground/80">
                    secrets-keeper://secret/ai/openai#key
                  </span>{" "}
                  to pull the key from your secrets backend at use-time instead of
                  keeping it here.
                </p>
              </div>

              <div className="flex items-center gap-3 pt-1">
                <Button onClick={() => void actions.saveSettings()} disabled={!dirty || busy}>
                  {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
                  Save
                </Button>
                <span className="text-xs">
                  {message ? (
                    <span className="flex items-center gap-1.5 text-emerald-400">
                      <CheckCircle2 className="size-3.5" /> {message}
                    </span>
                  ) : configured ? (
                    <span className="text-muted-foreground">A key is configured.</span>
                  ) : (
                    <span className="text-muted-foreground">No key configured yet.</span>
                  )}
                </span>
              </div>

              {!desktop && (
                <p className="rounded-md border border-amber-200/20 bg-amber-200/5 px-3 py-2 text-[11px] text-amber-200/80">
                  Preview mode: the key is saved in this browser only. Run the desktop app for the live AI assistant.
                </p>
              )}
            </CardContent>
          </Card>

          <div className="mt-4">
            <SecretsBackendCard />
          </div>
          <div className="mt-4">
            <RegistrySettingsCard />
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

function ThemeCard({ theme }: { theme: "dark" | "light" }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Palette className="size-4 text-icon-tile-foreground" /> Theme
        </CardTitle>
        <CardDescription>
          Switch the control UI between dark and light themes.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex items-center gap-3">
        <Button
          variant={theme === "dark" ? "default" : "outline"}
          onClick={() => void actions.saveTheme("dark")}
        >
          Dark
        </Button>
        <Button
          variant={theme === "light" ? "default" : "outline"}
          onClick={() => void actions.saveTheme("light")}
        >
          Light
        </Button>
      </CardContent>
    </Card>
  );
}

function SecretsBackendCard() {
  const snap = useSnapshot(store);
  const sk = snap.settings.secretsKeeper;
  const local = useRef(proxy({ reveal: false })).current;
  const ui = useSnapshot(local);
  const dirty =
    sk.urlDraft.trim().replace(/\/+$/, "") !== sk.url || sk.tokenDraft.trim().length > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldCheck className="size-4 text-icon-tile-foreground" /> Secrets backend
        </CardTitle>
        <CardDescription>
          Point Celeritas at a{" "}
          <span className="font-mono text-foreground/80">secrets-keeper</span> instance.
          Secret-bearing fields can then hold a{" "}
          <span className="font-mono text-foreground/80">secrets-keeper://…</span>{" "}
          reference resolved from here instead of an inline value.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <label htmlFor="sk-url" className="text-xs font-medium text-muted-foreground">
            Backend URL
          </label>
          <Input
            id="sk-url"
            value={sk.urlDraft}
            autoComplete="off"
            spellCheck={false}
            placeholder="https://127.0.0.1:8200"
            className="font-mono"
            onChange={(e) => actions.setSecretsKeeperUrlDraft(e.target.value)}
          />
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <label htmlFor="sk-token" className="text-xs font-medium text-muted-foreground">
              Token
            </label>
            {sk.tokenSet && (
              <span className="inline-flex items-center gap-1 rounded border border-emerald-500/25 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-medium text-emerald-400">
                <KeyRound className="size-2.5" /> stored in keychain
              </span>
            )}
          </div>
          <div className="relative">
            <Input
              id="sk-token"
              type={ui.reveal ? "text" : "password"}
              value={sk.tokenDraft}
              autoComplete="off"
              spellCheck={false}
              placeholder={sk.tokenSet ? "Enter a new token to replace…" : "skp.…"}
              className="pr-9 font-mono"
              onChange={(e) => actions.setSecretsKeeperTokenDraft(e.target.value)}
            />
            <button
              type="button"
              title={ui.reveal ? "Hide token" : "Show token"}
              aria-label={ui.reveal ? "Hide token" : "Show token"}
              onClick={() => (local.reveal = !local.reveal)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors hover:text-foreground"
            >
              {ui.reveal ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </button>
          </div>
          <p className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
            Held in your OS keychain (write-only) — used as a bearer token to read
            KV-v2 secrets.
            {sk.tokenSet && (
              <button
                type="button"
                onClick={() => void actions.clearSecretsKeeperToken()}
                disabled={sk.busy}
                className="text-destructive underline-offset-2 hover:underline disabled:opacity-50"
              >
                Clear stored token
              </button>
            )}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-1">
          <Button onClick={() => void actions.saveSecretsKeeper()} disabled={!dirty || sk.busy}>
            {sk.busy ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
            Save
          </Button>
          <Button
            variant="outline"
            onClick={() => void actions.testSecretsKeeper()}
            disabled={!sk.urlDraft.trim() || sk.busy}
          >
            <Plug className="size-3.5" /> Test connection
          </Button>
          {sk.message && (
            <span
              className={`flex items-center gap-1.5 text-xs ${
                sk.ok ? "text-emerald-400" : "text-amber-200/90"
              }`}
            >
              {sk.ok ? <CheckCircle2 className="size-3.5" /> : <CircleX className="size-3.5" />}
              {sk.message}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function RegistrySettingsCard() {
  const snap = useSnapshot(store);
  const [draft, setDraft] = useState("");
  const status = snap.settings.registryTestUrl && snap.settings.registryTestMessage
    ? {
        url: snap.settings.registryTestUrl,
        ok: snap.settings.registryTestOk,
        message: snap.settings.registryTestMessage,
      }
    : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="size-4 text-icon-tile-foreground" /> Registry URLs
        </CardTitle>
        <CardDescription>
          Manage the connector registries searched by the Store.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2">
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="https://example.com/registry.manifest.json"
            className="font-mono"
            aria-label="New registry URL"
          />
          <Button
            onClick={() => {
              actions.addRegistryUrl(draft);
              setDraft("");
            }}
            disabled={!draft.trim()}
          >
            <Plus className="size-3.5" /> Add
          </Button>
        </div>
        <div className="space-y-2">
          {snap.registryUrls.map((url) => (
            <div key={url} className="flex flex-wrap items-center gap-2 rounded-md border border-outline-subtle bg-surface-toolbar px-3 py-2">
              <span className="min-w-0 flex-1 truncate font-mono text-[11px]">{url}</span>
              <Button size="sm" variant="outline" onClick={() => void actions.testRegistryUrl(url)}>
                <RefreshCw className="size-3.5" /> Test
              </Button>
              <Button size="sm" variant="ghost" onClick={() => actions.removeRegistryUrl(url)}>
                <Trash2 className="size-3.5" />
              </Button>
            </div>
          ))}
        </div>
        {status && (
          <div className={`text-xs ${status.ok ? "text-emerald-400" : "text-amber-200/90"}`}>
            {status.url}: {status.message}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
