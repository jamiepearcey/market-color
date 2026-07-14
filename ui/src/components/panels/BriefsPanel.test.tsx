import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BriefsPanel } from "./BriefsPanel";

const BRIEF = {
  date: "2026-07-01",
  desk: "energy",
  markdown: "# Energy desk brief\n\nHormuz reopening unwound the supply-risk premium.",
  data: {
    fixture: true,
    new_fact_count: 5,
    movers: [
      { symbol: "CL=F", name: "WTI Crude", close: 61.42, ret_1d: -0.031, zscore_20d: -2.1 },
      { symbol: "NG=F", name: "Henry Hub Natgas", close: 3.02, ret_1d: 0.002, zscore_20d: 0.1 },
    ],
  },
};

function stubFetch() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/api/briefs")) {
      return new Response(JSON.stringify([{ date: "2026-07-01", desk: "energy" }]), {
        headers: { "content-type": "application/json" },
      });
    }
    if (url.endsWith("/api/briefs/2026-07-01/energy")) {
      return new Response(JSON.stringify(BRIEF), {
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ error: `no stub for ${url}` }), { status: 404 });
  });
}

describe("BriefsPanel", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", stubFetch());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("lists briefs, auto-opens the newest, renders markdown and red/green movers", async () => {
    render(<BriefsPanel />);

    // list entry + auto-selected detail
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Energy desk brief" })).toBeInTheDocument(),
    );
    expect(
      screen.getByText(/Hormuz reopening unwound the supply-risk premium/),
    ).toBeInTheDocument();

    // fixture + new-fact badges from the sibling json
    expect(screen.getByText(/dev fixture/)).toBeInTheDocument();
    expect(screen.getByText(/5 new facts/)).toBeInTheDocument();

    // movers table with red/green ret_1d coloring
    const down = screen.getByText("-3.1%");
    const up = screen.getByText("+0.2%");
    expect(down.className).toContain("text-rose-300");
    expect(up.className).toContain("text-emerald-300");
    expect(screen.getByText("CL=F")).toBeInTheDocument();
  });

  it("shows the empty state when no briefs exist", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("[]", { headers: { "content-type": "application/json" } })),
    );
    render(<BriefsPanel />);
    await waitFor(() => expect(screen.getByText("No briefs yet")).toBeInTheDocument());
  });

  it("shows an error state when the listing fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ error: "boom" }), { status: 500 })),
    );
    render(<BriefsPanel />);
    await waitFor(() => expect(screen.getByText("Could not list briefs")).toBeInTheDocument());
    expect(screen.getByText("boom")).toBeInTheDocument();
  });
});
