# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
News-conditioned dangerous distribution (v3, realised-calibrated + risk-off signature).

Model:
  - DIRECTIONAL channel: reliable mechanisms (earnings/demand/rates/guidance/M&A/
    supply) shock their factor by S_dir-weighted, count-normalized net pressure
    P(F)/sqrt(N_F)  — the expected signed abnormal move in z units. earnings +0.72
    reliably moves in-direction; data_surprise −0.39 is contrarian; etc.
  - RISK-OFF channel: geopolitics/default/contagion/rating carry high S_mag but
    unreliable per-edge direction (earnings 0.72 vs geopolitics 0.03). They signal a
    risk-off ENVIRONMENT, not a directional bet, so their S_mag-weighted PREVALENCE
    drives one systemic equity_idx-DOWN shock. The engine's factor covariance then
    propagates the safe-haven signature (bonds/gold up, dollar up = flight-to-quality)
    — measured, not hand-coded.
  - Runs an UNCONDITIONAL baseline and reports the 1%-tail DELTA + a flight-to-quality
    readout on the conditional factor moves.

Usage: uv run eventgraph/scripts/news_conditioned_risk.py --graph-dir /tmp/eg_live
"""
import argparse, json, subprocess, collections, os, math
from datetime import date
from pathlib import Path

RISK_MCP = os.path.expanduser("~/tmp/codex-cargo-target/release/risk_mcp")
DAILY = "/Users/jamiepearcey/projects/finance/quant-algos/data/parquet/daily/market=world/asset_class=*/data_0.parquet"
MECH2F = {"monetary_policy": "bonds", "rate_decision": "bonds", "fiscal_policy": "bonds", "data_surprise": "bonds",
          "earnings": "equity_idx", "guidance": "equity_idx", "demand_change": "equity_idx", "competition": "equity_idx",
          "mergers_acquisitions": "equity_idx", "geopolitics": "equity_idx", "default": "equity_idx",
          "contagion": "equity_idx", "rating_action": "equity_idx", "regulation": "equity_idx", "other": "equity_idx",
          "supply_shock": "metals"}
# measured from causal_sensitivity.py: S_dir = directional reliability, S_mag = how HARD it moves
S_DIR = {"earnings": 0.72, "supply_shock": 0.29, "other": 0.22, "rating_action": 0.21, "guidance": 0.20,
         "demand_change": 0.17, "monetary_policy": 0.15, "rate_decision": 0.10, "regulation": 0.08,
         "contagion": 0.04, "geopolitics": 0.03, "mergers_acquisitions": -0.02, "default": -0.11, "data_surprise": -0.39}
S_MAG = {"earnings": 1.19, "default": 1.16, "rate_decision": 1.10, "demand_change": 0.99, "supply_shock": 0.96,
         "geopolitics": 0.95, "other": 0.92, "monetary_policy": 0.92, "data_surprise": 0.86, "mergers_acquisitions": 0.85,
         "contagion": 0.81, "regulation": 0.74, "rating_action": 0.52, "guidance": 0.46}
# EM geography -> regional em_fx factor. The effect_entity id is a slug (e.g. "turkey__country"),
# so match country/currency tokens. A DOWN effect on an EM entity = EM stress = USDxxx UP = em_fx +.
EM_GEO = {
    "em_fx_latam": {"brazil", "brazilian", "mexico", "mexican", "argentina", "argentine", "chile",
                    "chilean", "colombia", "colombian", "peru", "peruvian", "bovespa", "petrobras"},
    "em_fx_emea": {"turkey", "turkish", "lira", "russia", "russian", "ruble", "rouble", "poland",
                   "polish", "zloty", "hungary", "hungarian", "forint", "egypt", "egyptian", "czech",
                   "romania", "nigeria", "nigerian", "ukraine", "ukrainian", "greece", "greek",
                   "south", "africa", "african", "rand", "gazprom", "sberbank"},  # 'south africa' -> tokens
    "em_fx_asia": {"china", "chinese", "yuan", "renminbi", "india", "indian", "rupee", "indonesia",
                   "indonesian", "rupiah", "korea", "korean", "won", "thailand", "thai", "baht",
                   "philippines", "philippine", "taiwan", "taiwanese", "malaysia", "malaysian",
                   "ringgit", "vietnam", "vietnamese", "hong", "kong"},
}
_EM_TOK2REG = {tok: reg for reg, toks in EM_GEO.items() for tok in toks}
def em_region(entity_id):
    name = (entity_id or "").split("__")[0]
    for tok in name.split("_"):
        r = _EM_TOK2REG.get(tok)
        if r: return r
    return None
# risk-off mechanisms: high S_mag, low/unreliable S_dir -> systemic uncertainty, not a per-edge bet.
RISKOFF = {"geopolitics", "default", "contagion", "rating_action"}
# reliable directional mechanisms -> shock their factor by the S_dir-weighted net pressure.
DIRECTIONAL = {"earnings", "demand_change", "guidance", "mergers_acquisitions", "monetary_policy", "rate_decision", "supply_shock", "competition"}
DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}

def rpc(spec):
    msgs = ['{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}',
            json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "run_story", "arguments": {"spec": spec}}})]
    p = subprocess.run([RISK_MCP], input="\n".join(msgs), capture_output=True, text=True, env={**os.environ, "QUANT_ALGOS_DATA": DAILY})
    for line in p.stdout.splitlines():
        try: m = json.loads(line)
        except Exception: continue
        if m.get("id") == 9:
            r = m.get("result", {})
            return {"error": r["content"][0]["text"]} if r.get("isError") else json.loads(r["content"][0]["text"])
    return {"error": "no response", "stderr": p.stderr[-300:]}

def val(m):
    for k in ("pnl", "move", "contribution", "loss", "delta_pnl"):
        if isinstance(m, dict) and k in m: return m[k]
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_live")
    ap.add_argument("--halflife-days", type=float, default=10.0); a = ap.parse_args()
    lake = Path(a.graph_dir) / "lake"
    docd = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}
    tmax = max((d[:10] for d in docd.values() if d), default="2026-07-17")
    def days_ago(ds):
        try: return (date.fromisoformat(tmax) - date.fromisoformat(ds[:10])).days
        except Exception: return 999

    # Two channels:
    #  (1) DIRECTIONAL  — reliable mechanisms (earnings/demand/rates/...) shock their
    #      factor by the S_dir-weighted, count-normalized net pressure  P(F)/sqrt(N_F).
    #  (2) RISK-OFF      — geopolitics/default/contagion/rating carry high S_mag but
    #      unreliable per-edge direction: they signal a risk-off ENVIRONMENT, not a bet.
    #      Their PREVALENCE (S_mag-weighted share of recent edges) drives a systemic
    #      equity_idx-DOWN shock; the engine's own factor covariance then propagates the
    #      safe-haven signature (bonds/gold up = flight-to-quality) — we do NOT hand-code it.
    pres = collections.Counter(); neff = collections.Counter()
    up = collections.Counter(); dn = collections.Counter()          # opposing mass per factor
    riskoff_w = 0.0; total_w = 0.0
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); mech = j.get("mechanism") or "other"
        d = docd.get(j.get("doc_id")); w = 0.5 ** (days_ago(d) / a.halflife_days) if d else 0.3
        total_w += w * S_MAG.get(mech, 0.9)
        if mech in RISKOFF:
            riskoff_w += w * S_MAG.get(mech, 0.9)
        if mech in DIRECTIONAL and j.get("effect_dir") in DIR:
            f = MECH2F.get(mech)
            if f:
                c = DIR[j["effect_dir"]] * S_DIR.get(mech, 0.05) * w         # expected signed z
                pres[f] += c; neff[f] += w
                (up if c >= 0 else dn)[f] += abs(c)                          # track conflict
        # EM geographic routing: ANY directed edge on an EM-country entity is an EM-specific
        # signal (crises are the point) -> route to the regional em_fx factor. down effect = stress = +.
        if j.get("effect_dir") in DIR:
            reg = em_region(j.get("effect_entity"))
            if reg:
                c = -DIR[j["effect_dir"]] * max(S_DIR.get(mech, 0.05), 0.15) * w
                pres[reg] += c; neff[reg] += w
                (up if c >= 0 else dn)[reg] += abs(c)
    # directional shocks (cap 3, drop noise)
    directional = {f: pres[f] / math.sqrt(neff[f]) for f in pres if neff[f] > 0}
    # risk-off overlay: prevalence -> systemic equity-down (SCALE tuned so a corpus that is
    # ~40% risk-off gives ~-1.2sd of extra equity stress).
    riskoff_frac = riskoff_w / total_w if total_w > 0 else 0.0
    RISKOFF_SCALE = 3.0
    riskoff_equity = -min(2.5, riskoff_frac * RISKOFF_SCALE)
    dir_eq = directional.get("equity_idx", 0.0)
    shocks = dict(directional)
    shocks["equity_idx"] = dir_eq + riskoff_equity          # risk-on earnings vs risk-off env
    shocks = {f: round(max(-3.0, min(3.0, v)), 2) for f, v in shocks.items() if abs(v) >= 0.1}

    # ---- DISPERSION channel: cancellation is itself a risk ----
    # Net pressure = 1st moment (mean shock). Opposing mass (up vs dn) = 2nd moment:
    # when earnings-up fights geopolitics-down and they net to ~0, that is NOT calm — it
    # is a wide, bimodal distribution. Route the conflict into a vol MULTIPLIER so the tail
    # fattens even when the mean cancels.
    # conflict_F = D_F·mag_F = 2·min(up,dn)/sqrt(neff) : balance-weighted opposing pressure (sd).
    # Map to a vol multiplier with a SATURATING curve so it discriminates instead of pegging a
    # cap, and stays bounded (a 2x vol = 4x variance is already a big widening).
    VOL_CAP, VOL_SCALE = 2.0, 2.5
    volmap = lambda c: 1.0 + (VOL_CAP - 1.0) * (1.0 - math.exp(-c / VOL_SCALE))
    conflict = {}                                                  # sd of opposing pressure
    for f in set(list(up) + list(dn)):
        conflict[f] = 2.0 * min(up[f], dn[f]) / math.sqrt(neff[f]) if neff[f] > 0 else 0.0
    # equity: the two CHANNELS (directional risk-on vs risk-off env) are opposing scenarios
    cross_eq = 2.0 * min(abs(dir_eq), abs(riskoff_equity))
    conflict["equity_idx"] = math.hypot(conflict.get("equity_idx", 0.0), cross_eq)
    vols = {f: round(volmap(c), 2) for f, c in conflict.items() if volmap(c) >= 1.15}

    print(f"news window ends {tmax}")
    print(f"=== news pressure (two channels) ===")
    print(f"  equity_idx: directional {dir_eq:+.2f}sd (earnings/demand = risk-on)  +  risk-off {riskoff_equity:+.2f}sd"
          f" (from {riskoff_frac:.1%} risk-off prevalence)  =  net {dir_eq+riskoff_equity:+.2f}sd")
    for f, dv in sorted(directional.items(), key=lambda kv: -abs(kv[1])):
        if f != "equity_idx" and abs(dv) >= 0.01:
            print(f"  {f:10}: directional {dv:+.2f}sd")
    print(f"  applied shocks (net, |.|>=0.1): {shocks}")
    if vols:
        print(f"  === dispersion / disagreement (cancellation = risk) ===")
        for f, m in sorted(vols.items(), key=lambda kv: -kv[1]):
            tag = " [earnings-up vs risk-off cancel -> wide]" if f == "equity_idx" and cross_eq > 0.2 else ""
            print(f"  {f:10}: opposing conflict {conflict[f]:.2f}sd -> vol x{m}{tag}")
    # Risk-off environment -> price the move under the CRISIS covariance (worst-equity
    # subsample), where cross-asset correlations rise (contagion). Triggered when the
    # risk-off channel meaningfully bites (net equity is pushed down, or prevalence high).
    net_eq = dir_eq + riskoff_equity
    risk_off_regime = (net_eq < -0.15) or (riskoff_frac > 0.25)
    regime_clause = ";regime=equity_idx:bottom:0.15" if risk_off_regime else ""
    print(f"  regime: {'CRISIS covariance (equity_idx:bottom:0.15) — risk-off environment' if risk_off_regime else 'calm covariance'}")
    print(f"  (engine covariance decides the cross-asset co-moves; the safe-haven signature is measured, not hand-coded)")

    q = "portfolio_pnl_q01_given_story"
    shock_clause = ";shocks=" + ",".join(f"{f}:{v}sd" for f, v in shocks.items()) if shocks else ";shocks=equity_idx:0sd"
    vol_clause = (";vol=" + ",".join(f"{f}:{m}" for f, m in vols.items())) if vols else ""
    base = rpc("name=unconditional;basis=agg;betas=calm;shocks=equity_idx:0sd")
    meanonly = rpc("name=mean;basis=agg;betas=calm" + regime_clause + shock_clause)
    spec = "name=news-conditioned;basis=agg;betas=calm" + regime_clause + shock_clause + vol_clause
    cond = rpc(spec)
    print(f"\nspec: {spec}")
    if "error" in cond: print("engine error:", cond["error"]); return

    bt, mt, ct = base.get(q), meanonly.get(q), cond.get(q)
    print("\n=== dangerous distribution ===")
    if bt is not None and ct is not None:
        print(f"  1%-tail P&L  unconditional {bt:+.4f}  ->  news-conditioned {ct:+.4f}   (Δ {ct-bt:+.4f})")
        if mt is not None and vols:
            print(f"    of which  mean-shift {bt:+.4f}->{mt:+.4f} (Δ {mt-bt:+.4f})  +  dispersion {mt:+.4f}->{ct:+.4f} (Δ {ct-mt:+.4f})")
            print(f"    -> the CANCELLATION widens the tail by {ct-mt:+.4f} on its own (news that nets flat but disagrees loudly)")
    coords = {c["factor"]: c["move"] for c in cond.get("scenario_aggregate_coords", [])}
    if coords:
        print("  conditional factor moves:", {f: round(v, 4) for f, v in coords.items()})
        eq = coords.get("equity_idx", 0.0)
        if eq < -1e-6:  # equity pushed down -> read out which assets act as havens vs sell off with it
            print("  safe-haven vs contagion readout (given equity is down):")
            for f, note in [("fx_major", "DOWN = dollar UP"), ("bonds", "world bond prices"), ("metals", "silver-heavy = risk metal")]:
                mv = coords.get(f)
                if mv is None: continue
                haven = (f == "fx_major" and mv < 0) or (f in ("bonds", "metals") and mv > 0)
                print(f"    {f:10} {mv:+.4f}  {'[SAFE HAVEN ✓ '+note+']' if haven else '[sells off WITH equity = contagion, '+note+']'}")
    # EM regional readout — the "lean into EM": which bloc is most stressed + its drivers.
    em = {f: coords.get(f) for f in ("em_fx_latam", "em_fx_emea", "em_fx_asia") if coords.get(f) is not None}
    if em:
        print("  EM regional stress (em_fx +ve = USD up = that bloc's currencies WEAKEN):")
        for f, mv in sorted(em.items(), key=lambda kv: -kv[1]):
            reg = f.split("_")[-1].upper()
            dv = pres[f] / math.sqrt(neff[f]) if neff.get(f, 0) > 0 else 0.0
            vm = vols.get(f)
            tag = f"  news: net {dv:+.2f}sd" + (f", vol x{vm}" if vm else "")
            flag = "  <- MOST STRESSED" if f == max(em, key=em.get) and mv > 0 else ""
            print(f"    {reg:6} {mv:+.4f}{tag}{flag}")
    los = cond.get("biggest_losers") or []
    if los:
        print("  MOST AT RISK (biggest losers under the news scenario):")
        for m in los[:10]:
            v = val(m); print(f"    {m.get('symbol', m.get('name','?')):12} {round(v,4) if isinstance(v,(int,float)) else json.dumps({k:m[k] for k in list(m)[:3]})}")

if __name__ == "__main__":
    main()
