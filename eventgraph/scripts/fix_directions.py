# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Targeted direction correction (the cheap fix, NOT a corpus re-extraction).

The 8b made stock-direction errors (a $53B bid for PayPal tagged 'down'). Direction is a
per-EDGE property judgeable from the QUOTE alone, and only matters for edges that resolve to
a tradeable US ticker. So we re-judge ONLY those edges (a few hundred short quotes, batched)
with a strong model, and flip only high-confidence disagreements. ~1% the cost of re-running
the corpus.

Writes a new graph-dir (/tmp/eg_live_fixed) with corrected edges; everything else (docs,
entity_symbol, price cache) reused. Every downstream script reads it unchanged.

  source data/tmp/groq.env
  uv run eventgraph/scripts/fix_directions.py --src /tmp/eg_live --out /tmp/eg_live_fixed
"""
import os, json, argparse, time
from pathlib import Path
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DIR = {"up", "down"}
SYSTEM = (
    "You judge the STOCK-PRICE direction of a news item for ONE specific company. Each item gives a "
    "company + ticker + a verbatim quote. Decide whether the news moves THAT COMPANY'S STOCK PRICE up "
    "or down. Reason about who benefits: a takeover BID lifts the TARGET; a lawsuit/probe/guidance-cut/"
    "downgrade/recall/miss lowers the subject; a contract win, beat, or upgrade lifts it; a supplier "
    "gains when its big customer ramps. If the quote is not clearly directional for that stock, return "
    '"unclear". Return ONLY JSON {"results":[{"i":<int>,"dir":"up|down|unclear","confidence":"high|med|low"}]}'
)


def groq(items, model, key):
    lines = [f'{k+1}. [{it["tkr"]}] {it["name"]} — {it["mech"]} — "{it["quote"][:200]}"' for k, it in enumerate(items)]
    body = {"model": model, "temperature": 0, "max_tokens": 1500, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Judge each:\n" + "\n".join(lines)}]}
    for attempt in range(5):
        try:
            r = httpx.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90)
            if r.status_code == 429:
                ra = r.headers.get("retry-after"); time.sleep(min(float(ra), 20) if ra else 3 * (attempt + 1)); continue
            if r.status_code != 200: time.sleep(2 * (attempt + 1)); continue
            txt = r.json()["choices"][0]["message"]["content"]; a, b = txt.find("{"), txt.rfind("}")
            return {r_["i"]: r_ for r_ in json.loads(txt[a:b + 1]).get("results", []) if isinstance(r_, dict)}
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="/tmp/eg_live"); ap.add_argument("--out", default="/tmp/eg_live_fixed")
    ap.add_argument("--model", default="openai/gpt-oss-120b"); ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--flip-conf", default="high", choices=["high", "med"]); a = ap.parse_args()
    key = os.environ.get("GROQ_API_KEY")
    if not key: raise SystemExit("set GROQ_API_KEY (source data/tmp/groq.env)")
    src = Path(a.src); out = Path(a.out); (out / "lake").mkdir(parents=True, exist_ok=True)

    # US securities only (the signal set)
    sym = {}
    for l in open(src / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j.get("kind") == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]

    lines = [json.loads(l) for l in open(src / "lake" / "causal_event_edge.jsonl")]
    items = []
    for idx, j in enumerate(lines):
        e = j.get("effect_entity")
        if e in sym and j.get("effect_dir") in DIR and (j.get("quote") or "").strip():
            items.append({"idx": idx, "tkr": sym[e], "name": e.split("__")[0].replace("_", " "),
                          "mech": j.get("mechanism") or "other", "quote": j["quote"], "orig": j["effect_dir"]})
    print(f"{len(lines)} total edges; {len(items)} resolve to a US ticker -> re-judging direction ({a.model})")

    conf_ok = {"high"} if a.flip_conf == "high" else {"high", "med"}
    agree = flip = unclear = 0
    for i in range(0, len(items), a.batch):
        batch = items[i:i + a.batch]
        res = groq(batch, a.model, key)
        for k, it in enumerate(batch):
            r = res.get(k + 1)
            if not r: continue
            d = (r.get("dir") or "").lower(); c = (r.get("confidence") or "low").lower()
            if d not in DIR: unclear += 1; continue
            if d == it["orig"]: agree += 1
            elif c in conf_ok:
                lines[it["idx"]]["effect_dir"] = d; lines[it["idx"]]["dir_fixed"] = True; flip += 1
        print(f"  batch {i//a.batch+1}/{(len(items)+a.batch-1)//a.batch}: agree {agree} flip {flip} unclear {unclear}")

    judged = agree + flip
    print(f"\njudged {judged} | 8b agreed {agree} ({agree/max(judged,1):.0%}) | corrected {flip} | unclear {unclear}")

    # write corrected graph-dir (reuse everything else)
    with open(out / "lake" / "causal_event_edge.jsonl", "w") as f:
        for j in lines: f.write(json.dumps(j) + "\n")
    (out / "lake" / "document.jsonl").write_text((src / "lake" / "document.jsonl").read_text())
    (out / "entity_symbol.jsonl").write_text((src / "entity_symbol.jsonl").read_text())
    pc = out / "prices"
    if not pc.exists():
        try: pc.symlink_to((src / "prices").resolve())
        except Exception: pc.mkdir(exist_ok=True)
    print(f"wrote corrected graph-dir -> {out}")


if __name__ == "__main__":
    main()
