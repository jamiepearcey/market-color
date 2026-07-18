# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Resolution-rule ambiguity, judged by an LLM (not a keyword count) — the clean test of
idea #4. The lexical proxy (ambiguity_test.py) was confounded: its 'clarity anchors'
just re-identified crypto coin-flips. An LLM reads the rule for genuine SUBJECTIVITY
('resolved at the resolver's discretion' = subjective; 'per Binance 1-min close' =
objective) independent of market type, so it isn't confounded that way.

8b first pass (llama-3.1-8b-instant): batch-score each unique resolution DESCRIPTION
0-3 (0=mechanical/objective, 3=highly discretionary), cached by hash. Then the clean
test: among FAVORITES (entry price >= 0.65), does higher rule-subjectivity predict the
favorite being OVER-priced (rules-risk premium)? Cluster-robust by event, monotonicity.

  source data/tmp/groq.env    # GROQ_API_KEY
  uv run eventgraph/scripts/llm_ambiguity.py --graph-dir eventgraph/data/eg_live
"""
import argparse, json, os, re, math, time, hashlib, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from ambiguity_test import settled, daily_hist, first_price

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GAMMA = "https://gamma-api.polymarket.com"
SYS = ("You rate how SUBJECTIVE/DISCRETIONARY a prediction-market RESOLUTION RULE is — how much "
       "human judgment (not a mechanical data lookup) decides the outcome. 0 = fully objective "
       "(a specific price/number/official source/timestamp settles it, no interpretation). "
       "1 = mostly objective, minor interpretation. 2 = notable subjectivity or discretion. "
       "3 = highly subjective / discretionary / dispute-prone / 'resolver decides'. "
       'Return ONLY JSON {"scores":[{"i":<int>,"s":<0|1|2|3>}]}')


def score_rules(uniq, key, model, cache, cf, batch=25):
    todo = [(h, t) for h, t in uniq if h not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        lines = [f'{k+1}. "{t[:300]}"' for k, (h, t) in enumerate(chunk)]
        body = {"model": model, "temperature": 0, "max_tokens": 900, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": "\n".join(lines)}]}
        got = {}
        for att in range(4):
            try:
                r = httpx.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90)
                if r.status_code in (401, 403):
                    raise SystemExit(f"Groq auth failed (HTTP {r.status_code}) — the API key is invalid/"
                                     f"revoked. Refresh GROQ_API_KEY in data/tmp/groq.env and re-run.")
                if r.status_code == 429:
                    time.sleep(3 * (att + 1)); continue
                txt = r.json()["choices"][0]["message"]["content"]; a, b = txt.find("{"), txt.rfind("}")
                got = {x["i"]: x.get("s") for x in json.loads(txt[a:b + 1]).get("scores", []) if isinstance(x, dict)}
                break
            except SystemExit:
                raise
            except Exception:
                time.sleep(2 * (att + 1))
        for k, (h, t) in enumerate(chunk):
            s = got.get(k + 1)
            if isinstance(s, (int, float)) and 0 <= s <= 3:
                cache[h] = float(s); cf.write(json.dumps({"h": h, "s": float(s)}) + "\n"); cf.flush()
        print(f"  scored {min(i+batch,len(todo))}/{len(todo)} unique rules")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--max-fetch", type=int, default=6000)
    ap.add_argument("--per-event", type=int, default=8)
    ap.add_argument("--min-volume", type=float, default=5000.0)
    a = ap.parse_args()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("set GROQ_API_KEY (source data/tmp/groq.env)")
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(parents=True, exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    # collect resolved markets with rules text + early price (paths mostly cached)
    WINDOWS = [("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-07-01"),
               ("2025-07-01", "2026-01-01"), ("2026-01-01", "2026-08-01")]
    events = []
    for lo, hi in WINDOWS:
        off = 0
        while True:
            try:
                page = sess.get(f"{GAMMA}/events", params={"closed": "true", "limit": 100, "offset": off,
                                "end_date_min": lo + "T00:00:00Z", "end_date_max": hi + "T00:00:00Z"}, timeout=30).json()
            except Exception:
                break
            if not isinstance(page, list) or not page:
                break
            events.extend([e for e in page if isinstance(e, dict)]); off += 100
            if off >= 2000:
                break
    obs = []; fetched = 0
    for e in events:
        legs = 0
        for m in e.get("markets") or []:
            if fetched >= a.max_fetch or legs >= a.per_event:
                break
            desc = (m.get("description") or "").strip()
            if len(desc) < 60:
                continue
            y = settled(m)
            try:
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
            except Exception:
                vol = 0
            try:
                clobs = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                clobs = []
            if y is None or vol < a.min_volume or not clobs or not m.get("startDate"):
                continue
            try:
                sd = dt.datetime.fromisoformat(m["startDate"].replace("Z", "+00:00"))
            except Exception:
                continue
            h = daily_hist(clobs[0], cache, sess); fetched += 1
            if not h:
                continue
            p = first_price(h, int(sd.timestamp()))
            if p is None or not (0.02 < p < 0.98):
                continue
            obs.append({"desc": desc, "p": p, "y": y, "bias": p - y,
                        "ev": e.get("title") or e.get("id")})
            legs += 1
        if fetched >= a.max_fetch:
            break
    print(f"{len(obs)} resolved markets with rules + early price")

    # LLM-score UNIQUE descriptions (templated crypto rules collapse to few uniques)
    hp = gd / "llm_ambiguity_cache.jsonl"; cache_s = {}
    if hp.exists():
        for l in open(hp):
            j = json.loads(l); cache_s[j["h"]] = j["s"]
    def dh(t):
        return hashlib.md5(t.encode()).hexdigest()[:16]
    uniq = {}
    for o in obs:
        uniq[dh(o["desc"])] = o["desc"]
    print(f"{len(uniq)} unique resolution rules ({len(cache_s)} already scored)")
    cf = open(hp, "a")
    score_rules(list(uniq.items()), dh, a.model, cache_s, cf)
    for o in obs:
        o["amb"] = cache_s.get(dh(o["desc"]))
    obs = [o for o in obs if o["amb"] is not None]
    print(f"{len(obs)} markets scored; score dist: " +
          ", ".join(f"{s}:{sum(1 for o in obs if round(o['amb'])==s)}" for s in (0, 1, 2, 3)) + "\n")

    def cl(sub):
        byu = collections.defaultdict(list)
        for ev, v in sub:
            byu[ev].append(v)
        c = np.array([np.mean(x) for x in byu.values()])
        return (c.mean(), c.std(ddof=1) / math.sqrt(len(c)), len(c)) if len(c) >= 4 else None

    print("FADE-FAVORITE — bias=price−outcome among favorites (p>=0.65) by LLM rule-subjectivity:")
    means = []
    for s in (0, 1, 2, 3):
        sub = [(o["ev"], o["bias"]) for o in obs if o["p"] >= 0.65 and round(o["amb"]) == s]
        c = cl(sub)
        if c:
            m, se, nev = c; means.append((s, m))
            flag = "  <-- CI excl 0" if abs(m) > 2 * se else ""
            print(f"  subjectivity {s}:  n={len(sub):4} ev={nev:3}  bias {m:+.3f} ±{2*se:.3f}{flag}")
        else:
            print(f"  subjectivity {s}:  (too few)")
    if len(means) >= 3:
        xs, ys = zip(*means)
        print(f"  monotonicity (subjectivity vs favorite over-pricing): "
              f"{np.corrcoef(xs, ys)[0,1]:+.2f}  (>0 => vague rules over-price the favorite)")

    print("\nDISPERSION — mean |price−outcome| by LLM subjectivity (surprise ~ rules risk):")
    for s in (0, 1, 2, 3):
        v = [o["abserr"] if "abserr" in o else abs(o["bias"]) for o in obs if round(o["amb"]) == s]
        if len(v) >= 20:
            print(f"  subjectivity {s}: |err| {np.mean(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
