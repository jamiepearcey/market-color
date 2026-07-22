# /// script
# requires-python = ">=3.10"
# ///
"""Enrich exposure_landscape_emb.json with each event's EVIDENCE (verbatim quote + mechanism + source
article) so the exposure map shows WHAT the event is, not just who's exposed. Reads the lake; fast."""
import json, collections
from pathlib import Path
G=Path("../data/eg_runs/eg100k_graph")
CANON=json.load(open(G/"catalyst_map.json"))
DIRV={"up","down","widen","tighten"}
docmeta={}
for l in open(G/"lake/document.jsonl"):
    j=json.loads(l); docmeta[j["doc_id"]]={"m":(j.get("published_at") or "")[:7],"h":j.get("headline"),"s":j.get("source"),"u":j.get("url"),"d":(j.get("published_at") or "")[:10]}
fmap=collections.defaultdict(list); seen=collections.defaultdict(set)
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); dm=docmeta.get(j.get("doc_id")); c=j.get("cause_entity"); q=(j.get("quote") or "").strip()
    if not dm or dm["m"][:4] not in {"2010","2011","2012"} or not c or j.get("effect_dir") not in DIRV or not q: continue
    ci=CANON.get(c); name=ci["name"] if ci else c.split("__")[0].replace("_"," ").title()
    key=(dm["m"],name)
    if q in seen[key] or len(fmap[key])>=6: continue
    seen[key].add(q); fmap[key].append({"q":q,"mech":j.get("mechanism"),"h":dm["h"],"s":dm["s"],"u":dm["u"],"d":dm["d"]})
land=json.load(open(G/"exposure_landscape_emb.json"))
for e in land["events"]: e["facts"]=fmap.get((e["month"],e["label"]),[])[:6]
json.dump(land,open(G/"exposure_landscape_emb.json","w"))
print(f"enriched {len(land['events'])} events with evidence facts")
