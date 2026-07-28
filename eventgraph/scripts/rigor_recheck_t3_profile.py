# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx","sentence-transformers","torch","scipy","scikit-learn"]
# ///
"""PROFILE-SOURCE VARIANT of rigor_recheck_t3 (F31).

F31's firm profiles are built from CAUSAL-EDGE quotes only, which is the
high-precision / low-recall path: median 3 documents per firm, and 81% of firms
rest on fewer than 10. This variant changes ONE thing -- where profile text comes
from -- and holds the universe, events, baskets, windows and non-mention
exclusion identical, so the forward text increment is comparable to F31's +0.028.

  --profile edge     causal-edge quote + headline          (F31 baseline)
  --profile subject  documents whose HEADLINE names the firm (primitive-pipeline
                     attribution: ~2.6x more firm-doc pairs)
  --profile both     union of the two

Original docstring follows.

FABLE #1 + R3: the decisive rigor recheck. For each event, does TEXT (embedding exposure) predict a
non-mentioned firm's co-movement with the event basket INCREMENTAL to the missing baseline = PRE-EVENT
trailing correlation with the basket? Report CONTEMPORANEOUS (the F22/F24 claim) AND FORWARD (R3: predictive
risk). Event-level inference: per-event IC, mean + bootstrap CI + t ACROSS events (not pooled). Leakage-clean:
trailing/baseline use only pre-event months; target windows are strictly separate."""
import json, collections, csv, sys, math, re, argparse, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=140
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
BROADCANON={"macro","monetary","fiscal"}; DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
_ap=argparse.ArgumentParser(); _ap.add_argument("--profile",choices=["edge","subject","both","edge_random","causal_lex"],default="edge")
_ap.add_argument("--chain",action="store_true",
  help="CAUSAL-CHAIN SEARCH. Adds a 2-hop graph feature: is the event's cause "
       "connected by an entity edge to one of the candidate firm's own drivers? "
       "Hubs excluded (macro hubs connect everything). Tests whether the CHAIN "
       "adds over the causal EMBEDDING, and vice versa.")
_ap.add_argument("--hub-pct",type=float,default=2.0,
  help="top %% of entities by degree excluded from bridge formation")
_ap.add_argument("--modality",choices=["any","forecast","happened"],default="any",
  help="Bias profiles toward FORWARD-LOOKING text. The extractor already tags "
       "causal edges with modality; 26%% are 'forecast' ('should drop more in the "
       "fourth quarter due to lower crude prices'). Those are EXPOSURE assertions "
       "rather than event reporting, which is what the exposure task actually needs.")
_ap.add_argument("--stability",action="store_true",
  help="Downrank unstable terms: keep only terms whose sign is consistent across "
       "both halves of the TRAINING period. Directly targets F55's failure mode -- "
       "the model learned 2010-12 topic vocabulary that did not transfer.")
_ap.add_argument("--learn",action="store_true",
  help="LEARNED SPARSE SELECTION. Instead of scoring by raw term overlap, learn "
       "WHICH shared terms predict forward co-movement (L1, so most weights go to "
       "zero) with a strict train-on-early / test-on-late split. L1 before a net "
       "on purpose: n is 114 events, and the point is to READ the selected terms "
       "-- interpretability is exactly what the raw overlap score failed at.")
_ap.add_argument("--min-idf",type=float,default=0.0,
  help="Drop terms below this idf before matching. Inspecting the actual matches "
       "showed the top driver is 'percent', with 'inc', 'much', 'group' also "
       "scoring -- generic newswire vocabulary, not exposure. This tests whether "
       "the statistic survives removing it.")
_ap.add_argument("--examples",type=int,default=0,
  help="Print WORKED EXAMPLES instead of test statistics: for N events, the "
       "actual event, the firms the method ranks as exposed while never being "
       "mentioned, the SHARED TERMS that drove each match, and what those firms' "
       "correlation with the affected basket actually did next.")
_ap.add_argument("--half-life",type=float,default=0.0,
  help="Exponential recency decay on a firm's quotes, in MONTHS (0 = off). "
       "preprofile treats a quote from 2010-01 identically to one from 2012-11 "
       "when scoring a 2012-12 event, and under best-match pooling a STALE "
       "perfect match beats a FRESH good one. Exposure profiles decay; this "
       "weights each quote by 0.5**(age/half_life) before pooling.")
_ap.add_argument("--sim",choices=["dense","lexical","hybrid"],default="dense",
  help="Similarity space. dense = all-MiniLM cosine (fuzzy: blurs 'second-line' "
       "and 'breakthrough' into generic pharma). lexical = idf-weighted TERM "
       "overlap, which is exact where specificity lives. hybrid = within-event "
       "z-score average of both. F45 already found tf-idf (+0.047) beating the "
       "centred embedding (+0.033) on the exposure task, so sparse is the "
       "favourite, not the challenger.")
_ap.add_argument("--nq-control",action="store_true",
  help="Control for log(number of pre-event quotes). max/top3 pooling gives a "
       "firm with 80 quotes 80 chances at a high match and a firm with 2 only 2, "
       "so the score is mechanically increasing in coverage -- and coverage tracks "
       "size, which tracks co-movement. This is the same class of confound as "
       "embedding anisotropy and must be excluded before believing the pooling win.")
_ap.add_argument("--pool",choices=["mean","max","top3"],default="mean",
  help="How firm quotes are compared to event quotes. mean = cosine of the MEAN "
       "profile vs the MEAN event vector (original). That averaging destroys "
       "information specificity: 80 quotes about a pharma name average into "
       "'generic pharma', losing 'second-line trial success, breakthrough'. "
       "max = the single best quote-to-quote match; top3 = mean of the best 3. "
       "If specificity carries the signal, max/top3 must beat mean.")
_ap.add_argument("--center",action="store_true",
  help="Mean-centre the embedding space before averaging. Kills the "
       "centroid-proximity confound: a heavily-covered firm has a diverse profile "
       "whose mean sits near the corpus centroid, hence near ANY event vector -- "
       "and heavily-covered firms are large and co-move more.")
_ap.add_argument("--sector-control",action="store_true",
  help="Add a SAME-SECTOR dummy alongside trailing correlation in the partial. "
       "Tests the deflationary hypothesis that the text profile is a sector "
       "proxy doing covariance shrinkage rather than carrying news information.")
_ap.add_argument("--strict-mention",action="store_true",
  help="ALSO exclude firms named in the HEADLINE of the event's documents. The "
       "original T2 exclusion covers causal-effect / sentiment / relation targets "
       "only, so a firm named in an event headline counted as 'non-mentioned'. "
       "Subject attribution surfaces exactly those firms, so this is the "
       "adversarial check on whether the subject result is leakage.")
ARGS=_ap.parse_args()
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
CANON=json.load(open(G/"catalyst_map.json"))
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
head={json.loads(l)["doc_id"]:(json.loads(l).get("headline") or "") for l in open(G/"lake/document.jsonl")}
_DOCTXT={}
if ARGS.profile=="edge_random":
    for _l in open(G/"lake/chunk.jsonl"):
        _j=json.loads(_l)
        if _j.get("text"): _DOCTXT[_j["doc_id"]]=_DOCTXT.get(_j["doc_id"],"")+" "+_j["text"]
    print(f"[edge_random] loaded text for {len(_DOCTXT)} documents")
# T2: broaden "mentioned" = any firm appearing as causal effect OR sentiment target OR relation endpoint in the event's docs
freq=collections.Counter(); comp_qm=collections.defaultdict(list); ev_q=collections.defaultdict(list); ev_named=collections.defaultdict(lambda: collections.defaultdict(int)); ev_docs=collections.defaultdict(set)
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); did=j.get("doc_id"); m=docm.get(did); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    if ARGS.modality!="any" and j.get("modality")!=ARGS.modality: continue
    freq[sym[e]]+=1; q=(j.get("quote") or "").strip()
    if q and len(comp_qm[sym[e]])<80:
        # edge_random: SAME document, SAME month, SAME count -- but a random
        # sentence instead of the extracted causal span. Isolates "the causal span
        # carries it" from "short selected text from the right document does".
        if ARGS.profile=="edge_random":
            _t=_DOCTXT.get(did,"")
            _sents=[x.strip() for x in re.split(r"(?<=[.!?]) +", _t) if 40<=len(x.strip())<=300]
            if not _sents: continue
            _pick=_sents[(len(comp_qm[sym[e]])*7+len(did))%len(_sents)]
            comp_qm[sym[e]].append((m,(_pick+". "+head.get(did,""))[:200]))
        else:
            comp_qm[sym[e]].append((m,(q+". "+head.get(did,""))[:200]))
    ci=CANON.get(c); key=ci["key"] if ci else c; typ=ci["type"] if ci else c.split("__")[-1]; lab=ci["name"] if ci else c
    if (ci and typ in BROADCANON) or (not ci and c.split("__")[-1] in MACRO_T): continue
    if q and len(ev_q[(m,key)])<40: ev_q[(m,key)].append((lab+": "+q)[:200])
    ev_named[(m,key)][sym[e]]+=DIRV[d]; ev_docs[(m,key)].add(did)
# ---- PROFILE SOURCE: optionally rebuild/augment comp_qm from HEADLINE-SUBJECT docs.
# Only the profile TEXT changes; universe, events, baskets and windows are untouched.
_CAUSAL_LEX = re.compile(
    r"\b(after|amid|on (concern|speculation|optimism|signs|bets|prospects|expectations)|"
    r"driven by|spurred by|boosted by|hurt by|weighed (down )?by|led by|"
    r"following|because of|due to|blamed|prompted by|triggered by|"
    r"in response to|on report|on news|sparked)\b", re.I)
if ARGS.profile in ("subject", "both", "causal_lex") or ARGS.strict_mention:
    _cand = {s for s, _ in freq.most_common(300)}
    _ALIAS = re.compile(r"INSERT INTO entity_alias \(alias,entity_id\) VALUES \('(.+?)','(.+?)'\)")
    _BAD = re.compile(r"^(the|a|an|inc|plc|corp|ltd|sa|ag|co|group|holdings?)$", re.I)
    _alias = {}
    for _line in open(G/"pg_upsert.sql", errors="ignore"):
        _m = _ALIAS.search(_line)
        if not _m: continue
        _a, _eid = _m.group(1).replace("''", "'"), _m.group(2)
        if _eid in sym and sym[_eid] in _cand and 4 <= len(_a) <= 60 \
           and not _BAD.match(_a) and re.search(r"[A-Za-z]{3}", _a):
            _alias.setdefault(_a, _eid)
    _order = sorted(_alias, key=len, reverse=True)
    _chunk1 = {}
    for _l in open(G/"lake/chunk.jsonl"):
        _j = json.loads(_l)
        if _j.get("seq") == 0 and _j.get("text"): _chunk1[_j["doc_id"]] = _j["text"]
    _HEADFIRMS = {}
    if ARGS.profile in ("subject", "causal_lex"): comp_qm = collections.defaultdict(list)
    _added = 0
    for _did, _mo in docm.items():
        if not _mo or _mo[:4] not in {"2010","2011","2012"}: continue
        _h = head.get(_did, "")
        if not _h: continue
        # causal_lex: keep ONLY documents whose headline asserts a driver relation.
        # This is the lexical stand-in for what a causal edge marks -- the test of
        # whether the LLM's DOCUMENT SELECTION (F44) is reproducible without it.
        if ARGS.profile == "causal_lex" and not _CAUSAL_LEX.search(_h): continue
        _pad = " " + _h + " "
        for _a in _order:
            if _a in _pad:
                _s = sym[_alias[_a]]
                _HEADFIRMS.setdefault(_did, set()).add(_s)
                if ARGS.profile in ("edge", "edge_random"): continue
                if len(comp_qm[_s]) < 80:
                    comp_qm[_s].append((_mo, (_h + ". " + _chunk1.get(_did, ""))[:200])); _added += 1
    print(f"[profile={ARGS.profile}] aliases={len(_alias)} subject firm-doc entries added={_added} "
          f"firms with profile text={len(comp_qm)}")

# ---- CAUSAL CHAIN: entity->entity adjacency, firm driver sets, hub list -------
# The chain is built from the SAME causal edges that carry the surviving signal
# (F43/F44), not from co-mention. bridge = an edge links the event's cause to one
# of the candidate firm's own drivers -- second-order exposure by structure rather
# than by sector (F46 killed the sector proxy).
# ---- CAUSAL CHAIN, WINDOWED. Built per-month so that (a) the driver set for a
# firm is only what drove it in the SAME trailing window as the correlation, and
# (b) no edge from after the event month can enter the feature. Aggregating all of
# 2010-12 gave median 3 / mean 6.4 / max 129 drivers per firm -- diluted enough to
# bridge to almost anything, and leaky. Windowed it is median 1 / mean 1.8.
# Hub degree is also WITHIN-WINDOW, per graph_transitivity.py's spec.
_EDGES_MO = collections.defaultdict(list)
if ARGS.chain:
    for _l in open(G/"lake/causal_event_edge.jsonl"):
        _j = json.loads(_l)
        _c, _e = _j.get("cause_entity"), _j.get("effect_entity")
        _m2 = docm.get(_j.get("doc_id"))
        if not _c or not _e or not _m2 or _m2[:4] not in {"2010","2011","2012"}:
            continue
        _EDGES_MO[_m2].append((_c, _e))
    print(f"[chain] windowed: {sum(len(v) for v in _EDGES_MO.values())} edges "
          f"across {len(_EDGES_MO)} months; hub degree computed within-window "
          f"(top {ARGS.hub_pct}%)")

_WIN_CACHE = {}
def _window_graph(months):
    """adjacency, firm->drivers and hub set, built ONLY from `months`."""
    k = tuple(months)
    if k in _WIN_CACHE: return _WIN_CACHE[k]
    adj = collections.defaultdict(set); drv = collections.defaultdict(set)
    deg = collections.Counter()
    for _m in months:
        for _c, _e in _EDGES_MO.get(_m, ()):
            adj[_c].add(_e); adj[_e].add(_c); deg[_c] += 1; deg[_e] += 1
            if _e in sym: drv[sym[_e]].add(_c)
    nk = max(1, int(len(deg) * ARGS.hub_pct / 100))
    hubs = {e for e, _ in deg.most_common(nk)}
    _WIN_CACHE[k] = (adj, drv, hubs)
    return _WIN_CACHE[k]

# broader mention set from sentiment + relation targets sharing the event's docs
doc_ment=collections.defaultdict(set)
for l in open(G/"lake/sentiment_annotation.jsonl"):
    j=json.loads(l); t=j.get("target_entity")
    if t in sym: doc_ment[j.get("doc_id")].add(sym[t])
if (G/"lake/relation_edge.jsonl").exists():
    for l in open(G/"lake/relation_edge.jsonl"):
        j=json.loads(l)
        for k in ("source_entity","target_entity"):
            if j.get(k) in sym: doc_ment[j.get("doc_id")].add(sym[j[k]])
if ARGS.strict_mention:
    for _d, _fs in _HEADFIRMS.items(): doc_ment[_d] |= _fs
    print(f"[strict-mention] headline-named firms folded into the exclusion set "
          f"for {len(_HEADFIRMS)} documents")
years={"2010","2011","2012"}
def ret(t): return logret(yahoo(t,cache,p1,p2))
spdr={e:ret(e) for e in SPDR}
uni=[]
for s,_ in freq.most_common():
    if len(uni)>=N_UNIV: break
    r=ret(s)
    if sum(1 for d in r if d[:4] in years)>400: uni.append(s)
R={s:ret(s) for s in uni}
alld=sorted({d for s in uni for d in R[s] if d[:4] in years and d in Fmap and all(d in spdr[e] for e in SPDR)})
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]
def design(days):
    X=np.column_stack([[Fmap[d][i] for d in days] for i in mcol]+[[spdr[e][d] for d in days] for e in SPDR])
    X=np.nan_to_num(X-np.nanmean(X,0)); return np.column_stack([np.ones(len(days)),X])
R2={}
for s in uni:
    days=[d for d in alld if d in R[s]]
    if len(days)<200: continue
    y=np.array([R[s][d] for d in days]); X=design(days); b,*_=np.linalg.lstsq(X,y,rcond=None); R2[s]=dict(zip(days,y-X@b))
uni=[s for s in uni if s in R2 and comp_qm.get(s)]
import torch
from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2",device="mps" if torch.backends.mps.is_available() else "cpu")
evs=[(m,k) for (m,k),nm in ev_named.items() if len([x for x in nm if x in R2])>=3 and ev_q[(m,k)]]
allq=[]; qmeta=[]; im={}
for s in uni:
    for (qm,qt) in comp_qm[s]: qmeta.append((s,qm)); allq.append(qt)
for k in evs: im[("E",k)]=(len(allq),len(allq)+len(ev_q[k])); allq+=ev_q[k]
emb=model.encode(allq,batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
if ARGS.center:
    _mu=emb.mean(0); emb=emb-_mu
    _n=np.linalg.norm(emb,axis=1,keepdims=True); emb=emb/np.where(_n>0,_n,1)
    print(f"[center] embedding space mean-centred (||mu||={np.linalg.norm(_mu):.3f})")
qbyfirm=collections.defaultdict(list)
for gi,(s,qm) in enumerate(qmeta): qbyfirm[s].append((gi,qm))
def ev_vec(k): a,b=im[("E",k)]; v=emb[a:b].mean(0); n=np.linalg.norm(v); return v/n if n else v
ev={k:ev_vec(k) for k in evs}
# ---- LEXICAL SPACE: idf-weighted term vectors over the SAME quotes ------------
import scipy.sparse as _sp
_STOP=set("the a an and or of to in on for at by with from as is are was were be been "
          "it its this that these those has have had will would may might can could "
          "said says say after before amid over under more most than then so but".split())
_TOK=re.compile(r"[a-z][a-z0-9-]{2,}")
def _terms(t): return {w for w in _TOK.findall(t.lower()) if w not in _STOP}
_vocab={}; _rows=[]; _cols=[]; _df=collections.Counter()
_qterms=[_terms(q) for q in allq]
for _ts in _qterms:
    for _w in _ts: _df[_w]+=1
_N=len(allq)
for _i,_ts in enumerate(_qterms):
    for _w in _ts:
        _j=_vocab.setdefault(_w,len(_vocab))
        _rows.append(_i); _cols.append(_j)
_idf=np.zeros(len(_vocab),dtype=np.float32)
for _w,_j in _vocab.items(): _idf[_j]=np.log(1.0+_N/(1.0+_df[_w]))
_LEX=_sp.csr_matrix((np.ones(len(_rows),dtype=np.float32),(_rows,_cols)),
                    shape=(_N,len(_vocab)))
if ARGS.min_idf>0:
    _keep=(_idf>=ARGS.min_idf)
    print(f"[min-idf {ARGS.min_idf}] keeping {_keep.sum()} of {len(_idf)} terms")
    _idf=_idf*_keep
_LEX=_LEX.multiply(_idf[None,:]).tocsr()
_nrm=np.sqrt(_LEX.multiply(_LEX).sum(axis=1)).A.ravel(); _nrm[_nrm==0]=1.0
_LEX=_sp.diags(1.0/_nrm) @ _LEX
print(f"[lexical] {_N} quotes, {len(_vocab)} terms")

def preprofile(s,evmonth):
    """mean-pooled profile (original). Returns None if <2 pre-event quotes."""
    idxs=[gi for gi,qm in qbyfirm.get(s,()) if qm< evmonth]
    if len(idxs)<2: return None
    v=emb[idxs].mean(0); n=np.linalg.norm(v); return v/n if n else None

def pre_idxs(s,evmonth):
    return [gi for gi,qm in qbyfirm.get(s,()) if qm< evmonth]

def exposure(s,key,evmonth):
    """Firm-to-event score under the chosen pooling. mean = cosine of averages;
    max/top3 = best quote-to-quote match, which PRESERVES specificity instead of
    averaging it away."""
    idxs=pre_idxs(s,evmonth)
    if len(idxs)<2: return None
    a,b=im[("E",key)]
    # recency weights, aligned to `idxs`
    _w=None
    if ARGS.half_life>0:
        _ei=idx.get(evmonth)
        if _ei is not None:
            _ages=np.array([max(0,_ei-idx.get(qm,_ei))
                            for gi,qm in qbyfirm.get(s,()) if qm<evmonth],dtype=float)
            if len(_ages)==len(idxs):
                _w=np.power(0.5,_ages/ARGS.half_life).astype(np.float32)
    def _pool(M):
        if M.size==0: return None
        if _w is not None and M.shape[0]==len(_w): M=M*_w[:,None]
        if ARGS.pool=="max": return float(M.max())
        k=min(3,M.size); return float(np.sort(M,axis=None)[-k:].mean())
    def _dense():
        if ARGS.pool=="mean":
            pp=preprofile(s,evmonth)
            return None if pp is None else float(ev[key]@pp)
        if b<=a: return None
        return _pool(emb[idxs] @ emb[a:b].T)
    def _lex():
        if b<=a: return None
        return _pool((_LEX[idxs] @ _LEX[a:b].T).toarray())
    if ARGS.sim=="dense":   return _dense()
    if ARGS.sim=="lexical": return _lex()
    d,l=_dense(),_lex()
    return None if (d is None or l is None) else (d,l)   # hybrid: combined per-event
mo=[f"{y}-{mm:02d}" for y in (2010,2011,2012) for mm in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def cw(a,basket,days):
    cm=[d for d in days if d in a and d in basket]
    if len(cm)<12: return None
    x=np.array([a[d] for d in cm]);y=np.array([basket[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
def _f(x): return np.array([np.nan if v is None else v for v in x],float)
def ic(x,y):
    x=_f(x);y=_f(y); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<8 or x.std()==0 or y.std()==0: return None
    return float(np.corrcoef(x,y)[0,1])
def partial(a,b,*ctrls):
    """Partial correlation of a,b controlling for ONE OR MORE covariates."""
    a,b=_f(a),_f(b); C=[_f(c) for c in ctrls]
    m=np.isfinite(a)&np.isfinite(b)
    for c in C: m&=np.isfinite(c)
    a,b=a[m],b[m]; C=[c[m] for c in C]
    if len(a)<10: return None
    X=np.column_stack([np.ones(len(a))]+C)
    if np.linalg.matrix_rank(X)<X.shape[1]: X=np.column_stack([np.ones(len(a))]+C[:1])
    def res(v):
        bb,*_=np.linalg.lstsq(X,v,rcond=None); return v-X@bb
    ra,rb=res(a),res(b)
    return None if ra.std()==0 or rb.std()==0 else float(np.corrcoef(ra,rb)[0,1])
_CAUSE_ENTS = collections.defaultdict(set)
if ARGS.chain:
    for _l in open(G/"lake/causal_event_edge.jsonl"):
        _j = json.loads(_l)
        _c = _j.get("cause_entity"); _m2 = docm.get(_j.get("doc_id"))
        if not _c or not _m2 or _m2[:4] not in {"2010","2011","2012"}: continue
        _ci = CANON.get(_c)
        _CAUSE_ENTS[_ci["key"] if _ci else _c].add(_c)
import gics
_SEC={s_:gics.sector(s_) for s_ in uni}
def _modal_sector(M):
    c=collections.Counter(_SEC.get(x) for x in M if _SEC.get(x) not in (None,"UNK"))
    return c.most_common(1)[0][0] if c else None
_EXROWS=[]; _LEARN_X=[]; _LEARN_Y=[]; _LEARN_M=[]; _LEARN_TR=[]
def _shared_terms(fs,evk,evmonth,topn=6):
    """the actual idf-weighted terms behind the best firm-quote/event-quote match"""
    idxs=pre_idxs(fs,evmonth); a,b=im[("E",evk)]
    if not idxs or b<=a: return []
    Msub=(_LEX[idxs] @ _LEX[a:b].T).toarray()
    fi,ei=np.unravel_index(np.argmax(Msub),Msub.shape)
    t1,t2=_qterms[idxs[fi]],_qterms[a+ei]
    sh=[(w,_idf[_vocab[w]]) for w in (t1&t2) if w in _vocab]
    sh.sort(key=lambda x:-x[1])
    return [w for w,_ in sh[:topn]]
per=collections.defaultdict(list)   # metric -> list of per-event IC
for (m,key) in evs:
    i=idx[m]
    if i<3 or i+3>=len(mo): continue
    ment=set(ev_named[(m,key)]) | {t for did in ev_docs[(m,key)] for t in doc_ment.get(did,())}
    M=[s for s in ev_named[(m,key)] if s in R2]
    if len(M)<3: continue
    preW=[d for d in alld if d[:7] in mo[i-3:i]]; conW=[d for d in alld if d[:7] in mo[i:i+2]]; fwdW=[d for d in alld if d[:7] in mo[i+2:i+4]]
    def basket(days): return {d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in days if sum(1 for s in M if d in R2[s])>=2}
    bp,bc,bf=basket(preW),basket(conW),basket(fwdW)
    E=[]; TR=[]; CO=[]; FW=[]; SEC=[]; BRIDGE=[]; NQ=[]; _HYB=[]; _UNI_USED=[]
    if ARGS.chain:
        _ADJ, _DRIVERS, _HUBS = _window_graph(mo[i-3:i])   # SAME window as trailing corr
    _ms=_modal_sector(M)
    for s in uni:
        if s in ment: continue                       # T2: broadened non-mentioned
        _x=exposure(s,(m,key),m)
        if _x is None: continue
        _UNI_USED.append(s)
        if ARGS.learn:
            _ii=pre_idxs(s,m); _a2,_b2=im[("E",(m,key))]
            if _ii and _b2>_a2:
                _fv=np.asarray(_LEX[_ii].max(axis=0).todense()).ravel()
                _ev2=np.asarray(_LEX[_a2:_b2].max(axis=0).todense()).ravel()
                _LEARN_X.append(np.minimum(_fv,_ev2))   # the SHARED-term vector
                _LEARN_M.append(idx.get(m,0))
        if ARGS.sim=="hybrid": _HYB.append(_x); _x=0.0
        NQ.append(float(np.log1p(len(pre_idxs(s,m)))))
        E.append(_x); TR.append(cw(R2[s],bp,preW)); CO.append(cw(R2[s],bc,conW)); FW.append(cw(R2[s],bf,fwdW))
        SEC.append(1.0 if (_ms and _SEC.get(s)==_ms) else 0.0)
        if ARGS.chain:
            # the event's cause entities = those whose canonical key is this event
            _dr = {d for d in _DRIVERS.get(s, ()) if d not in _HUBS}
            _hit = 0.0
            for _c0 in _CAUSE_ENTS.get(key, ()):
                if _c0 in _HUBS: continue
                if _c0 in _dr or (_ADJ.get(_c0, set()) & _dr):
                    _hit = 1.0; break
            BRIDGE.append(_hit)
    if ARGS.sim=="hybrid" and _HYB:
        _d=np.array([h[0] for h in _HYB]); _l=np.array([h[1] for h in _HYB])
        _z=lambda v: (v-v.mean())/(v.std() if v.std() else 1.0)
        E=list(_z(_d)+_z(_l))
    if ARGS.examples and len(_EXROWS)<ARGS.examples*1:
        _cands=[]
        for _si,_s2 in enumerate(_UNI_USED):
            if _si>=len(E): break
            _cands.append((E[_si],_s2,TR[_si],FW[_si]))
        _cands=[c for c in _cands if c[2] is not None and c[3] is not None]
        if len(_cands)>=8:
            _cands.sort(key=lambda x:-x[0])
            _EXROWS.append({"month":m,"key":key,"basket":sorted(M),
                            "quote":(ev_q[(m,key)][0] if ev_q[(m,key)] else ""),
                            "top":_cands[:5],"bot":_cands[-5:]})
    if ARGS.learn:
        for _k2 in range(len(E)):
            if _k2<len(FW) and FW[_k2] is not None and TR[_k2] is not None:
                _LEARN_Y.append(FW[_k2]); _LEARN_TR.append(TR[_k2])
            else:
                _LEARN_Y.append(np.nan); _LEARN_TR.append(np.nan)
    if sum(1 for v in CO if v is not None)<10: continue
    per["emb->contemp"].append(ic(E,CO)); per["trail->contemp"].append(ic(TR,CO)); per["emb|trail->contemp"].append(partial(E,CO,TR))
    per["emb->forward"].append(ic(E,FW)); per["trail->forward"].append(ic(TR,FW)); per["emb|trail->forward"].append(partial(E,FW,TR))
    if ARGS.chain and len(BRIDGE)==len(FW):
        per["bridge->forward"].append(ic(BRIDGE,FW))
        per["bridge|trail->forward"].append(partial(BRIDGE,FW,TR))
        per["emb|trail+bridge->forward"].append(partial(E,FW,TR,BRIDGE))
        per["bridge|trail+emb->forward"].append(partial(BRIDGE,FW,TR,E))
    if ARGS.nq_control:
        per["nquotes->forward"].append(ic(NQ,FW))
        per["emb|trail+nq->contemp"].append(partial(E,CO,TR,NQ))
        per["emb|trail+nq->forward"].append(partial(E,FW,TR,NQ))
    if ARGS.sector_control:
        per["sector->contemp"].append(ic(SEC,CO)); per["sector->forward"].append(ic(SEC,FW))
        per["emb|trail+sector->contemp"].append(partial(E,CO,TR,SEC))
        per["emb|trail+sector->forward"].append(partial(E,FW,TR,SEC))
if ARGS.learn:
    from sklearn.linear_model import LassoCV
    X=np.array(_LEARN_X,dtype=np.float32); y=np.array(_LEARN_Y,dtype=float)
    tr=np.array(_LEARN_TR,dtype=float); mth=np.array(_LEARN_M)
    ok=np.isfinite(y)&np.isfinite(tr)
    X,y,tr,mth=X[ok],y[ok],tr[ok],mth[ok]
    cut=int(np.percentile(mth,70))
    trn,tst=mth<=cut,mth>cut
    print(f"\n=== LEARNED SPARSE SELECTION  (L1, time-split)")
    print(f"  rows {X.shape[0]}  features {X.shape[1]}  |  train {trn.sum()} (month<= {mo[cut]})  test {tst.sum()}")
    # residualise the target on the price-history baseline FIRST, so the model is
    # forced to find something trailing correlation does not already have
    A=np.column_stack([np.ones(trn.sum()),tr[trn]])
    b,*_=np.linalg.lstsq(A,y[trn],rcond=None)
    yr_tr=y[trn]-A@b
    A2=np.column_stack([np.ones(tst.sum()),tr[tst]])
    yr_te=y[tst]-A2@b
    if ARGS.stability:
        # fit each half of TRAIN separately; keep only terms with a consistent sign
        mm=mth[trn]; half=np.median(mm)
        h1,h2=mm<=half,mm>half
        m1=LassoCV(cv=3,max_iter=4000,random_state=0).fit(X[trn][h1],yr_tr[h1])
        m2=LassoCV(cv=3,max_iter=4000,random_state=0).fit(X[trn][h2],yr_tr[h2])
        stable=(np.sign(m1.coef_)==np.sign(m2.coef_))&(m1.coef_!=0)&(m2.coef_!=0)
        print(f"  [stability] {int(stable.sum())} terms sign-consistent across both train halves")
        if stable.sum()>0:
            X=X*stable[None,:]
        else:
            print("  [stability] no term is sign-consistent -- nothing to learn from")
    mdl=LassoCV(cv=4,max_iter=5000,random_state=0).fit(X[trn],yr_tr)
    nz=np.flatnonzero(mdl.coef_)
    pred=mdl.predict(X[tst])
    oos=float(np.corrcoef(pred,yr_te)[0,1]) if pred.std()>0 else float("nan")
    ins=float(np.corrcoef(mdl.predict(X[trn]),yr_tr)[0,1]) if mdl.predict(X[trn]).std()>0 else float("nan")
    print(f"  alpha {mdl.alpha_:.5f}   non-zero terms {len(nz)} of {X.shape[1]}")
    print(f"  IC vs residualised target — in-sample {ins:+.3f}   OUT-OF-SAMPLE {oos:+.3f}")
    inv={v:k for k,v in _vocab.items()}
    if len(nz):
        order=np.argsort(-np.abs(mdl.coef_[nz]))[:25]
        print("\n  terms the model SELECTED (weight, term):")
        for i in order:
            j=nz[i]; print(f"    {mdl.coef_[j]:+8.4f}  {inv.get(j,'?')}")
    else:
        print("\n  L1 selected NO terms — nothing in the shared vocabulary predicts the")
        print("  residual once trailing correlation is removed.")
    import sys as _sys; _sys.exit(0)

if ARGS.examples:
    print(f"\n{'='*96}\nWORKED EXAMPLES — what the method actually surfaces\n{'='*96}")
    for ex in _EXROWS[:ARGS.examples]:
        print(f"\n\u25b6 {ex['month']}  ·  event: {ex['key']}")
        print(f"   directly affected (the basket): {', '.join(ex['basket'][:10])}")
        if ex['quote']: print(f"   from the event text: \"{ex['quote'][:150]}\"")
        def _blk(title,rows):
            print(f"\n   {title}")
            print(f"   {'firm':7} {'score':>6} {'corr before':>12} {'corr after':>11} {'change':>8}   shared terms driving the match")
            for sc,fs,tr,fw in rows:
                d=fw-tr
                print(f"   {fs:7} {sc:>6.3f} {tr:>12.2f} {fw:>11.2f} {d:>+8.2f}   "
                      f"{', '.join(_shared_terms(fs,(ex['month'],ex['key']),ex['month'])) or '-'}")
            return float(np.mean([r[3]-r[2] for r in rows]))
        hi=_blk("RANKED MOST EXPOSED (never mentioned in this event's coverage):",ex['top'])
        lo=_blk("RANKED LEAST EXPOSED (the control):",ex['bot'])
        print(f"\n   -> most-exposed correlation moved {hi:+.2f} on average; least-exposed {lo:+.2f}"
              f"   (gap {hi-lo:+.2f})")
    import sys as _sys; _sys.exit(0)

def report(name):
    a=np.array([v for v in per[name] if v is not None]); 
    if len(a)<5: print(f"  {name:22} n<5"); return
    m=a.mean(); t=m/(a.std(ddof=1)/len(a)**0.5)
    rng=np.random.RandomState(0); bs=[rng.choice(a,len(a)).mean() for _ in range(2000)]; lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"  {name:22} mean IC {m:+.3f}  t {t:+.1f}  95%CI [{lo:+.3f},{hi:+.3f}]  ({len(a)} events)")
print(f"\n=== [profile={ARGS.profile} pool={ARGS.pool} sim={ARGS.sim} hl={ARGS.half_life}] T3: STRICTLY PRE-EVENT profiles (leakage-free) — event-level IC, mean/t/bootstrap across events; T2 broadened non-mention) ===")
print("  --- CONTEMPORANEOUS (the F22/F24 exposure claim) ---")
for k in ["trail->contemp","emb->contemp","emb|trail->contemp"]: report(k)
if ARGS.sector_control:
    for k in ["sector->contemp","emb|trail+sector->contemp"]: report(k)
print("  --- FORWARD (R3: does text predict NEXT-window co-movement, incremental to trailing?) ---")
for k in ["trail->forward","emb->forward","emb|trail->forward"]: report(k)
if ARGS.nq_control:
    print("  --- COVERAGE CONTROL (log # pre-event quotes) ---")
    for k in ["nquotes->forward","emb|trail+nq->contemp","emb|trail+nq->forward"]: report(k)
if ARGS.chain:
    print("  --- CAUSAL CHAIN (2-hop bridge, hubs excluded) ---")
    for k in ["bridge->forward","bridge|trail->forward",
              "emb|trail+bridge->forward","bridge|trail+emb->forward"]: report(k)
if ARGS.sector_control:
    for k in ["sector->forward","emb|trail+sector->forward"]: report(k)
print("\n  trail=pre-event trailing corr (price-history baseline); emb=embedding text exposure; emb|trail=text INCREMENT over price history")
