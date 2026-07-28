# The primitive pipeline — chunk, mention, resolve, classify, emit

**Status:** proposed + prototyped · 2026-07-26
**Motivation:** F39 (attribution is doc-union) and F41 (95.5% of takeover news
never reaches a priceable name; 66% of the misses produce no event row at all).

## The problem with the current shape

```
document -> LLM extraction -> event rows + causal_event_edge -> cells
                                              ^
                              a cell only exists if a CAUSAL edge exists
```

Requiring a causal relation before a name gets a cell is the bottleneck. "X agrees
to buy Y for $2bn" is a complete, dateable, tradeable event that states no causal
claim, so the extractor often emits nothing and the document vanishes. Measured on
574 takeover headlines: **48.1% produced no causal edge, and two thirds of those
produced no event row either.**

Causality is a hard extraction task with poor recall. *"Which listed company is
this document about, and what kind of event is it?"* is a much easier task with
far better recall. The primitive pipeline inverts the priority accordingly.

## The process

```
1 CHUNK      document -> chunks (exists: lake/chunk.jsonl, text + seq + epoch)
2 MENTION    chunk text -> candidate entities, by alias match      [recall-first]
3 RESOLVE    entity -> VERIFIED ticker only (never `suggested`)    [precision gate]
4 PLACE      ticker -> market/venue, GICS sector, priceability     [the factors]
5 CLASSIFY   headline -> event class, controlled vocab             [taxonomy.py]
6 ROLE       is this entity the SUBJECT or merely MENTIONED?       [replaces doc-union]
7 EMIT       (ticker, date, class, role, quote, chunk_id)
8 FUNNEL     measure reach at every stage as a first-class output
```

### Why each stage is shaped this way

**2 MENTION is deliberately high-recall and low-precision.** Alias matching over
chunk text will over-generate; that is fine, because stage 6 (role) and stage 3
(verified resolution) do the filtering. The current pipeline fails the opposite
way — high precision on causal claims, catastrophic recall.

**3 RESOLVE never uses a suggested ticker as a merge key.** Existing doctrine, and
the india2021 `nifty50_resolved.json` hallucinations are why. Only tickers already
verified against price history + issuer-name match are eligible.

**5 CLASSIFY reads the HEADLINE, not an LLM-emitted `event_type` string.** This is
the key primitive move. The current chain is
`text -> LLM free-text event_type -> taxonomy -> class`, which loses ~13% of
events to the extractor emitting the literal string `"other"` (F35). Going
`headline -> taxonomy patterns -> class` removes a lossy hop. Headline rather than
full chunk because it is the topic sentence — much higher precision per unit text.

**6 ROLE is what replaces doc-union.** F39's bug was every mentioned entity
inheriting every class in the document. The fix is not only `issuer_entity` (32.5%
populated); it is asking, per entity, whether it is the subject. The primitive
proxy: **named in the headline = subject; body-only = mentioned.** Crude, cheap,
and exactly the distinction that was missing.

**8 FUNNEL is an output, not a diagnostic.** F41's reach decomposition was found a
year late by hand. Every run of this pipeline emits it.

## The LLM boundary

Stages 1-4 and 8 are fully deterministic. Only 5 and 6 have any judgement in them,
and both have a deterministic baseline that runs today at zero cost. That ordering
is deliberate: **establish the dumb baseline first, then measure what a model adds**
— the same doctrine as the covariance-baseline harness, where the bar mattered more
than the model.

When an LLM is used for 5/6 the contract is narrow and gated:

- input: headline + the matched entity + the sentence containing it
- output: `event_class` from the controlled vocab, `role` from
  {subject, counterparty, mentioned}, and a **verbatim quote span**
- rejected if: the class is off-vocab, the quote is not literally present in the
  chunk, or the entity is not the verified resolution

The model is a **transcriber and router, never a namer**. It may not invent a class
and may not resolve a ticker — the same boundary the EM calendar work settled on.

## What success looks like

The prototype is judged against the existing graph on the SAME corpus:

| metric | current graph | target |
|---|---|---|
| takeover docs reaching a priceable name | **4.7%** | materially higher |
| cells emitted per document | — | higher, with role separating them |
| measured lift on `subject` cells | 1.21x pooled `m_and_a` | at or above 1.80x |

If a purely lexical pipeline beats 4.7% reach, that is a finding about the
extraction architecture, not about the corpus.

## Known limitations, stated upfront

- Headline-only classification will miss events described only in the body.
- The subject/mentioned proxy will mislabel headlines that lead with a
  counterparty ("Astellas Extends Offer for OSI" makes Astellas the subject and
  OSI merely mentioned, when OSI is the name that gaps).
- Alias matching inherits every bad alias in the table (`"Inco's Net Soars"` is a
  registered alias), so aliases need length and shape filters.
- Nothing here recovers a genuinely delisted name; that leg still needs
  delisted-inclusive price data.

---

## Measured result (2026-07-26, `scripts/primitive_pipeline.py`, zero LLM)

### Head-to-head on the same 573 takeover documents

| | current graph | primitive |
|---|---|---|
| reaches a priceable name | **4.7%** | **56.4%** |
| >=1 verified entity mentioned | — | 95.3% |
| >=1 entity named in the HEADLINE | — | 41.2% |

**A 12.5x recall improvement from alias matching and headline classification, with
no model in the loop.** That settles the question the funnel was built to answer:
the 95.5% loss was an artefact of requiring a causal edge, not a property of the
corpus.

### Role separation works — and it is F39's bug, quantified

| role | n | lift | \|sigma\|>2 |
|---|---|---|---|
| **subject** (named in headline) | 77 | **1.36x** | 10% |
| mentioned (body only) | 679 | **1.05x** | 6% |

`m_and_a` splits **1.39x subject (n=45) vs 1.05x mentioned (n=425)**. Mentioned
sits at the control, which is exactly what doc-union was mixing into every bucket:
**90% of M&A cells are bystanders, and the pooled 1.21x is the average of a real
population and a null one.**

### Generalisation check (6,000 documents, unfiltered)

Reach 48.1% priceable, 96.1% with a verified entity mentioned. Role separation is
real but narrower off the takeover slice (subject 1.19x vs mentioned 1.08x) —
takeover documents name unusually many uninvolved companies, so that probe
flatters the split.

Convergent validation worth noting: **`earnings` subject measures 2.03x here vs
1.96x (doc-union) / 2.17x (issuer) in F36/F39** — three different attribution
paths agreeing on the strongest class is evidence the primitive classifier is not
producing noise.

### Where the LLM belongs, now measured rather than assumed

Two gaps survive the deterministic baseline, and they are precisely the two stages
flagged as judgement calls:

1. **58.7% of headlines yield no class** (34.2% on the takeover slice). Stage 5 is
   the biggest single leak remaining.
2. **Role precision.** The headline-subject proxy gives 1.38x where F41's genuinely
   priceable takeover events gave 1.80x. "Astellas Extends Offer for OSI" marks
   Astellas subject and OSI mentioned — but OSI is the name that gaps.

So the model is worth paying for at **stage 5 (classify) and stage 6 (role)**, and
demonstrably NOT needed for mention, resolution, placement or funnel. That is the
baseline-first doctrine paying off: the cheap pipeline captured the recall, and
what remains is a narrow, well-specified brief.
