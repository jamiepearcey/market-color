# Q4 — How is the Strait of Hormuz disruption reshaping how Asian emerging economies source their energy? (FULL methodology)

Run date: 2026-07-12. Corpus: raw article chunks. 13 search calls (1 causal_find, 1 direction, 1 corroborate, 10 retrieve incl. --diverse + --novel + disconfirming).

## Objectives & decision context
Decision: Asian-EM energy-security, oil-import-cost, FX and refiner positioning; supply-chain-reroute monitoring.
1. Establish the shock: Hormuz disruption during the Iran war (magnitude, closure/traffic).
2. Map how Asian EMs are re-sourcing: geographic diversification (Russia, Iraq/Gulf-at-discount, West Africa, Venezuela), strategic reserves, refining, biofuels/energy-transition.
3. Distinguish country responses (India, Indonesia, China, Japan-as-reference).
4. Separate genuine structural re-sourcing from cyclical/opportunistic behaviour (confound: Gulf discounting pulling buyers BACK; ceasefire/partial reopening).
5. Freshest developments + honest gaps.

## Executive answer
The Strait of Hormuz — which handles ~30% of global seaborne oil trade — suffered a near-total closure during the Iran war and remains partially reopened with traffic below pre-conflict levels and elevated war-insurance premiums (bba530b3, eafef616). Asian emerging economies are responding along FOUR reinforcing tracks, all evidenced:
(1) **Geographic diversification of crude sourcing away from Hormuz-transit Gulf barrels** — Indonesia took its first Russian crude cargo explicitly because Hormuz forced it to look for alternative supply (19a48d85); India stresses it has "already diversified," with West Asia now only 20-30% of imports, and leaned on Venezuela and other non-Hormuz sources (6370e0db, ca147059); Asian refiners are scrambling for alternative cargoes from West Africa (c69a6841).
(2) **Strategic-reserve build-out** — India/ONGC is adding 1.75mn t (12.8mn bbl) of SPR capacity in Mangalore, expressly to cushion future shocks after the Iran conflict (03db3380, d345a681, 4608945d, cf7d955e — four sources).
(3) **Domestic refining & energy-sovereignty policy** — India expanding refining (Modi); Indonesia reframing energy transition as a geopolitical-shock hedge ("energy sovereignty"); Japan's new August energy plan / "Power Asia" framework emphasises oil-security diversification (b66ed1a5, 5cf14c0c, 6670e211); Asia reviving biofuels to dodge Middle-East shortages (8b0c1bb7).
(4) **A supply-side reshaping that partly REVERSES the diversification pull**: Gulf producers relaunched a price war for Asian market share, with Saudi Arabia's OSP cut the sharpest in ~two decades (up to -$11/bbl for August), and China promptly bought 26mn barrels of Gulf crude on the discount (681e7cad, dbd69df2, f3f46413, b96f1bd1). So the net picture is NOT a clean exit from the Gulf: Asian buyers are simultaneously diversifying suppliers/routes AND opportunistically buying discounted Gulf barrels when freight/OSP economics favour it. Nigeria's Dangote refinery emerges as a structural winner as the conflict reshapes the refined-product trade (df7b02e0, 4d2e2453).
The core diversification chain is corroborated (4+ independent sources). The causal-direction probe was inconclusive (near-symmetric, margin 0.03) and is not used to assert direction; the documentary evidence carries it. India is flagged as the most import-exposed Asian EM (38d54812).

## Findings ledger

### F1. The shock: Hormuz near-total closure, partial reopening, traffic below pre-war, insurance spike [corroborated / high]
- Hormuz handled nearly 30% of global seaborne trade; disruption exposed import fragility (5cf14c0c, Jakarta Post). Numeric "~30%" verified.
- "Near-total closure triggered by the US-Iran war"; strait "partially reopened... traffic recovering but below pre-conflict levels" (bba530b3, Rigzone). Renewed attacks → tankers turn back / U-turns; shipping lines halt bookings (55bc82f2, 6370e0db, 309eaa2f). War-insurance costs risen sharply, Hormuz traffic constrained despite ceasefire (eafef616, FXStreet/Rabobank). Found by: cosine + disconfirming.

### F2. Track 1 — geographic diversification of crude sourcing [corroborated / high]
- Indonesia received its first Russian crude under an April deal; "the Iran war and Strait of Hormuz disruption forced the biggest economy in Southeast Asia to look for alternative supply, including from Russia" — an explicit strategy to diversify the import basket (19a48d85, OilPrice). Found by: --diverse + cosine.
- India: refiners "not unduly worried... India has already diversified energy imports, with West Asia supplying just 20-30% of imports" (6370e0db, Livemint). India turned to Venezuela (non-Hormuz) and hit a record ~5mn bpd of imports rebuilding stocks (ca147059, OilPrice). Numeric "20-30%" verified. Found by: cosine.
- Asian refiners "scrambling to secure alternative cargoes from West Africa" as ceasefire declared over (c69a6841, OilPrice). Found by: disconfirming search.
- Corroborate run on the central diversification claim: 4 confirming independent sources (Jakarta Post, OilPrice, Livemint) + 16 topical-only. Found by: --corroborate.

### F3. Track 2 — strategic-reserve build-out [corroborated / high]
- India's ONGC to add 1.75mn t (12.83mn bbl) of SPR capacity in Mangalore to cushion future supply shocks; may seek commercial-use rights (03db3380 OilPrice; d345a681 Argus; a55cfed6 OilPrice). Found by: cosine. Numeric verified across sources.
- India expanding reserves & strengthening supply partnerships for protection against future price shocks "following the Iran conflict" (cf7d955e IntelliNews; 4608945d Rigzone — the latter explicitly ties it to "near-total closure of the Strait of Hormuz... triggered an energy crisis"). Found by: cosine. Four independent sources → corroborated.

### F4. Track 3 — domestic refining, energy-sovereignty policy, biofuels [corroborated / medium-high]
- Modi: India to keep expanding refining as the West shuts capacity; new plant broadens ability to process wider crude grades and strengthens domestic fuel supply (b66ed1a5, ForexLive). Found by: cosine.
- Indonesia reframes energy transition from a climate imperative to "a strategic tool to reduce import dependence and protect the economy from geopolitical shocks" — "energy sovereignty" (5cf14c0c, Jakarta Post). Found by: --diverse.
- Japan's new August energy plan may emphasise oil security and diversification via the "Power Asia" framework for stable Asia-wide oil supply (6670e211, Argus). Found by: causal_find (candidate #6) + --novel. [Japan is a reference DM, not an EM — flagged.]
- "Asia Bets on Biofuels to Dodge Middle East Oil Shortages" — biofuel interest reviving on fossil-fuel price volatility (8b0c1bb7, OilPrice). Found by: --diverse.

### F5. Confound / counter-nuance — Gulf price war pulls Asian buyers BACK toward discounted Gulf crude [corroborated / high]
- Gulf producers relaunched the fight for Asian market share; Saudi Arabia cut Asian OSP the sharpest in two decades (Aramco slashed August Asian-bound OSP by ~$11/bbl, first such move since 2020) (681e7cad, dbd69df2, b96f1bd1). Found by: cosine (Gulf facet).
- China bought 26mn barrels of Gulf crude as the Saudi discount deepened to -$1.5/bbl — a "direct trigger for the surge," vs ~5.5mn/day regular pre-war Gulf buying (f3f46413, ForexLive). Found by: cosine. Numeric verified.
- Implication (named, not credited to the thesis): the "reshaping" is bidirectional. Hormuz risk pushes structural diversification (reserves, non-Gulf suppliers, refining), but discounted Gulf economics simultaneously pull opportunistic Gulf buying back. Reported as a genuine tension, not resolved in favour of a clean exit.

### F6. Product-trade reshaping — West Africa / Dangote as beneficiary [supported / medium]
- Dangote refinery (650,000 bpd) "emerges biggest winner from Iran war" (WSJ via BusinessDay); gasoline/diesel/jet exports accelerating as the conflict reshapes global fuel trade (df7b02e0, 4d2e2453, BusinessDay). Found by: cosine (West Africa facet). Single-outlet (BusinessDay carrying WSJ) → capped at supported; corroborated by the F2 West-Africa cargo-scramble passage as the demand side.

### F7. Vulnerability framing — India the most import-exposed Asian EM [supported / medium]
- "Oil import dependency remains the primary vulnerability for Asian emerging markets, with India the most exposed given thinner inventory buffers relative to China and Japan"; partial Hormuz reopening noted (38d54812, ForexLive/investingLive). Found by: --novel. Explains why India's response (F3 SPR, F4 refining) is the most aggressive.

### Disconfirming search (mandatory)
Query: "Asia oil buyers return to Gulf / Hormuz reopened / traffic normalizes / not diversifying." Result: surfaced the real counter-current (F5: Gulf discounts pulling China back; partial reopening) — which I incorporated as a named confound rather than suppressing. It does NOT overturn the diversification thesis (reserves, Russian/Venezuelan/West-African sourcing, policy shifts are all documented and forward-looking); it qualifies it: the shift is structural on the security/policy axis but partly reversible on the transactional crude-buying axis when Gulf economics are attractive. Thesis survives, qualified.

### Causal-direction check (mandatory)
direction("Strait of Hormuz shipping disruption during the Iran war" → "Asian economies diversify crude imports toward Russia and non-Gulf sources and build strategic reserves"): a→b 0.417 vs b→a 0.447, margin 0.029 → INCONCLUSIVE (near-symmetric, below a usable threshold). The probe does not confirm the arrow here; the causal ordering rests instead on the explicit documentary attributions in F2/F3 (sources state Hormuz "forced"/"triggered" the diversification). Reported honestly as a probe miss, not asserted.

## Discarded candidates (and why)
- 24ff42c3 (S.Korea-Mongolia renewables MOU), 2812e277 (Global Carbon Council MoU): energy-transition MoUs with no Hormuz-sourcing link — discarded.
- f741069e, f1f5ba74 (BusinessDay Nigeria governance/green-transition-banking), 734b455f (Kotak bank growth), 748... : causal_find noise (low cosine), no energy-sourcing content — discarded.
- daef5821 (global nuclear capacity +44% by 2036): long-horizon, not Asian-EM Hormuz re-sourcing — discarded.
- 2e4f3d3d (oil glut weakening Iran's Hormuz leverage), 5e9d6fbb (UAE output post-OPEC): supply/price-macro, not EM sourcing behaviour — noted as context, not asserted as the answer.

## Gaps
- No hard tonnage/percentage shift figures for the Russia/Venezuela/West-Africa reroutes (only India's "20-30% from West Asia" and the SPR volumes are quantified).
- China's structural response beyond opportunistic discount-buying (F5) is thin in-corpus — its SPR refill is implied (5e9d6fbb) but not detailed.
- Smaller Asian EMs (Philippines, Thailand, Vietnam, Bangladesh, Pakistan) sourcing responses not evidenced — a coverage gap, low confidence in any negative.
- Gas/LNG re-sourcing (vs crude) only lightly touched (tanker U-turns, Qatar LNG ties) — not developed.
