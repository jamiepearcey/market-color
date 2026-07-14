# Q3 — What are the spillovers of Russia's fuel crisis onto Central Asian and former-Soviet economies? (FULL methodology)

Run date: 2026-07-12. Corpus: raw article chunks. 12 search calls (1 causal_find, 1 direction, 1 corroborate, 9 retrieve incl. --diverse + disconfirming).

## Objectives & decision context
Decision: EM/frontier-market macro & FX risk monitoring for Central Asia and the former-Soviet space; regional inflation/fuel-security watch.
1. Establish the source shock: what is Russia's fuel crisis and why does it export abroad (baseline + causal direction).
2. Map the transmission channels into Central Asia (supply dependence, prices, government responses) and verify each per country.
3. Extend to the broader former-Soviet space (Belarus, Ukraine, Moldova/Transnistria) — is the spillover regional or Central-Asia-specific?
4. Separate the Russia driver from confounds (domestic policy taxes; Iran/Middle-East supply shock hitting the same prices).
5. Surface freshest developments and honest gaps (remittances/GDP channel).

## Executive answer
Russia's fuel crisis — Ukrainian drone strikes knocking out **over 30% of refining capacity**, a **~20% domestic gasoline shortfall**, gasoline-purchase limits and a **diesel export ban** — is spilling over into Central Asia through the **refined-product supply channel**: Tajikistan and Kyrgyzstan depend on Russia for "virtually all" of their petroleum products, and even energy-rich Uzbekistan relies on Russian refined fuel because regional refining capacity is limited (cbb85aa6). Fuel prices are rising across the region and governments are scrambling for alternative sources (e2234b9d). The country pattern is differentiated: **Kyrgyzstan hunts for new suppliers, Uzbek prices climb, and Kazakhstan guards its own market** (ef0d433b) — Kazakhstan restricting outflows because its high-octane gasoline is now ~40% cheaper than Russia's, creating cross-border arbitrage/smuggling pressure it is policing (7c8a2219). The chain is corroborated (7 independent sources) and the causal arrow (Russia export curbs → Central Asian shortages/price rises) is directionally confirmed. TWO confounds are named, not credited to the thesis: (a) **domestic policy** — Tajikistan added a 30 euro/tonne environmental fee on imported gasoline/diesel, raising prices independently (e2234b9d); (b) the concurrent **Iran/Middle-East supply shock** tightening global product markets. Broader former-Soviet spillover beyond Central Asia (Belarus, Ukraine, Moldova) is NOT evidenced in this window as a fuel-price channel — a searched near-negative. The remittances/GDP macro channel was searched and NOT found.

## Findings ledger

### F1. Source shock: Russia's refining capacity and export capability are severely degraded [corroborated / high]
- Ukrainian strikes knocked out **over 30% of Russia's oil refining capacity**; Russia faces a **~20% shortfall in domestic gasoline production** and plans to import up to **400,000 tonnes/month** of fuel from neighbours (3bfbb159, OilPrice). Found by: cosine (numeric figures verified in passage).
- Russia **banned diesel exports** to ensure domestic supply after drone strikes; will start importing fuel in July (3fefcbbf Al-Monitor; 451ddb48 SCMP; 393e4a68 Guardian). Found by: causal_find + corroborate.
- Putin publicly admitted fuel shortages / queues at petrol stations (8ab8ac36 BBC; e8665b86 Rigzone). Found by: causal_find.
- Russian jet-fuel exports fell to ~13,000 bpd from ~30,000 bpd in 2025 (d57fd807, OilPrice). Found by: causal_find (candidate #1).

### F2. Core spillover channel: Central Asia's dependence on Russian refined product → shortages & rising prices [corroborated / high]
- Tajikistan and Kyrgyzstan depend on Russia for "virtually all" of their petroleum-product supplies; even energy-rich Uzbekistan relies heavily on Russian refined fuel as regional refining capacity is limited — so disruptions at Russian refineries hit them directly (cbb85aa6, OilPrice). Found by: **receptor (causal_find candidate #4)** + cosine.
- "Fuel prices rising across the region and governments scrambling to find alternative sources"; a Bishkek taxi driver reports the immediate cost impact (e2234b9d, OilPrice/RFE-RL). Found by: cosine.
- Corroborate run: 7 confirming independent sources (OilPrice, SCMP, Nikkei, Eurasianet, Guardian, IntelliNews/BOFIT, ISW), 12 passages. Found by: --corroborate.

### F3. Differentiated country responses [corroborated / high]
- Nikkei headline/lede: "Kyrgyzstan hunts for suppliers, Uzbek prices climb, Kazakhstan guards its market" (ef0d433b + per-country chunks cb7220d3 Kyrgyzstan, 6e43a001 Uzbekistan, 594fd42f Kazakhstan). Found by: cosine (targeted country query).
- Kazakhstan tightens control over petroleum products / restricts outflows; high-octane gas in Russia is now ~40% higher than in Kazakhstan and climbing, prompting a crackdown on cars with extra fuel tanks (arbitrage/smuggling); regulator to work with importers to cut prices (7c8a2219, Eurasianet). Found by: cosine (Kazakhstan facet). Numeric "40%" verified in passage.

### F4. Confound A — domestic policy also raising Central Asian fuel prices [supported / high]
- Tajikistan introduced a **30 euro-per-tonne environmental fee** on imported gasoline and diesel, adding to price pressure independently of Russia; some officials cite fuel-price increases "without explicitly pointing to Russia as the cause" (e2234b9d, OilPrice). Found by: confound search. Attributed as a rival/additive driver — the report does not credit the full price rise to Russia alone.

### F5. Confound B — concurrent Iran/Middle-East supply shock tightening product markets [supported / medium]
- Asia's crude imports remained well below pre-war levels amid constrained Middle Eastern flows and high prices for alternative supply; the Iran war is a simultaneous global-product shock overlapping the window (9c7c3ba4, OilPrice). Found by: confound search. Named as a co-driver of higher regional fuel costs; not disentangled quantitatively in-corpus.

### F6. Second-order Russia macro pressure (spillover to Russia's own economy → knock-on) [supported / medium]
- BOFIT: Russia faces increasing challenges balancing economic policy; government uses compensation payments to entice refiners to sell domestically rather than export (64dc8713, IntelliNews). ISW: strike campaigns will likely continue to hurt Russia's economy (30cf4b5c). Found by: corroborate. Relevant because a weaker Russian economy is the classic transmission belt to Central Asia (remittances/trade) — but that macro link itself was not evidenced (see F7).

### F7. Remittances / migrant-labour / GDP channel — NOT found this window [no_evidence_found / medium confidence in the negative]
- Direct search ("Russia refinery outage → remittances/migrant workers/GDP Tajikistan Kyrgyzstan") returned only the fuel-price documents; no passage links the fuel crisis to remittance flows or growth. Searched; genuinely absent in-corpus (a decision-useful gap, medium confidence because coverage of Central-Asian macro data is thin here).

### F8. Broader former-Soviet spillover (Belarus, Ukraine, Moldova) — NOT evidenced as a fuel channel [no_evidence_found / medium]
- --diverse surfaced Transnistria/Moldova (2ee89da4) and Uzbekistan–Belarus trade (40afa080), but none tie to the fuel crisis; these are unrelated trade stories. The evidenced spillover is Central-Asia-specific in this window.

### Disconfirming search (mandatory)
Query: "Central Asia fuel supply stable / adequate reserves / no shortage / limited impact." Result: did NOT overturn the thesis. The one mitigating datum — Kyrgyzstan's oil-traders' association head says "supplies remain stable, enough reserves" (e2234b9d) — is a single official's reassurance that coexists with documented price rises and Kyrgyzstan actively hunting for new suppliers; it caps the severity for Kyrgyzstan specifically but does not contradict region-wide price spillover. Net: spillover thesis survives disconfirmation.

### Causal-direction check (mandatory)
direction("Russia bans fuel exports after Ukrainian drone strikes on refineries" → "Central Asian nations face fuel shortages and rising gasoline prices"): a→b 0.413 vs b→a 0.169, margin 0.244 → arrow agrees strongly with the asserted Russia→Central-Asia chain (a lead, consistent with the documentary evidence).

## Discarded candidates (and why)
- 0f45426f (Zero Hedge, Russia warns NATO): geopolitical rhetoric, no economic transmission — discarded.
- 2c28ad95 (Economist, Russia's economy / EU sanctions): Russia-EU framing, no Central-Asia link — discarded.
- 26cc36a5, afde818c (FXStreet gold/EUR-USD): FX/macro noise, no fuel-spillover content — discarded.
- ec34229c (ECB "back to basics"): 2022 gas cut-off reference, wrong episode — discarded.
- 2ee89da4 (Transnistria–EU trade), 40afa080 (Uzbekistan–Belarus trade): former-Soviet but unrelated to the fuel crisis — discarded from the fuel-channel claim (noted under F8).

## Gaps
- No quantified regional CPI / fuel-price-index figures (only "40% cheaper than Russia" and qualitative "climbing").
- Remittance/GDP transmission unmeasured (F7).
- Iran-vs-Russia contribution to the price rise not disentangled (F5).
- Broader former-Soviet space (Belarus/Ukraine/Moldova) fuel spillover unevidenced (F8).
