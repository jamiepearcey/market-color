# News sourcing research — full-text access, premium wires, vendors, OSS

Compiled 2026-06-30 (multi-agent web research). The question: **how do we build a full-text news corpus, and can we legally get Bloomberg (and other premium wires) full article text?**

## TL;DR verdicts

- **Open-web full text** → build it ourselves (this repo): sitemap+RSS discovery → **trafilatura** extraction. The purpose-built OSS equivalent is **news-please**.
- **Bloomberg full text** → only via **Bloomberg's own enterprise machine-readable news** (Textual News over B-PIPE / Data License), **~$100k+/yr**. No aggregator carries it; scraping is ToS-prohibited + anti-bot-walled. There is **no cheap or free legitimate path.**
- **WSJ / Reuters / FT / Dow Jones bodies in one legal feed** → **Dow Jones Factiva DNA** (Snapshots/Streams, per-source license-gated). **Does not carry Bloomberg.**
- **Reuters full text** → **LSEG Machine Readable News (MRN)** feed, or Eikon `get_news_story` (~$22k/yr seat). Archive to 1996.
- **Open-web breadth without maintaining 60 scrapers** → a licensed content API with full `body`: **Event Registry/NewsAPI.ai** (~$90/mo, archive to 2014) or **Webz.io** (archive to 2008). These carry public-web full text, **not** licensed premium wires.
- **GDELT is metadata/URLs/tone only — NOT full text** (copyright). This is precisely why `news-narrative-explainer` v3 produced ungrounded narratives.

## Access-tier framing (how we actually get each source)

| Tier | Sources | How | In this repo |
|---|---|---|---|
| **open** | BBC, CNBC, Guardian, OilPrice, ForexLive, Kitco, Al Jazeera, most verticals | crawl sitemaps+RSS → extract body | ✅ crawler gets full body |
| **headline** | Bloomberg / WSJ / FT / Economist / NYT | RSS gives headline+snippet; body paywalled | headline only unless licensed |
| **google_news** | Reuters, AP (no usable RSS) | Google News RSS `site:` + link decode | AP/Kitco/Metal.com crawl; **Reuters 401** (blocked) |
| **licensed** | Bloomberg/Reuters/WSJ full text | paid enterprise feed | not wired |

Google News RSS is **discovery only** (headline stubs, ~100-item cap, obfuscated `CBMi…`/`AU_yqL…` links needing decode) — not a content mirror. The `googlenewsdecoder` lib resolves the links; whether the body is then fetchable depends on the publisher (AP/Kitco/Metal.com/Fastmarkets = yes; Reuters/WSJ = 401/paywall).

## Bloomberg — the definitive answer

- **Public RSS** (feeds.bloomberg.com): headline-only, no bodies (RSS discontinued ~2006; "Bloomberg RSS feeds" online are third-party scrapers).
- **Terminal + BLPAPI**: news is display-only / rate-limited; ToS forbids scraping, redistribution, bulk extraction. Terminal ~$32k/yr/seat but does **not** grant a corpus feed.
- **The only legitimate full-text path**: **Textual News** ("headlines and **full story bodies where available**") via **B-PIPE** (real-time) and **Data License / Enterprise Access Point** (EOD bulk, archive to **1992**). Negotiated enterprise contract, **~$100k+/yr**, redistribution-restricted.
- **Caveat**: full body reliable for **Bloomberg's own News + press wires + web/social**; carried third-party wires (Dow Jones/WSJ) are often Terminal-display-only off-Terminal.
- **Aggregators do NOT carry Bloomberg News** — Factiva, RavenPack, NewsData, GNews, Webz, Event Registry all lack it. Bloomberg competes with Dow Jones/Reuters and doesn't syndicate its wire.

## Financial-grade wire/analytics vendors

| Vendor | Full body text? | Premium wires carried | Access | Pricing | Archive |
|---|---|---|---|---|---|
| **Dow Jones Factiva / DNA** | **Yes** (Avro `body`, per-source license-gated; Explain job = metadata only) | WSJ (to 1979), Barron's, NYT, WaPo, FT, Reuters, AFP, DJ Newswires. **No Bloomberg** | DNA Snapshots (bulk/GCS) / Streams (real-time) / REST | ~$18k → $500k+/yr; 8,000+ GenAI-licensed subset | 1980s (archive to 1950s) |
| **LSEG / Refinitiv (Reuters)** | **Yes** — MRN feed; Eikon `get_news_story` (HTML body) | Reuters + third-party newswires | MRN streaming/bulk; Eikon per-seat | Eikon ~$22k/yr seat; MRN higher | 1996 |
| **Reuters Connect** | **Yes** (republication-grade) | Reuters + ~15 agencies (PA, EFE, DPA, AAP…); **not AP/AFP** | REST/OAuth marketplace | ~$10–50k small; 6-figure financial | — |
| **RavenPack / Bigdata.com** | **No** — analytics + snippets/chunks, not full bodies | DJ, WSJ, Barron's, MarketWatch, FT, Benzinga, MT Newswires. **Not Reuters/Bloomberg** | REST / Snowflake / SFTP / MCP | ~$250k+ ACV enterprise | 2000 |
| **AlphaSense** | In-platform yes; **export entitlement-gated per-doc, not a corpus-out feed** | Broker research, Tegus transcripts, DJ/WSJ/Barron's/FT news | UI + Agent API (cited answers) + inbound Ingestion API | ~$10–20k/seat; ~$40k+ w/ Tegus; ~$125k/yr enterprise | multi-year |

## Licensed content APIs (open-web full text; NOT premium wires)

| API | Full body? | Sources | Archive | Pricing | Premium wires |
|---|---|---|---|---|---|
| **Event Registry / NewsAPI.ai** | Yes (`body`, `articleBodyLen=-1`) | ~150k | 2014 | from **$90/mo** (5K); higher gated | No (public web) |
| **Webz.io** (ex-Webhose) | Yes (`text`) | "millions" (300k+ news) | **2008** | custom/quote; free Lite | No (public web; licensed separate) |
| **NewsData.io** | Yes (`content`, paid only) | 84–97k | 6mo→10yr | $199 / $349 / $1,300/mo | No (public web) |
| **GNews** | Yes (paid; free truncated ~250 char) | 80k | 2020 | €50 / €100 / €250/mo | No (aggregator) |
| **Newscatcher** | Yes (`content`) | ~90–140k | 2019 | News API quote (~$10k/mo ent.; ~$399/mo historic) | No (public web) |
| **Diffbot** | Yes (`text` from URLs you supply / 10B-entity crawl) | your URLs / crawl | crawl-timing | $299 / $899/mo | No — extraction, not licensing; no paywall bypass |
| **Aylien / Quantexa News API** | Yes (`body`) | ~80k | 2012 | enterprise/contract only (post-acquisition) | Not confirmed |

## Open-source extraction/crawling stacks

| Tool | Role | Extraction F1 | Notes |
|---|---|---|---|
| **trafilatura** | Extractor (+ light sitemap/feed discovery) | **0.91–0.96** (benchmark leader) | What we use. Not an autonomous crawler. |
| **news-please** | Full pipeline (crawl + extract), native RSS/sitemap/CC-NEWS | ~0.81 | Purpose-built for news corpora; Scrapy + newspaper4k/readability. Closest to our crawler. |
| **newspaper4k** | Full pipeline | ~0.95 | Maintained successor to abandoned newspaper3k. |
| **Scrapy** | Crawl framework | — | Bring your own extractor. |
| **Common Crawl CC-NEWS** | Ready-made WARC archive (2016→) | — | **No paywalled/premium-wire content**; pair with trafilatura/fastwarc. |
| **GDELT** | Metadata / URLs / tone at planetary scale | — | **NOT full text** (copyright). Full-text search returns URLs+snippets. |

## Recommendation (what to pay for vs build)

- **Now (free):** our crawler for open-web full text (trafilatura). Covers most usable desk color.
- **If breadth without scraper maintenance is wanted:** add **Event Registry** (~$90/mo, full text, archive to 2014) as a second corpus source.
- **If premium-wire full text is required:** **Factiva DNA** is the best single legal source (WSJ/Reuters/FT/DJ bodies) — but **Bloomberg is only Bloomberg**, at ~$100k+/yr. Park Bloomberg unless a desk mandates it.
