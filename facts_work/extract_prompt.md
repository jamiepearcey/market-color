You are a financial-news fact extractor. Work ONLY from the file `energy_sample.jsonl` in the current directory. Do NOT use the network.

Each line of `energy_sample.jsonl` is a JSON object: {doc_id, source_name, published_date, url, title, body_text}.

For EACH document, extract the atomic, market-relevant FACTS explicitly supported by its title+body_text. An atomic fact is a single checkable claim about an asset, commodity, company, country, central bank, or market — with direction/magnitude/time where stated, and a causal link where the text states one. Do NOT invent or infer beyond the text. If a document has no market-relevant facts (e.g. a job posting), return an empty facts list — that is correct.

Each fact uses this shape:
{
  "claim": "<one self-contained sentence>",
  "subject": "<primary entity, e.g. 'WTI crude', 'bp', 'India'>",
  "predicate": "<one of: price_move, supply_change, demand_change, production_change, policy_action, deal_or_contract, corporate_action, forecast, sanction, statement, event, other>",
  "object": "<counterparty/target entity or null>",
  "direction": "<up|down|flat|na>",
  "magnitude": "<value/percent as written or null>",
  "time": "<date/period as written, else the doc published_date>",
  "cause": "<the MOST SPECIFIC stated driver, naming the concrete event/actor (e.g. 'Iran fired on the tanker Kiku in the Strait of Hormuz', not 'the conflict') — when the text states both a general and a specific cause, use the specific one; null if none stated>",
  "entities": ["<canonical entity names in this fact>"],
  "confidence": <0.0-1.0>
}

Write `energy_facts.jsonl` in the current directory: ONE JSON object per input doc, in order:
{"doc_id":"<id>","title":"<title>","facts":[ <fact>, ... ]}

Then print one line: docs processed and total facts. Write no other files.
