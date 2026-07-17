//! Postgres tier writer: emits idempotent upsert SQL for the reference + lifecycle
//! tables (entity, entity_alias, proposition, extraction_run). Apply via `psql`.
//! Facts do NOT come here -- they go to DuckLake (see lake.rs).

use crate::model::GraphBatch;
use crate::pipeline::RunMeta;

fn lit(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}
fn opt(o: &Option<String>) -> String {
    o.as_ref().map(|s| lit(s)).unwrap_or_else(|| "NULL".into())
}

pub fn upsert_sql(batch: &GraphBatch, meta: &RunMeta) -> String {
    let mut s = String::new();
    s.push_str("SET search_path = eventgraph, public;\nBEGIN;\n");

    s.push_str(&format!(
        "INSERT INTO extraction_run (run_id,model,prompt_version,schema_version,created_at) \
         VALUES ({},{},{},{},{}) ON CONFLICT (run_id) DO NOTHING;\n",
        lit(&meta.run_id), lit(&meta.model), lit(&meta.prompt_version),
        lit(&meta.schema_version), lit(&meta.created_at)
    ));

    for e in &batch.entities {
        s.push_str(&format!(
            "INSERT INTO entity (entity_id,canonical_name,type,identifier,sector,country) \
             VALUES ({},{},{},{},{},{}) ON CONFLICT (entity_id) DO UPDATE SET \
             identifier=COALESCE(entity.identifier,EXCLUDED.identifier), \
             sector=COALESCE(entity.sector,EXCLUDED.sector), \
             country=COALESCE(entity.country,EXCLUDED.country);\n",
            lit(&e.entity_id), lit(&e.canonical_name), lit(&e.r#type),
            opt(&e.identifier), opt(&e.sector), opt(&e.country)
        ));
    }
    for a in &batch.aliases {
        s.push_str(&format!(
            "INSERT INTO entity_alias (alias,entity_id) VALUES ({},{}) ON CONFLICT DO NOTHING;\n",
            lit(&a.alias), lit(&a.entity_id)
        ));
    }
    for p in &batch.propositions {
        s.push_str(&format!(
            "INSERT INTO proposition (proposition_id,text,subject_entity,resolution_date,\
             resolution_criteria,status,contract_ref,implied_instrument,first_doc_id) \
             VALUES ({},{},{},{},{},{},{},{},{}) ON CONFLICT (proposition_id) DO NOTHING;\n",
            lit(&p.proposition_id), lit(&p.text), opt(&p.subject_entity),
            opt(&p.resolution_date), opt(&p.resolution_criteria), lit(&p.status),
            opt(&p.contract_ref), opt(&p.implied_instrument), lit(&p.doc_id)
        ));
    }
    s.push_str("COMMIT;\n");
    s
}
