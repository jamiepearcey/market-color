//! Corpus row model + the exact 22-column Arrow/Parquet schema consumed by the
//! eventgraph pipeline (matches `../data/news_corpus/dt=*/part-000.parquet`).
//!
//! `Row` is the on-disk shape; `Article` is the freshly-scraped shape (they map
//! 1:1). Kept in one place so the writer, the merge-read, and `ParquetFeed` all
//! agree on column order + nullability.

use anyhow::{Context, Result};
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Int64Array, Int64Builder, ListArray,
    ListBuilder, StringArray, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// One corpus row. Field order == schema column order == Python `Article`.
#[derive(Debug, Clone)]
pub struct Row {
    pub doc_id: String,
    pub source_name: String,
    pub source_scope: String,
    pub source_tier: i64,
    pub source_access: String,
    pub source_method: String,
    pub source_domain: String,
    pub desks: Vec<String>,
    pub discovery: String,
    pub title: String,
    pub body_text: Option<String>,
    pub author: Option<String>,
    pub url: String,
    pub canonical_url: String,
    pub lang: Option<String>,
    pub word_count: i64,
    pub extraction_ok: bool,
    pub published_utc: Option<String>,
    pub published_date: String,
    pub published_is_estimated: bool,
    pub fetched_utc: String,
    pub content_hash: Option<String>,
}

/// The 22-column schema, in the exact order the corpus uses.
pub fn corpus_schema() -> Arc<Schema> {
    let desks = Field::new(
        "desks",
        DataType::List(Arc::new(Field::new("element", DataType::Utf8, true))),
        true,
    );
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("source_name", DataType::Utf8, false),
        Field::new("source_scope", DataType::Utf8, false),
        Field::new("source_tier", DataType::Int64, false),
        Field::new("source_access", DataType::Utf8, false),
        Field::new("source_method", DataType::Utf8, false),
        Field::new("source_domain", DataType::Utf8, false),
        desks,
        Field::new("discovery", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("body_text", DataType::Utf8, true),
        Field::new("author", DataType::Utf8, true),
        Field::new("url", DataType::Utf8, false),
        Field::new("canonical_url", DataType::Utf8, false),
        Field::new("lang", DataType::Utf8, true),
        Field::new("word_count", DataType::Int64, false),
        Field::new("extraction_ok", DataType::Boolean, false),
        Field::new("published_utc", DataType::Utf8, true),
        Field::new("published_date", DataType::Utf8, false),
        Field::new("published_is_estimated", DataType::Boolean, false),
        Field::new("fetched_utc", DataType::Utf8, false),
        Field::new("content_hash", DataType::Utf8, true),
    ]))
}

/// Build one RecordBatch from a slice of rows.
pub fn rows_to_batch(rows: &[Row]) -> Result<RecordBatch> {
    let mut doc_id = StringBuilder::new();
    let mut source_name = StringBuilder::new();
    let mut source_scope = StringBuilder::new();
    let mut source_tier = Int64Builder::new();
    let mut source_access = StringBuilder::new();
    let mut source_method = StringBuilder::new();
    let mut source_domain = StringBuilder::new();
    let mut desks = ListBuilder::new(StringBuilder::new())
        .with_field(Arc::new(Field::new("element", DataType::Utf8, true)));
    let mut discovery = StringBuilder::new();
    let mut title = StringBuilder::new();
    let mut body_text = StringBuilder::new();
    let mut author = StringBuilder::new();
    let mut url = StringBuilder::new();
    let mut canonical_url = StringBuilder::new();
    let mut lang = StringBuilder::new();
    let mut word_count = Int64Builder::new();
    let mut extraction_ok = BooleanBuilder::new();
    let mut published_utc = StringBuilder::new();
    let mut published_date = StringBuilder::new();
    let mut published_is_estimated = BooleanBuilder::new();
    let mut fetched_utc = StringBuilder::new();
    let mut content_hash = StringBuilder::new();

    for r in rows {
        doc_id.append_value(&r.doc_id);
        source_name.append_value(&r.source_name);
        source_scope.append_value(&r.source_scope);
        source_tier.append_value(r.source_tier);
        source_access.append_value(&r.source_access);
        source_method.append_value(&r.source_method);
        source_domain.append_value(&r.source_domain);
        for d in &r.desks {
            desks.values().append_value(d);
        }
        desks.append(true);
        discovery.append_value(&r.discovery);
        title.append_value(&r.title);
        body_text.append_option(r.body_text.as_deref());
        author.append_option(r.author.as_deref());
        url.append_value(&r.url);
        canonical_url.append_value(&r.canonical_url);
        lang.append_option(r.lang.as_deref());
        word_count.append_value(r.word_count);
        extraction_ok.append_value(r.extraction_ok);
        published_utc.append_option(r.published_utc.as_deref());
        published_date.append_value(&r.published_date);
        published_is_estimated.append_value(r.published_is_estimated);
        fetched_utc.append_value(&r.fetched_utc);
        content_hash.append_option(r.content_hash.as_deref());
    }

    let cols: Vec<ArrayRef> = vec![
        Arc::new(doc_id.finish()),
        Arc::new(source_name.finish()),
        Arc::new(source_scope.finish()),
        Arc::new(source_tier.finish()),
        Arc::new(source_access.finish()),
        Arc::new(source_method.finish()),
        Arc::new(source_domain.finish()),
        Arc::new(desks.finish()),
        Arc::new(discovery.finish()),
        Arc::new(title.finish()),
        Arc::new(body_text.finish()),
        Arc::new(author.finish()),
        Arc::new(url.finish()),
        Arc::new(canonical_url.finish()),
        Arc::new(lang.finish()),
        Arc::new(word_count.finish()),
        Arc::new(extraction_ok.finish()),
        Arc::new(published_utc.finish()),
        Arc::new(published_date.finish()),
        Arc::new(published_is_estimated.finish()),
        Arc::new(fetched_utc.finish()),
        Arc::new(content_hash.finish()),
    ];
    RecordBatch::try_new(corpus_schema(), cols).context("build record batch")
}

/// Write rows to a single-part parquet file (overwrites).
pub fn write_parquet(path: &Path, rows: &[Row]) -> Result<()> {
    let batch = rows_to_batch(rows)?;
    let file = File::create(path).with_context(|| format!("create {path:?}"))?;
    let mut w = ArrowWriter::try_new(file, corpus_schema(), None).context("arrow writer")?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

fn str_opt(arr: &ArrayRef, i: usize) -> Option<String> {
    let a = arr.as_any().downcast_ref::<StringArray>()?;
    if a.is_null(i) {
        None
    } else {
        Some(a.value(i).to_string())
    }
}

fn str_req(arr: &ArrayRef, i: usize) -> String {
    str_opt(arr, i).unwrap_or_default()
}

/// Read a parquet part back into rows (used for append-merge + `ParquetFeed`).
pub fn read_parquet(path: &Path) -> Result<Vec<Row>> {
    let file = File::open(path).with_context(|| format!("open {path:?}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).context("parquet reader")?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let n = batch.num_rows();
        let col = |name: &str| batch.column_by_name(name).cloned();
        let tier = col("source_tier")
            .and_then(|a| a.as_any().downcast_ref::<Int64Array>().cloned());
        let wc = col("word_count").and_then(|a| a.as_any().downcast_ref::<Int64Array>().cloned());
        let eok = col("extraction_ok")
            .and_then(|a| a.as_any().downcast_ref::<BooleanArray>().cloned());
        let est = col("published_is_estimated")
            .and_then(|a| a.as_any().downcast_ref::<BooleanArray>().cloned());
        let desks_col = col("desks").and_then(|a| a.as_any().downcast_ref::<ListArray>().cloned());
        for i in 0..n {
            let desks = desks_col
                .as_ref()
                .map(|la| {
                    let v = la.value(i);
                    v.as_any()
                        .downcast_ref::<StringArray>()
                        .map(|sa| (0..sa.len()).map(|j| sa.value(j).to_string()).collect())
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            out.push(Row {
                doc_id: col("doc_id").map(|a| str_req(&a, i)).unwrap_or_default(),
                source_name: col("source_name").map(|a| str_req(&a, i)).unwrap_or_default(),
                source_scope: col("source_scope").map(|a| str_req(&a, i)).unwrap_or_default(),
                source_tier: tier.as_ref().map(|a| a.value(i)).unwrap_or(0),
                source_access: col("source_access").map(|a| str_req(&a, i)).unwrap_or_default(),
                source_method: col("source_method").map(|a| str_req(&a, i)).unwrap_or_default(),
                source_domain: col("source_domain").map(|a| str_req(&a, i)).unwrap_or_default(),
                desks,
                discovery: col("discovery").map(|a| str_req(&a, i)).unwrap_or_default(),
                title: col("title").map(|a| str_req(&a, i)).unwrap_or_default(),
                body_text: col("body_text").and_then(|a| str_opt(&a, i)),
                author: col("author").and_then(|a| str_opt(&a, i)),
                url: col("url").map(|a| str_req(&a, i)).unwrap_or_default(),
                canonical_url: col("canonical_url").map(|a| str_req(&a, i)).unwrap_or_default(),
                lang: col("lang").and_then(|a| str_opt(&a, i)),
                word_count: wc.as_ref().map(|a| a.value(i)).unwrap_or(0),
                extraction_ok: eok.as_ref().map(|a| a.value(i)).unwrap_or(false),
                published_utc: col("published_utc").and_then(|a| str_opt(&a, i)),
                published_date: col("published_date").map(|a| str_req(&a, i)).unwrap_or_default(),
                published_is_estimated: est.as_ref().map(|a| a.value(i)).unwrap_or(false),
                fetched_utc: col("fetched_utc").map(|a| str_req(&a, i)).unwrap_or_default(),
                content_hash: col("content_hash").and_then(|a| str_opt(&a, i)),
            });
        }
    }
    Ok(out)
}
