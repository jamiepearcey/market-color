use std::fs;
use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use reqwest::header::{ACCEPT, AUTHORIZATION, USER_AGENT};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::controlplane::{ControlStore, Job, PackSource, PackVersion};

#[derive(Debug, Clone, Deserialize)]
struct CommitResponse {
    sha: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestFile {
    pack: PackManifest,
}

#[derive(Debug, Clone, Deserialize)]
struct PackManifest {
    id: String,
    name: String,
    #[serde(default)]
    version: String,
}

pub fn parse_github_repo(repo_url: &str) -> Result<(String, String)> {
    let trimmed = repo_url.trim().trim_end_matches('/');
    if let Some(rest) = trimmed.strip_prefix("git@github.com:") {
        let without_git = rest.strip_suffix(".git").unwrap_or(rest);
        return split_owner_repo(without_git);
    }
    let marker = "github.com/";
    let Some((_, rest)) = trimmed.split_once(marker) else {
        bail!("only github.com repositories are supported");
    };
    let without_git = rest.strip_suffix(".git").unwrap_or(rest);
    split_owner_repo(without_git)
}

fn split_owner_repo(path: &str) -> Result<(String, String)> {
    let mut parts = path.split('/').filter(|p| !p.is_empty());
    let owner = parts
        .next()
        .ok_or_else(|| anyhow!("missing GitHub owner"))?;
    let repo = parts.next().ok_or_else(|| anyhow!("missing GitHub repo"))?;
    if owner.contains("..") || repo.contains("..") {
        bail!("invalid GitHub repository path");
    }
    Ok((owner.to_string(), repo.to_string()))
}

pub async fn sync_source(source: &PackSource, cache_root: &Path) -> Result<PackVersion> {
    fs::create_dir_all(cache_root).with_context(|| format!("creating {}", cache_root.display()))?;
    let token = source.auth_ref.as_deref().and_then(resolve_token);
    let client = reqwest::Client::new();
    let commit_sha = resolve_commit(&client, source, token.as_deref()).await?;
    let zip = download_zipball(&client, source, &commit_sha, token.as_deref()).await?;
    let extracted = extract_zip(&zip, &scratch_dir(&source.id, &commit_sha)?)?;
    let pack_src = find_pack_subdir(&extracted, &source.pack_path)?;
    let manifest = read_manifest(&pack_src)?;
    let short = short_sha(&commit_sha);
    let cache_id = format!("{}-{}-{short}", slug(&source.id), slug(&manifest.id));
    let dest = cache_root.join(cache_id);
    if dest.exists() {
        fs::remove_dir_all(&dest).with_context(|| format!("clearing {}", dest.display()))?;
    }
    copy_dir(&pack_src, &dest)?;
    let content_hash = hash_dir(&dest)?;
    let version_id = format!("{}:{}:{}", source.id, manifest.id, commit_sha);
    Ok(PackVersion {
        id: version_id,
        source_id: source.id.clone(),
        pack_id: manifest.id,
        pack_name: manifest.name,
        manifest_version: manifest.version,
        commit_sha,
        ref_name: source.ref_name.clone(),
        content_hash,
        cache_dir: dest.to_string_lossy().to_string(),
        discovered_at: String::new(),
    })
}

pub fn cache_dirs() -> Vec<PathBuf> {
    if let Ok(val) = std::env::var("QUANT_FABRIC_GITHUB_PACKS") {
        return val
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
    }
    vec![PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../quant-fabric/packs/github"
    ))]
}

pub fn cache_root() -> Result<PathBuf> {
    let root = cache_dirs()
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no GitHub pack cache directory"))?;
    fs::create_dir_all(&root)?;
    Ok(root)
}

pub async fn sync_and_record_source(
    controlplane: &ControlStore,
    source_id: &str,
    cache_root: &Path,
) -> Result<PackVersion> {
    let source = controlplane
        .get_pack_source(source_id)
        .await?
        .ok_or_else(|| anyhow!("unknown pack source `{source_id}`"))?;
    match sync_source(&source, cache_root).await {
        Ok(version) => {
            controlplane.upsert_pack_version(&version).await?;
            controlplane
                .mark_pack_source_checked(&source.id, "synced", Some(&version.commit_sha), None)
                .await?;
            controlplane
                .record_pack_sync_event(
                    &source.id,
                    "synced",
                    Some(&version.commit_sha),
                    "pack synced",
                )
                .await?;
            Ok(version)
        }
        Err(error) => {
            let message = error.to_string();
            let _ = controlplane
                .mark_pack_source_checked(&source.id, "failed", None, Some(&message))
                .await;
            let _ = controlplane
                .record_pack_sync_event(&source.id, "failed", None, &message)
                .await;
            Err(error)
        }
    }
}

pub async fn apply_tracked_update(
    controlplane: &ControlStore,
    job: &mut Job,
    cache_root: &Path,
) -> Result<bool> {
    let Some(binding) = job.definition.get("packBinding").cloned() else {
        return Ok(false);
    };
    if binding.get("updatePolicy").and_then(|v| v.as_str()) != Some("track") {
        return Ok(false);
    }
    let source_id = binding
        .get("sourceId")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("tracked pack binding is missing sourceId"))?;
    let latest = sync_and_record_source(controlplane, source_id, cache_root).await?;
    if binding.get("versionId").and_then(|v| v.as_str()) == Some(latest.id.as_str()) {
        return Ok(false);
    }
    if let Some(current_pack_id) = binding.get("packId").and_then(|v| v.as_str()) {
        if current_pack_id != latest.pack_id {
            bail!(
                "latest pack id `{}` differs from bound pack `{current_pack_id}`",
                latest.pack_id
            );
        }
    }
    let job_template_name = binding
        .get("jobTemplate")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("tracked pack binding is missing jobTemplate"))?;
    let pack = load_cached_pack(&latest)?;
    let job_template = pack
        .job_templates
        .iter()
        .find(|jt| jt.name == job_template_name)
        .ok_or_else(|| {
            anyhow!("latest pack does not contain job template `{job_template_name}`")
        })?;
    if !job_template.inputs.is_empty() {
        bail!("latest job template declares new inputs and needs review");
    }

    job.definition["steps"] = merge_steps(&job.definition, &job_template.steps)?;
    if job
        .definition
        .get("schedule")
        .and_then(|v| v.as_str())
        .is_none()
    {
        if let Some(schedule) = &job_template.schedule {
            job.definition["schedule"] = serde_json::Value::String(schedule.clone());
        }
    }
    let mut next_binding = binding;
    next_binding["sourceId"] = serde_json::Value::String(latest.source_id.clone());
    next_binding["versionId"] = serde_json::Value::String(latest.id.clone());
    next_binding["packId"] = serde_json::Value::String(latest.pack_id.clone());
    next_binding["packName"] = serde_json::Value::String(latest.pack_name.clone());
    next_binding["refName"] = serde_json::Value::String(latest.ref_name.clone());
    next_binding["commitSha"] = serde_json::Value::String(latest.commit_sha.clone());
    next_binding["contentHash"] = serde_json::Value::String(latest.content_hash.clone());
    job.definition["packBinding"] = next_binding;
    controlplane.upsert_job(job).await?;
    Ok(true)
}

fn merge_steps(
    definition: &serde_json::Value,
    latest: &[quant_fabric::packs::JobTemplateStep],
) -> Result<serde_json::Value> {
    let existing = definition
        .get("steps")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut next = Vec::with_capacity(latest.len());
    for (idx, step) in latest.iter().enumerate() {
        let mut value = serde_json::to_value(step)?;
        if existing
            .get(idx)
            .and_then(|v| v.get("template"))
            .and_then(|v| v.as_str())
            == Some(step.template.as_str())
        {
            if let (Some(args), Some(obj)) = (
                existing
                    .get(idx)
                    .and_then(|v| v.get("args"))
                    .and_then(|v| v.as_object()),
                value.get_mut("args").and_then(|v| v.as_object_mut()),
            ) {
                for (key, val) in args {
                    obj.insert(key.clone(), val.clone());
                }
            }
        }
        next.push(value);
    }
    Ok(serde_json::Value::Array(next))
}

fn load_cached_pack(version: &PackVersion) -> Result<quant_fabric::packs::Pack> {
    let cache_dir = PathBuf::from(&version.cache_dir);
    let parent = cache_dir
        .parent()
        .ok_or_else(|| anyhow!("cached pack has no parent directory"))?;
    quant_fabric::packs::discover(
        &[parent.to_path_buf()],
        quant_fabric::packs::PackOrigin::Github,
    )
    .into_iter()
    .find(|pack| Path::new(&pack.dir) == cache_dir)
    .ok_or_else(|| anyhow!("cached pack not found at {}", cache_dir.display()))
}

async fn resolve_commit(
    client: &reqwest::Client,
    source: &PackSource,
    token: Option<&str>,
) -> Result<String> {
    let url = format!(
        "https://api.github.com/repos/{}/{}/commits/{}",
        source.owner, source.repo, source.ref_name
    );
    let response = request(client.get(url), token).send().await?;
    if !response.status().is_success() {
        bail!("GitHub commit lookup failed: HTTP {}", response.status());
    }
    Ok(response.json::<CommitResponse>().await?.sha)
}

async fn download_zipball(
    client: &reqwest::Client,
    source: &PackSource,
    commit_sha: &str,
    token: Option<&str>,
) -> Result<Vec<u8>> {
    let url = format!(
        "https://api.github.com/repos/{}/{}/zipball/{}",
        source.owner, source.repo, commit_sha
    );
    let response = request(client.get(url), token).send().await?;
    if !response.status().is_success() {
        bail!("GitHub zipball download failed: HTTP {}", response.status());
    }
    Ok(response.bytes().await?.to_vec())
}

fn request(builder: reqwest::RequestBuilder, token: Option<&str>) -> reqwest::RequestBuilder {
    let builder = builder
        .header(USER_AGENT, "quant-fabric-app")
        .header(ACCEPT, "application/vnd.github+json");
    if let Some(token) = token {
        builder.header(AUTHORIZATION, format!("Bearer {token}"))
    } else {
        builder
    }
}

fn resolve_token(auth_ref: &str) -> Option<String> {
    auth_ref
        .strip_prefix("env:")
        .and_then(|name| std::env::var(name).ok())
        .or_else(|| {
            if auth_ref.trim().is_empty() {
                None
            } else {
                Some(auth_ref.to_string())
            }
        })
}

fn scratch_dir(source_id: &str, commit_sha: &str) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!(
        "qf-github-pack-{}-{}",
        slug(source_id),
        short_sha(commit_sha)
    ));
    if dir.exists() {
        fs::remove_dir_all(&dir)?;
    }
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn extract_zip(zip_bytes: &[u8], dest: &Path) -> Result<PathBuf> {
    let mut archive = zip::ZipArchive::new(Cursor::new(zip_bytes))?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let Some(enclosed) = file.enclosed_name().map(|p| p.to_path_buf()) else {
            continue;
        };
        let out = dest.join(enclosed);
        if file.is_dir() {
            fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        fs::write(&out, bytes)?;
    }
    let mut roots = fs::read_dir(dest)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect::<Vec<_>>();
    roots.sort();
    roots
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("GitHub archive was empty"))
}

fn find_pack_subdir(extracted_root: &Path, pack_path: &str) -> Result<PathBuf> {
    if pack_path.trim().is_empty() {
        if extracted_root.join("pack.toml").is_file() {
            return Ok(extracted_root.to_path_buf());
        }
        let mut candidates = Vec::new();
        find_pack_tomls(extracted_root, &mut candidates);
        return candidates
            .into_iter()
            .next()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .ok_or_else(|| anyhow!("no pack.toml found in repository archive"));
    }
    let mut p = extracted_root.to_path_buf();
    for comp in Path::new(pack_path).components() {
        match comp {
            Component::Normal(c) => p.push(c),
            _ => bail!("invalid pack path `{pack_path}`"),
        }
    }
    if !p.join("pack.toml").is_file() {
        bail!("no pack.toml at `{pack_path}`");
    }
    Ok(p)
}

fn find_pack_tomls(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_pack_tomls(&path, out);
        } else if path.file_name().and_then(|n| n.to_str()) == Some("pack.toml") {
            out.push(path);
        }
    }
    out.sort();
}

fn read_manifest(pack_dir: &Path) -> Result<PackManifest> {
    let src = fs::read_to_string(pack_dir.join("pack.toml"))?;
    let parsed = toml::from_str::<ManifestFile>(&src).context("validating pack.toml")?;
    if parsed.pack.id.trim().is_empty() {
        bail!("pack.toml has an empty pack id");
    }
    Ok(parsed.pack)
}

fn copy_dir(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            fs::copy(&from, &to)
                .with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
        }
    }
    Ok(())
}

fn hash_dir(dir: &Path) -> Result<String> {
    let mut files = Vec::new();
    collect_files(dir, &mut files);
    files.sort();
    let mut hasher = Sha256::new();
    for file in files {
        let rel = file.strip_prefix(dir).unwrap_or(&file).to_string_lossy();
        hasher.update(rel.as_bytes());
        hasher.update([0]);
        hasher.update(fs::read(&file)?);
        hasher.update([0]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn slug(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn short_sha(sha: &str) -> String {
    sha.chars().take(12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use quant_fabric::packs::JobTemplateStep;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn step(template: &str, args: &[(&str, &str)]) -> JobTemplateStep {
        JobTemplateStep {
            template: template.to_string(),
            args: args
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn merge_steps_preserves_existing_args_for_same_template_position() {
        let definition = json!({
            "steps": [
                { "template": "bond_return_sharpe", "args": { "group_by": "desk", "returns": "daily_return" } }
            ]
        });
        let latest = vec![step(
            "bond_return_sharpe",
            &[("group_by", "portfolio"), ("returns", "ret")],
        )];

        let merged = merge_steps(&definition, &latest).expect("merge");

        assert_eq!(merged[0]["template"], "bond_return_sharpe");
        assert_eq!(merged[0]["args"]["group_by"], "desk");
        assert_eq!(merged[0]["args"]["returns"], "daily_return");
    }

    #[test]
    fn merge_steps_uses_pack_defaults_when_template_position_changes() {
        let definition = json!({
            "steps": [
                { "template": "old_template", "args": { "group_by": "desk", "returns": "daily_return" } }
            ]
        });
        let latest = vec![step(
            "bond_return_sharpe",
            &[("group_by", "portfolio"), ("returns", "ret")],
        )];

        let merged = merge_steps(&definition, &latest).expect("merge");

        assert_eq!(merged[0]["template"], "bond_return_sharpe");
        assert_eq!(merged[0]["args"]["group_by"], "portfolio");
        assert_eq!(merged[0]["args"]["returns"], "ret");
    }
}
