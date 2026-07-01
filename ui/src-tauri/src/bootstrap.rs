//! Cluster bootstrap wizard (ADR-0021 P6 / ADR-0005 / ADR-0006).
//!
//! Turns a desired-topology description into the exact `quant-fabric` launch
//! commands — coordinator + each worker — including OpenRaft HA flags and the
//! `QUANT_FABRIC_TLS_*` mTLS environment. Pure + deterministic so it's unit-
//! tested in cargo (no GUI needed); the desktop wizard just renders the result
//! with copy buttons. The engine uses a worker-pull model (workers self-register
//! via `POST /v1/workers/register`), so "joining a node" *is* launching a worker
//! pointed at the coordinator — exactly what these commands do.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerPlan {
    /// Local bind address, e.g. `0.0.0.0:7101`.
    pub bind: String,
    /// URL the coordinator/peers reach this worker on, e.g. `http://10.0.0.5:7101`.
    pub advertise: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub slots: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RaftPlan {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub data_dir: Option<String>,
    #[serde(default)]
    pub node_id: Option<u64>,
    /// All member `base_url`s including self (ADR-0005 §6).
    #[serde(default)]
    pub peers: Vec<String>,
    /// Emit `--raft-init` (one-shot single-node bootstrap).
    #[serde(default)]
    pub init: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MtlsPlan {
    #[serde(default)]
    pub enabled: bool,
    /// PEM bundle of trusted cluster CA certs (shared by all nodes).
    #[serde(default)]
    pub ca_bundle: Option<String>,
    /// Reject non-mTLS peers (the end-state of the ADR-0006 §6 migration).
    #[serde(default)]
    pub required: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterPlan {
    /// Coordinator local bind address, e.g. `0.0.0.0:7000`.
    pub coordinator_bind: String,
    /// Coordinator URL workers connect to, e.g. `http://10.0.0.1:7000`.
    pub coordinator_url: String,
    #[serde(default)]
    pub workers: Vec<WorkerPlan>,
    #[serde(default)]
    pub raft: Option<RaftPlan>,
    #[serde(default)]
    pub mtls: Option<MtlsPlan>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EnvVar {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandBlock {
    /// `coordinator` | `worker`.
    pub role: String,
    pub title: String,
    /// Environment to export before the command (mTLS, etc.).
    pub env: Vec<EnvVar>,
    pub command: String,
}

fn env(key: &str, value: impl Into<String>) -> EnvVar {
    EnvVar {
        key: key.to_string(),
        value: value.into(),
    }
}

/// The `QUANT_FABRIC_TLS_*` env for one node. `cert_base` names this node's leaf
/// cert/key (each node needs its own); the CA bundle is shared.
fn mtls_env(m: &MtlsPlan, cert_base: &str) -> Vec<EnvVar> {
    if !m.enabled {
        return Vec::new();
    }
    let mut v = vec![env("QUANT_FABRIC_TLS_MODE", "files")];
    if let Some(ca) = &m.ca_bundle {
        v.push(env("QUANT_FABRIC_TLS_CA_BUNDLE", ca.clone()));
    }
    v.push(env("QUANT_FABRIC_TLS_CERT", format!("{cert_base}.pem")));
    v.push(env("QUANT_FABRIC_TLS_KEY", format!("{cert_base}-key.pem")));
    if m.required {
        v.push(env("QUANT_FABRIC_TLS_REQUIRED", "true"));
    }
    v
}

/// Generate the launch commands for the whole cluster from a desired topology.
pub fn generate_bootstrap(plan: &ClusterPlan) -> Vec<CommandBlock> {
    let mtls = plan.mtls.clone().unwrap_or_default();
    let mut blocks = Vec::new();

    // --- Coordinator ---
    let mut cmd = format!("quant-fabric coordinator --bind {}", plan.coordinator_bind);
    if let Some(raft) = &plan.raft {
        if raft.enabled {
            if let Some(dir) = &raft.data_dir {
                cmd.push_str(&format!(" --raft-data-dir {dir}"));
            }
            if let Some(id) = raft.node_id {
                cmd.push_str(&format!(" --raft-node-id {id}"));
            }
            if !raft.peers.is_empty() {
                cmd.push_str(&format!(" --raft-peers {}", raft.peers.join(",")));
            }
            if raft.init {
                cmd.push_str(" --raft-init");
            }
        }
    }
    blocks.push(CommandBlock {
        role: "coordinator".into(),
        title: "Coordinator".into(),
        env: mtls_env(&mtls, "coordinator"),
        command: cmd,
    });

    // --- Workers ---
    for (i, w) in plan.workers.iter().enumerate() {
        let name = w
            .name
            .clone()
            .unwrap_or_else(|| format!("worker-{}", i + 1));
        let mut cmd = format!(
            "quant-fabric worker --coordinator {} --bind {} --advertise {} --pull",
            plan.coordinator_url, w.bind, w.advertise
        );
        if let Some(s) = w.slots {
            cmd.push_str(&format!(" --slots {s}"));
        }
        cmd.push_str(&format!(" --name {name}"));
        blocks.push(CommandBlock {
            role: "worker".into(),
            title: name.clone(),
            env: mtls_env(&mtls, &name),
            command: cmd,
        });
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(bind: &str, advertise: &str, name: Option<&str>, slots: Option<u32>) -> WorkerPlan {
        WorkerPlan {
            bind: bind.into(),
            advertise: advertise.into(),
            name: name.map(String::from),
            slots,
        }
    }

    #[test]
    fn single_node_bootstrap_is_one_coordinator_with_raft_init() {
        let plan = ClusterPlan {
            coordinator_bind: "127.0.0.1:7000".into(),
            coordinator_url: "http://127.0.0.1:7000".into(),
            workers: vec![worker(
                "127.0.0.1:7101",
                "http://127.0.0.1:7101",
                None,
                Some(4),
            )],
            raft: Some(RaftPlan {
                enabled: true,
                data_dir: Some("/var/lib/qf/raft".into()),
                node_id: Some(1),
                peers: vec!["http://127.0.0.1:7000".into()],
                init: true,
            }),
            mtls: None,
        };
        let blocks = generate_bootstrap(&plan);
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0].command,
            "quant-fabric coordinator --bind 127.0.0.1:7000 \
             --raft-data-dir /var/lib/qf/raft --raft-node-id 1 \
             --raft-peers http://127.0.0.1:7000 --raft-init"
        );
        assert!(blocks[0].env.is_empty(), "no mTLS env when disabled");
        assert_eq!(
            blocks[1].command,
            "quant-fabric worker --coordinator http://127.0.0.1:7000 \
             --bind 127.0.0.1:7101 --advertise http://127.0.0.1:7101 --pull \
             --slots 4 --name worker-1"
        );
    }

    #[test]
    fn multi_node_with_mtls_emits_tls_env_per_node() {
        let plan = ClusterPlan {
            coordinator_bind: "0.0.0.0:7000".into(),
            coordinator_url: "https://coord.internal:7000".into(),
            workers: vec![
                worker(
                    "0.0.0.0:7101",
                    "https://w1.internal:7101",
                    Some("alpha"),
                    None,
                ),
                worker("0.0.0.0:7102", "https://w2.internal:7102", None, Some(8)),
            ],
            raft: None,
            mtls: Some(MtlsPlan {
                enabled: true,
                ca_bundle: Some("/etc/qf/ca.pem".into()),
                required: true,
            }),
        };
        let blocks = generate_bootstrap(&plan);
        assert_eq!(blocks.len(), 3);
        // Coordinator gets a coordinator-named leaf cert + shared CA + required flag.
        assert_eq!(
            blocks[0].env,
            vec![
                env("QUANT_FABRIC_TLS_MODE", "files"),
                env("QUANT_FABRIC_TLS_CA_BUNDLE", "/etc/qf/ca.pem"),
                env("QUANT_FABRIC_TLS_CERT", "coordinator.pem"),
                env("QUANT_FABRIC_TLS_KEY", "coordinator-key.pem"),
                env("QUANT_FABRIC_TLS_REQUIRED", "true"),
            ]
        );
        // Named worker uses its name for the leaf cert; unnamed falls back to index.
        assert_eq!(blocks[1].title, "alpha");
        assert!(blocks[1]
            .env
            .contains(&env("QUANT_FABRIC_TLS_CERT", "alpha.pem")));
        assert_eq!(blocks[2].title, "worker-2");
        assert!(blocks[2]
            .env
            .contains(&env("QUANT_FABRIC_TLS_KEY", "worker-2-key.pem")));
        // No raft flags when raft is disabled.
        assert!(!blocks[0].command.contains("--raft"));
    }
}
