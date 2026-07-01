export function parseRunSummary(stdout) {
  const streams = [];
  let total = 0;
  const lines = String(stdout || "").split("\n");
  let inRecords = false;
  for (const raw of lines) {
    const line = raw.replace(/\x1b\[[0-9;]*m/g, "");
    const trimmed = line.trim();
    const totalMatch = trimmed.match(/^Total records:\s*(\d+)/);
    if (totalMatch) {
      total = Number(totalMatch[1]);
      inRecords = false;
      continue;
    }
    if (/^Records:\s*$/.test(trimmed)) {
      inRecords = true;
      continue;
    }
    if (!inRecords) {
      continue;
    }
    if (trimmed === "(none)") {
      continue;
    }
    const streamMatch = trimmed.match(/^(.+?):\s*(\d+)\s*$/);
    if (streamMatch) {
      streams.push({ stream: streamMatch[1].trim(), count: Number(streamMatch[2]) });
      continue;
    }
    if (trimmed && !trimmed.startsWith("[")) {
      inRecords = false;
    }
  }
  return { streams, total };
}
