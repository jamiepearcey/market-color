import fs from "node:fs";
import path from "node:path";

export const STORE_VERSION = 1;

function envelope(data) {
  return {
    version: STORE_VERSION,
    data,
  };
}

function fsyncDirectory(dir) {
  const fd = fs.openSync(dir, "r");
  try {
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
}

export function writeStoreFile(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tempFile = `${file}.${process.pid}.${Date.now()}.tmp`;
  const payload = JSON.stringify(envelope(value), null, 2);
  const fd = fs.openSync(tempFile, "w", 0o600);
  try {
    fs.writeFileSync(fd, payload);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tempFile, file);
  fsyncDirectory(path.dirname(file));
}

function quarantineCorruptFile(file) {
  const backup = `${file}.${Date.now()}.bak`;
  fs.renameSync(file, backup);
  return backup;
}

function decodeStoreValue(parsed) {
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed) && parsed.version === STORE_VERSION && "data" in parsed) {
    return parsed.data;
  }
  return parsed;
}

export function readStoreFile(file, fallback, { log } = {}) {
  try {
    if (!fs.existsSync(file)) return fallback;
    const raw = fs.readFileSync(file, "utf8");
    if (!raw.trim()) return fallback;
    return decodeStoreValue(JSON.parse(raw));
  } catch (err) {
    let backup = null;
    try {
      if (fs.existsSync(file)) {
        backup = quarantineCorruptFile(file);
      }
    } catch (quarantineErr) {
      log?.("store quarantine failed", file, quarantineErr?.message ?? String(quarantineErr));
    }
    writeStoreFile(file, fallback);
    log?.("store reset after parse failure", file, err?.message ?? String(err), backup);
    return fallback;
  }
}
