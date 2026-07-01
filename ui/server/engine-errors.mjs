import { ERROR_CODES, appError } from "./errors.mjs";

function combinedText(result) {
  return `${result?.stderr || ""}\n${result?.stdout || ""}`.toLowerCase();
}

export function classifyEngineFailure(result, fallbackMessage = "celeritas command failed") {
  const text = combinedText(result);
  const message = (result?.stderr || result?.stdout || fallbackMessage || "celeritas command failed").trim();

  if (
    /unknown (plugin|connector|extractor|loader)/i.test(text)
    || /plugin .* not found/i.test(text)
    || /connector .* not found/i.test(text)
  ) {
    return appError(ERROR_CODES.CONNECTOR_NOT_FOUND, 404, message);
  }
  if (
    /not installed/i.test(text)
    || /needs install/i.test(text)
    || /run celeritas add/i.test(text)
    || /cargo install/i.test(text)
  ) {
    return appError(ERROR_CODES.NEEDS_INSTALL, 409, message);
  }
  if (
    /validation/i.test(text)
    || /invalid config/i.test(text)
    || /invalid configuration/i.test(text)
    || /missing required/i.test(text)
    || /unknown field/i.test(text)
  ) {
    return appError(ERROR_CODES.CONFIG_INVALID, 400, message);
  }
  if (
    /authentication failed/i.test(text)
    || /unauthorized/i.test(text)
    || /forbidden/i.test(text)
    || /access denied/i.test(text)
    || /permission denied/i.test(text)
  ) {
    return appError(ERROR_CODES.AUTH_FAILED, 401, message);
  }
  if (
    /connection refused/i.test(text)
    || /econnrefused/i.test(text)
    || /network is unreachable/i.test(text)
    || /timed out/i.test(text)
    || /temporary failure in name resolution/i.test(text)
  ) {
    return appError(ERROR_CODES.CONNECTIVITY, 502, message);
  }
  return appError(ERROR_CODES.ENGINE_FAILED, 502, message || fallbackMessage);
}
