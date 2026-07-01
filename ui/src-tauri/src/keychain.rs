//! OS user keychain custody for Celeritas secrets.
//!
//! The secrets-keeper bearer token is "secret zero" — it unlocks every other
//! secret Celeritas reads — so it must not sit in the plaintext pref store.
//! It lives here instead: macOS Keychain / Windows Credential Manager / Linux
//! Secret Service, via the `keyring` crate. This module is the single custody
//! boundary; callers never see where the bytes rest.
//!
//! Operations are synchronous (the `keyring` API is blocking and may briefly
//! prompt the OS). Call from `spawn_blocking` on hot paths.

const SERVICE: &str = "celeritas";

fn entry(account: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE, account).map_err(|e| e.to_string())
}

/// Read a stored secret, or `None` when no entry exists.
pub fn get(account: &str) -> Result<Option<String>, String> {
    match entry(account)?.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(err) => Err(err.to_string()),
    }
}

/// Store a secret. An empty value deletes the entry (so "clear the token" is
/// just saving an empty field).
pub fn set(account: &str, value: &str) -> Result<(), String> {
    let entry = entry(account)?;
    if value.is_empty() {
        return match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(err) => Err(err.to_string()),
        };
    }
    entry.set_password(value).map_err(|err| err.to_string())
}
