// Minimal Tauri shell for Market Color.
//
// The desktop client is a thin webview that loads the static UI (dist/). All
// data flows over HTTP to the Market Color API server, whose base URL the user
// configures in the app's Settings — so there are no native commands here and
// the same UI runs unchanged in a browser or bundled in Tauri.
//
// (The Celeritas donor backend that shipped with this template is parked on
// disk under src/ but not compiled — autobins/autolib are off and only this
// bin is declared in Cargo.toml.)

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running Market Color");
}
