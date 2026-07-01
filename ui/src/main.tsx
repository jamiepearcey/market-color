import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Market Color is a dark-themed chat app; honor a saved theme override if present.
try {
  const theme = localStorage.getItem("ui.theme");
  if (theme === "light" || theme === "dark") {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
  }
} catch {
  /* ignore */
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
