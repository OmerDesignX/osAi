import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource/manrope/400.css";
import "@fontsource/manrope/600.css";
import "@fontsource/playfair-display/600.css";
import "@fontsource/fira-code/400.css";
import "./styles.css";
import osAiIcon from "./assets/osai-icon.png";

const root = ReactDOM.createRoot(document.getElementById("root")!);

if (!window.osai) {
  root.render(
    <main className="bridge-error">
      <img src={osAiIcon} alt="" />
      <h1>osAi couldn&apos;t start</h1>
      <p>The secure desktop bridge did not load. Restart or reinstall osAi.</p>
    </main>,
  );
} else {
  void import("./App").then(({ App }) => {
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );
  });
}
