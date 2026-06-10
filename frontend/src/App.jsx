import { useState } from "react";
import Demo      from "./pages/Demo";
import Analytics from "./pages/Analytics";

export default function App() {
  const [page, setPage] = useState("demo");

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0a" }}>

      {/* bottom nav */}
      <nav style={{
        position:        "fixed",
        bottom:          0,
        left:            0,
        right:           0,
        background:      "#1a1a1a",
        borderTop:       "1px solid #2a2a2a",
        display:         "flex",
        justifyContent:  "center",
        gap:             "48px",
        padding:         "12px 0 20px",
        zIndex:          100,
      }}>
        {[
          { id: "demo",      label: "Demo",      icon: "⚡" },
          { id: "analytics", label: "Analytics", icon: "📊" },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setPage(tab.id)}
            style={{
              background:  "none",
              color:       page === tab.id ? "#8b5cf6" : "#6b7280",
              display:     "flex",
              flexDirection:"column",
              alignItems:  "center",
              gap:         "3px",
              fontSize:    "18px",
              padding:     "4px 12px",
              borderRadius:"8px",
              transition:  "color 0.2s",
            }}
          >
            <span>{tab.icon}</span>
            <span style={{ fontSize: "10px", fontWeight: "600" }}>
              {tab.label}
            </span>
          </button>
        ))}
      </nav>

      {/* page content with bottom padding for nav */}
      <div style={{ paddingBottom: "80px" }}>
        {page === "demo"      && <Demo />}
        {page === "analytics" && <Analytics />}
      </div>
    </div>
  );
}