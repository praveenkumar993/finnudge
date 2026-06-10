export default function PersonalizationBadge({ data }) {
  if (!data) return null;

  const color = data.score >= 80 ? "#8b5cf6"
              : data.score >= 50 ? "#10b981"
              : data.score >= 20 ? "#f59e0b"
              : "#6b7280";

  return (
    <div style={{
      background:   "#1a1a1a",
      border:       `1px solid ${color}40`,
      borderRadius: "12px",
      padding:      "12px 16px",
      display:      "flex",
      alignItems:   "center",
      gap:          "12px",
    }}>

      {/* circular progress */}
      <div style={{ position: "relative", flexShrink: 0 }}>
        <svg width="48" height="48" viewBox="0 0 48 48">
          <circle cx="24" cy="24" r="20"
            fill="none" stroke="#2a2a2a" strokeWidth="4"/>
          <circle cx="24" cy="24" r="20"
            fill="none"
            stroke={color}
            strokeWidth="4"
            strokeDasharray={`${data.score * 1.257} 125.7`}
            strokeLinecap="round"
            transform="rotate(-90 24 24)"
          />
        </svg>
        <span style={{
          position:  "absolute",
          top: "50%", left: "50%",
          transform: "translate(-50%, -50%)",
          fontSize:  "11px",
          fontWeight:"700",
          color:     color,
        }}>
          {data.score}
        </span>
      </div>

      <div>
        <div style={{
          fontSize:   "13px",
          fontWeight: "600",
          color:      "#f1f5f9",
        }}>
          {data.label}
        </div>
        <div style={{
          fontSize: "11px",
          color:    "#6b7280",
          marginTop:"2px",
        }}>
          {data.archetype} · drift {data.drift?.toFixed(4)}
        </div>
      </div>
    </div>
  );
}