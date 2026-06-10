import { TrendingUp, Bell, AlertCircle, 
         PiggyBank, CreditCard, Gift, 
         Shield, Tag, Zap } from "lucide-react";

const CATEGORY_CONFIG = {
  investment: { icon: TrendingUp, color: "#8b5cf6", bg: "#1e1b4b" },
  savings:    { icon: PiggyBank,  color: "#10b981", bg: "#064e3b" },
  reminder:   { icon: Bell,       color: "#f59e0b", bg: "#451a03" },
  alert:      { icon: AlertCircle,color: "#ef4444", bg: "#450a0a" },
  emi:        { icon: CreditCard, color: "#3b82f6", bg: "#1e3a5f" },
  rewards:    { icon: Gift,       color: "#ec4899", bg: "#4a044e" },
  insurance:  { icon: Shield,     color: "#14b8a6", bg: "#042f2e" },
  offers:     { icon: Tag,        color: "#f97316", bg: "#431407" },
  budgeting:  { icon: Zap,        color: "#eab308", bg: "#422006" },
};

export default function NudgeCard({ nudge, onAction, index }) {
  const config = CATEGORY_CONFIG[nudge.category] || 
                 CATEGORY_CONFIG.investment;
  const Icon   = config.icon;

  return (
    <div style={{
      background:    "#1a1a1a",
      border:        `1px solid ${nudge.context_boosted 
                       ? config.color + "60" : "#2a2a2a"}`,
      borderRadius:  "16px",
      padding:       "20px",
      marginBottom:  "12px",
      animation:     `slideIn 0.3s ease ${index * 0.1}s both`,
      position:      "relative",
      overflow:      "hidden",
    }}>

      {/* glow effect for context boosted */}
      {nudge.context_boosted && (
        <div style={{
          position:   "absolute",
          top: 0, left: 0, right: 0,
          height:     "2px",
          background: `linear-gradient(90deg, 
                       transparent, ${config.color}, transparent)`,
        }}/>
      )}

      <div style={{ display: "flex", 
                    alignItems: "flex-start", gap: "14px" }}>

        {/* icon */}
        <div style={{
          width: "44px", height: "44px",
          borderRadius: "12px",
          background:   config.bg,
          display:      "flex",
          alignItems:   "center",
          justifyContent: "center",
          flexShrink:   0,
        }}>
          <Icon size={20} color={config.color} />
        </div>

        {/* content */}
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", 
                        justifyContent: "space-between",
                        alignItems: "flex-start" }}>
            <h3 style={{
              fontSize:   "15px",
              fontWeight: "600",
              color:      "#f1f5f9",
              lineHeight: "1.4",
              maxWidth:   "75%",
            }}>
              {nudge.title}
            </h3>

            {/* score badge */}
            <span style={{
              fontSize:   "11px",
              fontWeight: "700",
              color:      config.color,
              background: config.bg,
              padding:    "3px 8px",
              borderRadius: "20px",
            }}>
              {Math.round(nudge.score * 100)}%
            </span>
          </div>

          {/* explanation */}
          <p style={{
            fontSize:   "12px",
            color:      "#6b7280",
            marginTop:  "6px",
            lineHeight: "1.5",
          }}>
            {nudge.explanation}
          </p>

          {/* tags row */}
          <div style={{
            display:    "flex",
            gap:        "6px",
            marginTop:  "10px",
            flexWrap:   "wrap",
          }}>
            {nudge.context_boosted && (
              <span style={{
                fontSize:   "10px",
                color:      config.color,
                background: config.bg,
                padding:    "2px 8px",
                borderRadius: "10px",
                fontWeight: "600",
              }}>
                ⚡ Context boosted
              </span>
            )}
            {nudge.is_replacement && (
              <span style={{
                fontSize:   "10px",
                color:      "#10b981",
                background: "#064e3b",
                padding:    "2px 8px",
                borderRadius: "10px",
                fontWeight: "600",
              }}>
                ✨ Fresh pick
              </span>
            )}
            <span style={{
              fontSize:   "10px",
              color:      "#6b7280",
              background: "#242424",
              padding:    "2px 8px",
              borderRadius: "10px",
            }}>
              {nudge.category}
            </span>
          </div>

          {/* action buttons */}
          <div style={{
            display:   "flex",
            gap:       "8px",
            marginTop: "14px",
          }}>
            <button
              onClick={() => onAction(nudge.nudge_id, "click")}
              style={{
                flex:         1,
                padding:      "9px",
                background:   config.color,
                color:        "#fff",
                borderRadius: "10px",
                fontSize:     "13px",
                fontWeight:   "600",
                transition:   "opacity 0.2s",
              }}
              onMouseOver={e => e.target.style.opacity = "0.85"}
              onMouseOut={e  => e.target.style.opacity = "1"}
            >
              Take Action
            </button>
            <button
              onClick={() => onAction(nudge.nudge_id, "skip")}
              style={{
                padding:      "9px 16px",
                background:   "#242424",
                color:        "#6b7280",
                borderRadius: "10px",
                fontSize:     "13px",
              }}
            >
              Skip
            </button>
            <button
              onClick={() => onAction(nudge.nudge_id, "dismiss")}
              style={{
                padding:      "9px 16px",
                background:   "#242424",
                color:        "#6b7280",
                borderRadius: "10px",
                fontSize:     "13px",
              }}
            >
              ✕
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}