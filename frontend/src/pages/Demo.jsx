import { useState, useEffect } from "react";
import axios from "axios";
import NudgeCard from "../components/NudgeCard";
import PersonalizationBadge from "../components/PersonalizationBadge";
import { User, RefreshCw } from "lucide-react";

const API = import.meta.env.VITE_API_URL ||
            "http://localhost:8000";

const ACTIONS = [
  { value: "opened_app",      label: "🏠 Opened App" },
  { value: "checked_stocks",  label: "📈 Checked Stocks" },
  { value: "paid_bill",       label: "⚡ Paid a Bill" },
  { value: "received_salary", label: "💰 Received Salary" },
  { value: "checked_balance", label: "💳 Checked Balance" },
  { value: "made_payment",    label: "✅ Made Payment" },
];

// one user per archetype — shows clear difference
const SAMPLE_USERS = [
  { id: "user_0001", label: "user_0001 — Investor" },
  { id: "user_0003", label: "user_0003 — Saver" },
  { id: "user_0005", label: "user_0005 — Spender" },
  { id: "user_0007", label: "user_0007 — Bill Payer" },
  { id: "user_0009", label: "user_0009 — Borrower" },
  { id: "user_0002", label: "user_0002 — Investor" },
  { id: "user_0004", label: "user_0004 — Mixed" },
  { id: "user_0006", label: "user_0006 — Bill Payer" },
];

export default function Demo() {
  const [userId,      setUserId]      = useState("user_0001");
  const [action,      setAction]      = useState("opened_app");
  const [result,      setResult]      = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [feedback,    setFeedback]    = useState({});
  const [actedNudges, setActedNudges] = useState(new Set());
  const [eventLog,    setEventLog]    = useState([]);
  const [userInfo,    setUserInfo]    = useState(null);

  const fetchUserInfo = async (uid) => {
    try {
      const res = await axios.get(`${API}/user/${uid}`);
      setUserInfo(res.data);
    } catch {}
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    setActedNudges(new Set());
    setFeedback({});
    try {
      const res = await axios.post(`${API}/recommend`, {
        user_id:        userId,
        top_k:          3,
        current_action: action,
      });
      setResult(res.data);
      addLog(`Fetched nudges for ${userId} · ${res.data.ab_group}`);
    } catch (err) {
      addLog("Error fetching recommendations");
    }
    setLoading(false);
  };

  const handleAction = async (nudgeId, actionType) => {
    if (actedNudges.has(nudgeId)) return;
    setActedNudges(prev => new Set([...prev, nudgeId]));
    setFeedback(prev => ({ ...prev, [nudgeId]: actionType }));
    try {
      await axios.post(`${API}/feedback`, {
        user_id:  userId,
        nudge_id: nudgeId,
        action:   actionType,
        ab_group: result?.ab_group || "treatment",
      });
      addLog(`${actionType.toUpperCase()} → ${nudgeId}`);
      if (actionType === "click") {
        setTimeout(fetchRecommendations, 800);
      }
    } catch {}
  };

  const addLog = (msg) => {
    const time = new Date().toLocaleTimeString();
    setEventLog(prev =>
      [`[${time}] ${msg}`, ...prev].slice(0, 8)
    );
  };

  // when user changes — fetch their info + new recommendations
  useEffect(() => {
    fetchUserInfo(userId);
    fetchRecommendations();
  }, [userId]);

  // when action changes — just re-fetch
  useEffect(() => {
    if (result) fetchRecommendations();
  }, [action]);

  return (
    <div style={{
      maxWidth:  "480px",
      margin:    "0 auto",
      padding:   "24px 16px",
      minHeight: "100vh",
    }}>

      {/* header */}
      <div style={{
        display:        "flex",
        justifyContent: "space-between",
        alignItems:     "center",
        marginBottom:   "24px",
      }}>
        <div>
          <h1 style={{
            fontSize:   "22px",
            fontWeight: "700",
            background: "linear-gradient(135deg, #8b5cf6, #ec4899)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor:  "transparent",
          }}>
            FinNudge
          </h1>
          <p style={{ fontSize: "12px", color: "#6b7280" }}>
            Real-time personalization engine
          </p>
        </div>
        <button
          onClick={fetchRecommendations}
          style={{
            background:   "#1a1a1a",
            border:       "1px solid #2a2a2a",
            borderRadius: "10px",
            padding:      "8px 12px",
            color:        "#6b7280",
            display:      "flex",
            alignItems:   "center",
            gap:          "6px",
            fontSize:     "13px",
          }}
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* user info card — shows archetype + city + age */}
      {userInfo && (
        <div style={{
          background:   "#1a1a1a",
          border:       "1px solid #2a2a2a",
          borderRadius: "12px",
          padding:      "12px 16px",
          marginBottom: "12px",
          display:      "flex",
          gap:          "12px",
          alignItems:   "center",
        }}>
          <div style={{
            width:          "40px",
            height:         "40px",
            borderRadius:   "50%",
            background:     "linear-gradient(135deg, #8b5cf6, #ec4899)",
            display:        "flex",
            alignItems:     "center",
            justifyContent: "center",
            fontSize:       "16px",
            flexShrink:     0,
          }}>
            {userInfo.archetype === "investor"   ? "📈" :
             userInfo.archetype === "saver"      ? "🏦" :
             userInfo.archetype === "spender"    ? "🛍️" :
             userInfo.archetype === "bill_payer" ? "⚡" :
             userInfo.archetype === "borrower"   ? "💳" : "👤"}
          </div>
          <div>
            <p style={{
              fontSize:   "13px",
              fontWeight: "600",
              color:      "#f1f5f9",
            }}>
              {userInfo.user_id}
            </p>
            <p style={{ fontSize: "11px", color: "#6b7280" }}>
              {userInfo.archetype} · {userInfo.city} · Age {userInfo.age}
            </p>
          </div>
        </div>
      )}

      {/* controls */}
      <div style={{
        background:   "#1a1a1a",
        border:       "1px solid #2a2a2a",
        borderRadius: "16px",
        padding:      "16px",
        marginBottom: "16px",
      }}>

        {/* user selector */}
        <div style={{ marginBottom: "12px" }}>
          <label style={{
            fontSize:     "11px",
            color:        "#6b7280",
            display:      "block",
            marginBottom: "6px",
            fontWeight:   "600",
            letterSpacing:"0.05em",
          }}>
            SELECT USER
          </label>
          <select
            value={userId}
            onChange={e => setUserId(e.target.value)}
            style={{
              width:        "100%",
              background:   "#242424",
              color:        "#f1f5f9",
              padding:      "10px 12px",
              borderRadius: "10px",
              fontSize:     "13px",
              border:       "1px solid #2a2a2a",
            }}
          >
            {SAMPLE_USERS.map(u => (
              <option key={u.id} value={u.id}>
                {u.label}
              </option>
            ))}
          </select>
        </div>

        {/* action selector */}
        <div>
          <label style={{
            fontSize:     "11px",
            color:        "#6b7280",
            display:      "block",
            marginBottom: "6px",
            fontWeight:   "600",
            letterSpacing:"0.05em",
          }}>
            WHAT DID THEY JUST DO?
          </label>
          <select
            value={action}
            onChange={e => setAction(e.target.value)}
            style={{
              width:        "100%",
              background:   "#242424",
              color:        "#f1f5f9",
              padding:      "10px 12px",
              borderRadius: "10px",
              fontSize:     "13px",
              border:       "1px solid #2a2a2a",
            }}
          >
            {ACTIONS.map(a => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={fetchRecommendations}
          disabled={loading}
          style={{
            width:        "100%",
            marginTop:    "12px",
            padding:      "12px",
            background:   loading
                          ? "#2a2a2a"
                          : "linear-gradient(135deg, #8b5cf6, #7c3aed)",
            color:        loading ? "#6b7280" : "#fff",
            borderRadius: "10px",
            fontSize:     "14px",
            fontWeight:   "600",
            transition:   "all 0.2s",
          }}
        >
          {loading ? "Loading..." : "⚡ Get Nudges"}
        </button>
      </div>

      {/* result */}
      {result && (
        <>
          {/* status bar */}
          <div style={{
            display:      "flex",
            gap:          "8px",
            marginBottom: "12px",
            flexWrap:     "wrap",
          }}>
            <span style={{
              fontSize:   "11px",
              background: "#1a1a1a",
              border:     "1px solid #2a2a2a",
              padding:    "4px 10px",
              borderRadius:"20px",
              color:      "#6b7280",
              display:    "flex",
              alignItems: "center",
              gap:        "4px",
            }}>
              <User size={10} />
              {result.archetype}
            </span>

            <span style={{
              fontSize:   "11px",
              background: result.ab_group === "treatment"
                          ? "#1e1b4b" : "#242424",
              border:     "1px solid #2a2a2a",
              padding:    "4px 10px",
              borderRadius:"20px",
              color:      result.ab_group === "treatment"
                          ? "#a78bfa" : "#6b7280",
            }}>
              {result.ab_group === "treatment"
               ? "🧪 Personalized" : "🎲 Control (Random)"}
            </span>

            <span style={{
              fontSize:   "11px",
              background: "#1a1a1a",
              border:     "1px solid #2a2a2a",
              padding:    "4px 10px",
              borderRadius:"20px",
              color:      result.latency_ms < 20
                          ? "#10b981" : "#6b7280",
            }}>
              ⚡ {result.latency_ms}ms
            </span>

            <span style={{
              fontSize:   "11px",
              background: "#1a1a1a",
              border:     "1px solid #2a2a2a",
              padding:    "4px 10px",
              borderRadius:"20px",
              color:      "#6b7280",
            }}>
              📍 {result.context}
            </span>
          </div>

          {/* personalization badge */}
          {result.personalization && (
            <div style={{ marginBottom: "12px" }}>
              <PersonalizationBadge
                data={result.personalization}
              />
            </div>
          )}

          {/* nudge cards */}
          <div>
            {result.nudges.map((nudge, i) => (
              <div key={nudge.nudge_id} style={{
                opacity:   actedNudges.has(nudge.nudge_id)
                           ? 0.5 : 1,
                transition:"opacity 0.3s",
                position:  "relative",
              }}>
                {actedNudges.has(nudge.nudge_id) && (
                  <div style={{
                    position:    "absolute",
                    top: "50%",  left: "50%",
                    transform:   "translate(-50%, -50%)",
                    zIndex:      10,
                    background:  feedback[nudge.nudge_id] === "click"
                                 ? "#10b981" : "#ef4444",
                    color:       "#fff",
                    padding:     "6px 14px",
                    borderRadius:"20px",
                    fontSize:    "12px",
                    fontWeight:  "700",
                  }}>
                    {feedback[nudge.nudge_id] === "click"
                     ? "✓ Acted" : "Skipped"}
                  </div>
                )}
                <NudgeCard
                  nudge={nudge}
                  index={i}
                  onAction={handleAction}
                />
              </div>
            ))}
          </div>

          {/* what changed explainer */}
          <div style={{
            background:   "#1a1a1a",
            border:       "1px solid #2a2a2a",
            borderRadius: "12px",
            padding:      "14px 16px",
            marginTop:    "12px",
          }}>
            <p style={{
              fontSize:     "10px",
              color:        "#6b7280",
              fontWeight:   "600",
              marginBottom: "8px",
              letterSpacing:"0.05em",
            }}>
              HOW THIS WORKS
            </p>
            <p style={{
              fontSize:   "12px",
              color:      "#4b5563",
              lineHeight: "1.6",
            }}>
              {result.ab_group === "treatment"
               ? `Two-tower neural network matched 
                  ${result.archetype} behavioral embedding 
                  to top nudges via FAISS retrieval. 
                  Context: "${result.context}" applied 
                  category boost. Personalization score: 
                  ${result.personalization?.score}/100.`
               : `This user is in the control group — 
                  receiving random nudges for A/B comparison. 
                  Switch to a treatment user to see 
                  personalization in action.`}
            </p>
          </div>

          {/* event log */}
          {eventLog.length > 0 && (
            <div style={{
              background:   "#1a1a1a",
              border:       "1px solid #2a2a2a",
              borderRadius: "12px",
              padding:      "12px",
              marginTop:    "12px",
            }}>
              <p style={{
                fontSize:     "10px",
                color:        "#6b7280",
                marginBottom: "8px",
                fontWeight:   "600",
                letterSpacing:"0.05em",
              }}>
                EVENT LOG
              </p>
              {eventLog.map((log, i) => (
                <p key={i} style={{
                  fontSize:    "11px",
                  color:       i === 0 ? "#a78bfa" : "#4b5563",
                  marginBottom:"3px",
                  fontFamily:  "monospace",
                }}>
                  {log}
                </p>
              ))}
            </div>
          )}
        </>
      )}

      <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}