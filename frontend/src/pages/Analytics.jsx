import { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from "recharts";

const API = import.meta.env.VITE_API_URL ||
            "http://localhost:8000";

const COLORS = ["#8b5cf6", "#10b981", "#f59e0b", "#ef4444"];

export default function Analytics() {
  const [abData,   setAbData]   = useState(null);
  const [history,  setHistory]  = useState([]);
  const [persona,  setPersona]  = useState(null);
  const [metrics,  setMetrics]  = useState(null);
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [ab, hist, pers, met] = await Promise.all([
          axios.get(`${API}/abtest/results`),
          axios.get(`${API}/metrics/history`),
          axios.get(`${API}/metrics/personalization`),
          axios.get(`${API}/metrics/summary`),
        ]);
        setAbData(ab.data);
        setHistory(hist.data.history.reverse());
        setPersona(pers.data);
        setMetrics(met.data);
      } catch (err) {
        console.error(err);
      }
      setLoading(false);
    };
    fetchAll();
  }, []);

  if (loading) return (
    <div style={{
      display:        "flex",
      justifyContent: "center",
      alignItems:     "center",
      height:         "100vh",
      color:          "#6b7280",
    }}>
      Loading analytics...
    </div>
  );

  const pieData = persona ? [
    { name: "New User",      value: persona.buckets.new_user },
    { name: "Learning",      value: persona.buckets.learning },
    { name: "Personalized",  value: persona.buckets.personalized },
    { name: "Highly",        value: persona.buckets.highly },
  ] : [];

  const ctrData = abData ? [
    {
      group: "Control\n(Random)",
      ctr:   Math.round(abData.control.ctr   * 100),
      fill:  "#6b7280",
    },
    {
      group: "Treatment\n(Personalized)",
      ctr:   Math.round(abData.treatment.ctr * 100),
      fill:  "#8b5cf6",
    },
  ] : [];

  return (
    <div style={{
      maxWidth: "900px",
      margin:   "0 auto",
      padding:  "24px 16px",
    }}>

      {/* header */}
      <div style={{ marginBottom: "28px" }}>
        <h1 style={{
          fontSize:   "22px",
          fontWeight: "700",
          background: "linear-gradient(135deg, #8b5cf6, #ec4899)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor:  "transparent",
        }}>
          Analytics Dashboard
        </h1>
        <p style={{ fontSize: "12px", color: "#6b7280" }}>
          Real-time A/B test results + personalization metrics
        </p>
      </div>

      {/* stat cards */}
      {abData && (
        <div style={{
          display:             "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap:                 "12px",
          marginBottom:        "24px",
        }}>
          {[
            {
              label: "Control CTR",
              value: `${Math.round(abData.control.ctr * 100)}%`,
              sub:   `${abData.control.users} users`,
              color: "#6b7280",
            },
            {
              label: "Treatment CTR",
              value: `${Math.round(abData.treatment.ctr * 100)}%`,
              sub:   `${abData.treatment.users} users`,
              color: "#8b5cf6",
            },
            {
              label: "Lift",
              value: `+${abData.stats.lift_percent}%`,
              sub:   "personalization gain",
              color: "#10b981",
            },
            {
              label: "P-Value",
              value: abData.stats.p_value < 0.001
                     ? "<0.001" : abData.stats.p_value,
              sub:   abData.stats.is_significant
                     ? "✓ Significant" : "Not significant",
              color: abData.stats.is_significant
                     ? "#10b981" : "#f59e0b",
            },
          ].map((card, i) => (
            <div key={i} style={{
              background:   "#1a1a1a",
              border:       `1px solid ${card.color}30`,
              borderRadius: "14px",
              padding:      "16px",
            }}>
              <p style={{
                fontSize: "11px", color: "#6b7280",
                marginBottom: "6px",
              }}>
                {card.label}
              </p>
              <p style={{
                fontSize:   "24px",
                fontWeight: "700",
                color:      card.color,
              }}>
                {card.value}
              </p>
              <p style={{ fontSize: "11px", color: "#4b5563" }}>
                {card.sub}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* verdict banner */}
      {abData?.stats?.verdict && (
        <div style={{
          background:   abData.stats.is_significant
                        ? "#064e3b" : "#1a1a1a",
          border:       `1px solid ${abData.stats.is_significant
                        ? "#10b981" : "#2a2a2a"}`,
          borderRadius: "12px",
          padding:      "14px 18px",
          marginBottom: "24px",
          fontSize:     "14px",
          fontWeight:   "600",
          color:        abData.stats.is_significant
                        ? "#10b981" : "#6b7280",
        }}>
          {abData.stats.verdict}
          {abData.stats.is_significant && (
            <span style={{
              fontSize:   "12px",
              fontWeight: "400",
              color:      "#6b7280",
              marginLeft: "12px",
            }}>
              95% confidence interval: {abData.stats.ci_lower?.toFixed(3)}
              {" — "}
              {abData.stats.ci_upper?.toFixed(3)}
            </span>
          )}
        </div>
      )}

      {/* charts row */}
      <div style={{
        display:             "grid",
        gridTemplateColumns: "1fr 1fr",
        gap:                 "16px",
        marginBottom:        "24px",
      }}>

        {/* CTR comparison */}
        <div style={{
          background:   "#1a1a1a",
          border:       "1px solid #2a2a2a",
          borderRadius: "14px",
          padding:      "20px",
        }}>
          <p style={{
            fontSize:     "13px",
            fontWeight:   "600",
            color:        "#f1f5f9",
            marginBottom: "16px",
          }}>
            CTR: Control vs Treatment
          </p>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={ctrData}>
              <XAxis dataKey="group"
                tick={{ fill: "#6b7280", fontSize: 11 }}
                axisLine={false} tickLine={false}
              />
              <YAxis
                tick={{ fill: "#6b7280", fontSize: 11 }}
                axisLine={false} tickLine={false}
                unit="%"
              />
              <Tooltip
                contentStyle={{
                  background: "#242424",
                  border:     "1px solid #2a2a2a",
                  borderRadius:"8px",
                  color:      "#f1f5f9",
                }}
                formatter={v => [`${v}%`, "CTR"]}
              />
              <Bar dataKey="ctr" radius={[6, 6, 0, 0]}>
                {ctrData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* personalization distribution */}
        <div style={{
          background:   "#1a1a1a",
          border:       "1px solid #2a2a2a",
          borderRadius: "14px",
          padding:      "20px",
        }}>
          <p style={{
            fontSize:     "13px",
            fontWeight:   "600",
            color:        "#f1f5f9",
            marginBottom: "16px",
          }}>
            Personalization Distribution
          </p>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%" cy="50%"
                innerRadius={40}
                outerRadius={65}
                paddingAngle={3}
                dataKey="value"
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background:   "#242424",
                  border:       "1px solid #2a2a2a",
                  borderRadius: "8px",
                  color:        "#f1f5f9",
                }}
              />
              <Legend
                iconSize={8}
                wrapperStyle={{ fontSize: "11px", color: "#6b7280" }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* daily events chart */}
      {history.length > 0 && (
        <div style={{
          background:   "#1a1a1a",
          border:       "1px solid #2a2a2a",
          borderRadius: "14px",
          padding:      "20px",
          marginBottom: "24px",
        }}>
          <p style={{
            fontSize:     "13px",
            fontWeight:   "600",
            color:        "#f1f5f9",
            marginBottom: "16px",
          }}>
            Daily Events (last 30 days)
          </p>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={history}>
              <XAxis dataKey="date"
                tick={{ fill: "#6b7280", fontSize: 10 }}
                axisLine={false} tickLine={false}
                tickFormatter={d => d?.slice(5)}
              />
              <YAxis
                tick={{ fill: "#6b7280", fontSize: 11 }}
                axisLine={false} tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  background:   "#242424",
                  border:       "1px solid #2a2a2a",
                  borderRadius: "8px",
                  color:        "#f1f5f9",
                }}
              />
              <Line type="monotone" dataKey="events"
                stroke="#8b5cf6" strokeWidth={2}
                dot={false} name="Events"
              />
              <Line type="monotone" dataKey="clicks"
                stroke="#10b981" strokeWidth={2}
                dot={false} name="Clicks"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* model benchmark */}
      <div style={{
        background:   "#1a1a1a",
        border:       "1px solid #2a2a2a",
        borderRadius: "14px",
        padding:      "20px",
      }}>
        <p style={{
          fontSize:     "13px",
          fontWeight:   "600",
          color:        "#f1f5f9",
          marginBottom: "16px",
        }}>
          Model Benchmark
        </p>
        <div style={{
          display:             "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap:                 "12px",
        }}>
          {[
            { label: "Recall@1", value: "15.6%",
              baseline: "5%",   color: "#8b5cf6" },
            { label: "Recall@3", value: "47.9%",
              baseline: "15%",  color: "#8b5cf6" },
            { label: "Recall@5", value: "62.8%",
              baseline: "25%",  color: "#8b5cf6" },
          ].map((m, i) => (
            <div key={i} style={{
              background:   "#242424",
              borderRadius: "10px",
              padding:      "14px",
              textAlign:    "center",
            }}>
              <p style={{
                fontSize: "11px", color: "#6b7280",
                marginBottom: "6px",
              }}>
                {m.label}
              </p>
              <p style={{
                fontSize:   "22px",
                fontWeight: "700",
                color:      m.color,
              }}>
                {m.value}
              </p>
              <p style={{ fontSize: "10px", color: "#4b5563" }}>
                vs {m.baseline} random
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}