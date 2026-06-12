# FinNudge — Real-Time Financial Personalization Engine

> A behavioral personalization engine that learns individual user 
> financial patterns and serves hyper-relevant product nudges in 
> real time — built on a two-tower retrieval model, online learning, 
> and an A/B testing framework with statistically validated results.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Live Demo

| Service | URL |
|---------|-----|
| Frontend (Vercel) | [finnudge.vercel.app](https://finnudge.vercel.app) |
| Backend API (Render) | [finnudge-backend.onrender.com](https://finnudge-backend.onrender.com) |
| API Docs (Swagger) | [/docs](https://finnudge-backend.onrender.com/docs) |

> Note: Render free tier spins down after 15 minutes of 
> inactivity. First request may take 30-60 seconds to cold start.

---

## The Problem

Fintech apps show the same nudges to all users. A stock investor 
sees bill payment reminders. A borrower sees SIP suggestions. 
Generic recommendations waste user attention and reduce engagement.

**The solution:** Learn each user's financial behavior pattern and 
serve the most relevant nudge at the right moment — updating in 
real time as users interact.

---

## Results

| Metric | Value | Baseline | Improvement |
|--------|-------|----------|-------------|
| Recall@1 | 15.6% | 5% (random) | 3.1x |
| Recall@3 | 47.9% | 15% (random) | 3.2x |
| Recall@5 | 62.8% | 25% (random) | 2.5x |
| Treatment CTR | 42.2% | 18.3% (control) | +130% lift |
| P-value | <0.0001 | — | Statistically significant |
| Hot path latency | 5ms | — | Production ready |

---

## Architecture
┌─────────────────────────────────────────────────────┐

│                   DATA PIPELINE                      │

│  Synthetic generation → Feature engineering →        │

│  45-dim user vectors + 16-dim nudge vectors         │

└──────────────────────┬──────────────────────────────┘

↓

┌─────────────────────────────────────────────────────┐

│                  ML PIPELINE                         │

│  Two-Tower PyTorch Model (contrastive loss)         │

│  UserTower: 45→128→64  NudgeTower: 16→64→64        │

│  FAISS IndexFlatIP (64-dim embeddings)              │

└──────────────────────┬──────────────────────────────┘

↓

┌─────────────────────────────────────────────────────┐

│            PERSONALIZATION LAYERS                    │

│  MMR Diversity Reranking (λ=0.7)                   │

│  Session Context Awareness (6 action types)         │

│  Nudge Fatigue Prevention (impression capping)      │

│  Online Learning (real-time embedding updates)      │

│  Personalization Score (0-100 drift metric)         │

└──────────────────────┬──────────────────────────────┘

↓

┌─────────────────────────────────────────────────────┐

│                  API LAYER                           │

│  FastAPI + Pydantic + SQLite event logging          │

│  A/B Testing (Welch t-test, 95% CI, lift)          │

└──────────────────────┬──────────────────────────────┘

↓

┌─────────────────────────────────────────────────────┐

│                  FRONTEND                            │

│  React + Vite · Recharts · Dark theme              │

│  Demo UI + Analytics Dashboard                      │

└─────────────────────────────────────────────────────┘

---

## ML System Deep Dive

### Two-Tower Model

The core architecture uses two separate neural networks that learn 
to map users and nudges into a shared 64-dimensional embedding 
space where relevant pairs are geometrically close.
User Behavioral Vector (45-dim)    Nudge Feature Vector (16-dim)

↓                                    ↓

UserTower                           NudgeTower

Linear(45→128)                      Linear(16→64)

BatchNorm + ReLU                    BatchNorm + ReLU

Dropout(0.2)                        Dropout(0.1)

Linear(128→64)                      Linear(64→64)

L2 Normalize                        L2 Normalize

↓                                    ↓

64-dim embedding              64-dim embedding

└──────────┬───────────────┘

dot product

↓

similarity score

**Why two towers:** Pre-compute all nudge embeddings once. At 
query time, one user forward pass + FAISS search = <10ms. 
Scales to millions of users and thousands of nudges without 
architectural changes.

### Feature Engineering

**User features (45 dimensions):**
- 15 behavioral dims: avg amount, transaction count, 
  time patterns, amount distribution, age, salary day
- 20 category dims: transaction ratio per category 
  (stocks, SIP, food, EMI, etc.)
- 10 city dims: one-hot encoding for 10 Indian cities

**Nudge features (16 dimensions):**
- 5 archetype dims: one-hot target archetype
- 9 category dims: one-hot nudge category
- 2 score dims: urgency + value

### Contrastive Loss

```python
pos_loss = labels * clamp(1 - similarity, min=0)
neg_loss = (1-labels) * clamp(similarity - margin, min=0)
loss = (pos_loss + neg_loss).mean()
```

Pulls positive pairs (user + relevant nudge) closer together 
in embedding space. Pushes negative pairs beyond margin=0.3. 
Teaches the model similarity structure rather than 
classification.

### Recall@K Evaluation

Correct metric for retrieval systems. For each user, rank all 
20 nudges by similarity score, check if relevant nudges appear 
in top K.
Recall@3: 47.9% — 3.2x improvement over 15% random baseline

### Online Learning

Real-time embedding updates without full model retraining:

```python
# user clicks investment nudge
delta = nudge_features[:15] - user_vector[:15]
user_vector[:15] += learning_rate * delta
# user's behavioral embedding shifts toward investment profile
```

One click shifts embedding by ~0.02 magnitude. After sustained 
interaction patterns, embeddings meaningfully drift toward 
new behavioral territory.

### MMR Diversity Reranking

Prevents filter bubbles using Maximal Marginal Relevance:
MMR = λ × relevance - (1-λ) × max_similarity_to_selected

λ = 0.7 (70% relevance, 30% diversity)

Ensures top-3 recommendations span at least 2 categories.

### A/B Testing Framework

- Hash-based user bucketing (consistent assignment)
- Welch's t-test (handles unequal variance)
- 95% confidence interval calculation
- Statistical significance at p < 0.05

**Results: Treatment CTR 42.2% vs Control 18.3%**  
**Lift: +130% | P-value: <0.0001 | Significant: True**

---

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| PyTorch 2.3 | Two-tower neural network |
| FAISS-CPU | Sub-10ms vector similarity search |
| River | Online learning framework |
| FastAPI | REST API framework |
| SQLite | Event logging + user data |
| Scikit-learn | Feature preprocessing |
| Scipy | Statistical testing (Welch t-test) |
| NumPy / Pandas | Data processing |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 + Vite | UI framework + build tool |
| Recharts | Analytics charts |
| Lucide React | Icon system |
| Axios | HTTP client |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Render | Backend deployment (free tier) |
| Vercel | Frontend deployment (free tier) |
| Docker | Containerization |
| GitHub Actions | CI/CD |

---

## Project Structure
finnudge/

├── backend/

│   ├── app/

│   │   ├── api/

│   │   ├── core/

│   │   │   └── config.py

│   │   ├── models/

│   │   │   └── two_tower.py        ← neural network

│   │   ├── services/

│   │   │   ├── recommender.py      ← FAISS retrieval

│   │   │   ├── online_learner.py   ← real-time updates

│   │   │   └── ab_testing.py       ← experiment framework

│   │   └── main.py                 ← FastAPI app

│   ├── requirements.txt

│   └── requirements-render.txt

├── frontend/

│   ├── src/

│   │   ├── components/

│   │   │   ├── NudgeCard.jsx

│   │   │   └── PersonalizationBadge.jsx

│   │   ├── pages/

│   │   │   ├── Demo.jsx

│   │   │   └── Analytics.jsx

│   │   ├── App.jsx

│   │   └── index.css

│   └── vercel.json

├── scripts/

│   ├── generate_users.py           ← synthetic data

│   ├── feature_engineering.py      ← 45-dim vectors

│   ├── build_nudge_features.py     ← 16-dim vectors

│   ├── train_model.py              ← contrastive training

│   ├── build_faiss_index.py        ← vector index

│   └── simulate_interactions.py    ← A/B test data

├── data/

│   └── processed/

│       ├── users.json

│       ├── features.json

│       ├── nudge_features.json

│       ├── nudge_embeddings.json

│       ├── nudge.index             ← FAISS index

│       └── finnudge.db             ← SQLite

├── notebooks/

│   └── 01_eda.ipynb

├── Dockerfile

├── docker-compose.yml

├── render.yaml

└── README.md

---

## API Reference

### POST /recommend
Get personalized nudges for a user.

**Request:**
```json
{
  "user_id": "user_0001",
  "top_k": 3,
  "current_action": "received_salary"
}
```

**Response:**
```json
{
  "user_id": "user_0001",
  "archetype": "investor",
  "nudges": [
    {
      "rank": 1,
      "nudge_id": "n20",
      "title": "Save tax — invest in ELSS before March",
      "category": "investment",
      "score": 1.0,
      "explanation": "Based on 92 investment transactions in last 6 months",
      "context_boosted": true,
      "boost_amount": 0.15
    }
  ],
  "latency_ms": 5.29,
  "source": "two_tower",
  "personalization": {
    "score": 56,
    "label": "well personalized",
    "drift": 0.085432
  },
  "ab_group": "treatment"
}
```

### POST /feedback
Log user interaction and trigger online learning.

**Request:**
```json
{
  "user_id": "user_0001",
  "nudge_id": "n20",
  "action": "click",
  "ab_group": "treatment"
}
```

### GET /abtest/results
Get A/B test statistics with statistical significance.

### GET /metrics/summary
Get overall system metrics including model benchmark.

### GET /metrics/personalization
Get personalization score distribution across users.

### GET /user/{user_id}/personalization
Get personalization score for a specific user.

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Setup

```bash
# clone
git clone https://github.com/praveenkumar993/finnudge.git
cd finnudge

# backend
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r backend/requirements.txt

# generate data + train model
python scripts/generate_users.py
python scripts/feature_engineering.py
python scripts/build_nudge_features.py
python scripts/train_model.py
python scripts/build_faiss_index.py
python scripts/simulate_interactions.py

# start backend
uvicorn backend.app.main:app --reload --port 8000

# frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Key Design Decisions

**Why synthetic data:**
Real fintech transaction data is private and RBI-regulated. 
Synthetic generation with domain-realistic archetypes is 
standard practice — used by Google, Meta, and Microsoft in 
published research.

**Why two-tower over collaborative filtering:**
Collaborative filtering requires user-user or item-item 
similarity matrices — O(n²) memory. Two-tower is O(n) and 
supports new users via cold-start fallback. Scales to 
production without architectural changes.

**Why contrastive loss over cross-entropy:**
Cross-entropy optimizes classification. Contrastive loss 
optimizes similarity structure — the correct objective for 
retrieval. Enables FAISS-based nearest neighbor search which 
isn't possible with classification outputs.

**Why Recall@K over accuracy:**
Binary accuracy conflates "did the model rank this pair 
correctly" with "did the model surface relevant items for 
this user." Recall@K measures the actual retrieval objective — 
are relevant nudges in the top positions a user sees.

**Why Welch's t-test:**
Control and treatment groups have different CTR distributions — 
unequal variance. Student's t-test assumes equal variance and 
would give incorrect p-values. Welch's handles unequal variance 
correctly.

---

## Planned Enhancements

- JWT authentication for multi-user production deployment
- WhatsApp notification integration for high-urgency nudges
- ONNX export for faster CPU inference
- Expansion to 500+ nudges to fully utilize FAISS ANN algorithms
- Kafka integration for real-time event streaming at scale
- MLflow experiment tracking for model versioning

---

## What This Project Demonstrates

**ML Engineering depth:**
Two-tower architecture, contrastive loss, FAISS retrieval, 
online learning, Recall@K evaluation — the complete stack 
used in production recommendation systems at YouTube, 
Airbnb, and Pinterest.

**Production thinking:**
A/B testing with statistical significance, personalization 
scoring, nudge fatigue prevention, MMR diversity — 
addressing the same challenges production systems face.

**Full-stack capability:**
From PyTorch model training to FastAPI backend to React 
dashboard — end-to-end ownership of the entire system.

---

## Author

**Praveen Kumar**  
GenAI | AI/ML | Data Science