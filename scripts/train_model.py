import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app.models.two_tower import TwoTowerModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

USER_FEATURES_PATH  = "data/processed/features.json"
NUDGE_FEATURES_PATH = "data/processed/nudge_features.json"
MODEL_OUTPUT_PATH   = "data/processed/two_tower.pt"
EMBEDDINGS_PATH     = "data/processed/nudge_embeddings.json"


class FinNudgeDataset(Dataset):
    def __init__(self, users, nudges, noise=0.0):
        self.pairs  = []
        self.noise  = noise
        nudge_ids   = [n["nudge_id"] for n in nudges]
        nudge_map   = {n["nudge_id"]: n for n in nudges}

        for user in users:
            pos_ids = set(user["relevant_nudges"])
            neg_ids = [nid for nid in nudge_ids if nid not in pos_ids]

            for nid in pos_ids:
                if nid in nudge_map:
                    self.pairs.append((
                        user["vector"],
                        nudge_map[nid]["vector"],
                        1.0
                    ))

            neg_sample = random.sample(neg_ids, min(len(pos_ids), len(neg_ids)))
            for nid in neg_sample:
                self.pairs.append((
                    user["vector"],
                    nudge_map[nid]["vector"],
                    0.0
                ))

        random.shuffle(self.pairs)
        pos = sum(1 for _, _, l in self.pairs if l == 1.0)
        neg = sum(1 for _, _, l in self.pairs if l == 0.0)
        print(f"  dataset: {len(self.pairs)} pairs ({pos} pos, {neg} neg)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_vec, nudge_vec, label = self.pairs[idx]
        u = torch.tensor(user_vec,  dtype=torch.float32)
        n = torch.tensor(nudge_vec, dtype=torch.float32)
        if self.noise > 0:
            u = u + torch.randn_like(u) * self.noise
        return u, n, torch.tensor(label, dtype=torch.float32)


class ContrastiveLoss(nn.Module):
    """
    margin=0.3: stable default
    pos_loss: push positive pairs to similarity > 1-margin
    neg_loss: push negative pairs to similarity < margin
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, similarity, labels):
        pos_loss = labels       * torch.clamp(1 - similarity, min=0)
        neg_loss = (1 - labels) * torch.clamp(similarity - self.margin, min=0)
        return (pos_loss + neg_loss).mean()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for user_vec, nudge_vec, labels in loader:
        user_vec  = user_vec.to(device)
        nudge_vec = nudge_vec.to(device)
        labels    = labels.to(device)
        optimizer.zero_grad()
        similarity, _, _ = model(user_vec, nudge_vec)
        loss = criterion(similarity, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, users_data, nudges, device, k=3):
    """
    Recall@K evaluation — correct metric for retrieval models
    For each user: rank all 20 nudges by similarity
    Check if relevant nudges appear in top K
    """
    model.eval()
    recall_scores = []

    nudge_map = {n["nudge_id"]: n for n in nudges}
    nudge_ids = [n["nudge_id"] for n in nudges]

    # precompute all nudge embeddings
    nudge_vecs = torch.tensor(
        [nudge_map[nid]["vector"] for nid in nudge_ids],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        nudge_embs = model.get_nudge_embedding(nudge_vecs)

        for user in users_data:
            user_vec = torch.tensor(
                [user["vector"]], dtype=torch.float32
            ).to(device)
            user_emb = model.get_user_embedding(user_vec)

            # similarity to all 20 nudges
            sims = (user_emb * nudge_embs).sum(dim=-1)
            ranked_indices = sims.argsort(descending=True).cpu().tolist()
            top_k_ids = set([nudge_ids[i] for i in ranked_indices[:k]])

            relevant  = set(user["relevant_nudges"])
            hits      = len(top_k_ids & relevant)
            recall    = hits / max(len(relevant), 1)
            recall_scores.append(recall)

    return np.mean(recall_scores)


def save_nudge_embeddings(model, nudges, device):
    model.eval()
    records = []
    with torch.no_grad():
        for n in nudges:
            vec = torch.tensor([n["vector"]], dtype=torch.float32).to(device)
            emb = model.get_nudge_embedding(vec)
            records.append({
                "nudge_id":  n["nudge_id"],
                "title":     n["title"],
                "archetype": n["archetype"],
                "category":  n["category"],
                "embedding": emb.cpu().numpy().tolist()[0],
            })
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"saved {len(records)} nudge embeddings → {EMBEDDINGS_PATH}")


def main():
    device = torch.device("cpu")
    print(f"device: {device}")

    with open(USER_FEATURES_PATH)  as f: users  = json.load(f)
    with open(NUDGE_FEATURES_PATH) as f: nudges = json.load(f)
    print(f"loaded {len(users)} users, {len(nudges)} nudges")

    random.shuffle(users)
    split       = int(len(users) * 0.85)
    train_users = users[:split]
    val_users   = users[split:]

    print("train dataset:")
    train_dataset = FinNudgeDataset(train_users, nudges, noise=0.03)
    print("val dataset:")
    val_dataset   = FinNudgeDataset(val_users, nudges, noise=0.0)
    # keep val_users raw for Recall@K evaluation


    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)

    model     = TwoTowerModel(user_input_dim=45, nudge_input_dim=16, embedding_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = ContrastiveLoss(margin=0.3)

    print(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n── training ──")
    best_recall = 0
    epochs      = 30

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        recall_at3 = evaluate(model, val_users, nudges, device, k=3)
        scheduler.step()

        marker = ""
        if recall_at3 > best_recall:
            best_recall = recall_at3
            torch.save({
                "model_state": model.state_dict(),
                "recall_at_3": recall_at3,
                "epoch":       epoch,
            }, MODEL_OUTPUT_PATH)
            marker = " ← saved"

        print(f"  epoch {epoch:02d}/{epochs} | "
              f"loss: {train_loss:.4f} | "
              f"Recall@3: {recall_at3:.1%}{marker}")

    checkpoint = torch.load(MODEL_OUTPUT_PATH)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nbest Recall@3 : {best_recall:.1%}")

    save_nudge_embeddings(model, nudges, device)
    print("\nDay 3 training complete.")


if __name__ == "__main__":
    main()