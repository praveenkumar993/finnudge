import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    """
    Takes a 45-dim user behavioral vector
    Outputs a 64-dim embedding
    """
    def __init__(self, input_dim=45, embedding_dim=64):
        super(UserTower, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.network(x), dim=-1)


class NudgeTower(nn.Module):
    """
    Takes a 16-dim nudge feature vector
    Outputs a 64-dim embedding
    """
    def __init__(self, input_dim=16, embedding_dim=64):
        super(NudgeTower, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.network(x), dim=-1)


class TwoTowerModel(nn.Module):
    """
    Full two-tower model combining UserTower + NudgeTower
    Similarity = dot product of normalized embeddings
    """
    def __init__(self, user_input_dim=45, nudge_input_dim=16, embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        self.user_tower  = UserTower(user_input_dim, embedding_dim)
        self.nudge_tower = NudgeTower(nudge_input_dim, embedding_dim)

    def forward(self, user_vec, nudge_vec):
        user_emb  = self.user_tower(user_vec)
        nudge_emb = self.nudge_tower(nudge_vec)
        similarity = (user_emb * nudge_emb).sum(dim=-1)
        return similarity, user_emb, nudge_emb

    def get_user_embedding(self, user_vec):
        return self.user_tower(user_vec)

    def get_nudge_embedding(self, nudge_vec):
        return self.nudge_tower(nudge_vec)