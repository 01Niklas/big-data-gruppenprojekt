from typing import Optional, Sequence, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from recommender import Recommender


class RatingsDataset(Dataset):
    # PyTorch dataset that returns (user_idx, item_idx, rating) tuples
    def __init__(self, df: pd.DataFrame):
        # cast once to tensors that avoids conversions inside the training loop
        self.u = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.r = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]


class HybridMF(nn.Module):
    # Matrix‑factorisation with bias terms + linear projection of item features
    def __init__(self, num_users: int, num_items: int, d: int, item_features: torch.Tensor, dropout: float = 0.15):
        super().__init__()

        # user / item embeddings (latent factors)
        self.P = nn.Embedding(num_users, d)
        self.Q = nn.Embedding(num_items, d)

        # bias embeddings
        self.bu = nn.Embedding(num_users, 1)
        self.bi = nn.Embedding(num_items, 1)
        self.mu = nn.Parameter(torch.zeros(1))  # global mean

        # linear projection that maps item side‑features into the same latent space
        self.F = nn.Linear(item_features.shape[1], d, bias=False)
        self.register_buffer("item_features",
                             item_features)  # moved to GPU/CPU automatically to improve performance based on system

        # dropout for a bit of regularisation
        self.drop = nn.Dropout(dropout)

        # lightweight weight initialisation
        nn.init.normal_(self.P.weight, std=0.05)
        nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.xavier_uniform_(self.F.weight)

    def forward(self, u, i):
        # build item representation: latent factors + projected features
        q = self.Q(i) + self.F(self.item_features[i])
        q = self.drop(q)

        # final prediction for each (u,i) pair
        return (self.P(u) * q).sum(-1) + self.mu + self.bu(u).squeeze() + self.bi(
            i).squeeze()  # product of latent vectors + global bias + user bias + item bias


class DeepLearningRecommender(Recommender):
    # Pre‑processing, training loop, evaluation & prediction are encapsulated here.
    def __init__(
            self,
            ratings: pd.DataFrame,
            item_profile: pd.DataFrame,
            numeric_cols: Optional[Sequence[str]] = ("runtime", "budget", "revenue"),
            val_ratio: float = 0.2,
            embedding_dim: int = 64,
            batch_size: int = 1024,
            epochs: int = 60,
            lr: float = 1e-3,
            weight_decay: float = 3e-5,
            dropout_p: float = 0.20,
            early_stopping_rounds: int = 10,
            device: Optional[str] = None,
            seed: int = 42,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.ratings_raw = ratings.copy()
        self.item_profile_raw = item_profile.copy()
        self.numeric_cols = numeric_cols
        self.val_ratio = val_ratio

        # complete preparation + training pipeline
        self._preprocess_data()
        self._build_model()
        self.fit()  # auto‑train so that external testers don’t need to call fit()

    def _preprocess_data(self):
        # cast ids to str to avoid categorical type issues
        self.ratings_raw[["user_ID", "item_ID"]] = self.ratings_raw[["user_ID", "item_ID"]].astype(str)

        # dense id mapping
        self.user2idx = {u: i for i, u in enumerate(self.ratings_raw["user_ID"].unique())}
        self.item2idx = {m: j for j, m in enumerate(self.ratings_raw["item_ID"].unique())}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {j: m for m, j in self.item2idx.items()}

        ratings = self.ratings_raw.copy()
        ratings["user_idx"] = ratings["user_ID"].map(self.user2idx)
        ratings["item_idx"] = ratings["item_ID"].map(self.item2idx)

        # Feature matrix (one‑hot genres + scaled numeric cols)
        ip = self.item_profile_raw[self.item_profile_raw["item_ID"].isin(self.item2idx)].copy()
        ip["item_idx"] = ip["item_ID"].map(self.item2idx)
        ip.sort_values("item_idx", inplace=True)
        feat_df = ip.filter(regex="^Genre_")  # already one‑hot 0/1 columns

        # numeric columns (if present)
        if self.numeric_cols:
            present = [c for c in self.numeric_cols if c in ip.columns]
            if missing := set(self.numeric_cols) - set(present):
                logger.warning(f"Skipped numeric cols: {missing}")
            if present and not ip.empty:
                feat_df[present] = StandardScaler().fit_transform(ip[present].fillna(0))
            elif present:
                # no rows –> create zero columns to keep matrix shape consistent
                for col in present:
                    feat_df[col] = 0

        # if no features at all – fallback to a single zero column
        self.feat_mat = torch.tensor(feat_df.values if not feat_df.empty else np.zeros((len(self.item2idx), 1)),
                                     dtype=torch.float32)

        # optional but "why-not"...  global mean used for cold‑start cases
        self.global_mean = ratings["rating"].mean()

        # pivot for optional Pearson similarity (required by base class... for testing we do not remove this)
        self.data = ratings.pivot(index="user_ID", columns="item_ID", values="rating")

        # create loaders
        train_df, val_df = train_test_split(ratings, test_size=self.val_ratio, random_state=self.seed)
        self.train_loader = DataLoader(RatingsDataset(train_df), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(RatingsDataset(val_df), batch_size=self.batch_size)

    # model construction
    def _build_model(self):
        self.model = HybridMF(
            num_users=len(self.user2idx),
            num_items=len(self.item2idx),
            d=self.embedding_dim,
            item_features=self.feat_mat,
            dropout=self.dropout_p,
        ).to(self.device)

    def fit(self):
        # Adam optimiser... suited for sparse embeddings
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # scheduler lowers LR (Learn-Rate) by ×0.5 if val‑MAE plateaus for 3 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

        best_mae, epochs_no_imp = float("inf"), 0  # track early‑stopping progress

        for ep in range(1, self.epochs + 1):
            self.model.train()
            # iterate over mini‑batches (u,i,r) from DataLoader
            for u, i, r in self.train_loader:
                # move to device once per batch
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)

                opt.zero_grad()  # reset gradients
                pred = self.model(u, i)  # forward pass

                # Smooth L1 ≈ MAE for |err|>β, MSE for |err|<β – stable & robust
                loss = nn.functional.smooth_l1_loss(pred, r, beta=1.0)
                loss.backward()  # back‑prop
                opt.step()  # Adam update

            # validation after each epoch
            val_mae = self._evaluate_loader(self.val_loader)
            logger.info(f"ep{ep:02d} • val MAE {val_mae:.4f}")
            scheduler.step(val_mae)  # maybe reduce LR

            # early‑stopping logic
            if val_mae + 1e-4 < best_mae:  # question: significant improvement? ... yes / no?
                best_mae, epochs_no_imp = val_mae, 0
                torch.save(self.model.state_dict(), "hybrid_best.pt")  # checkpoint (save in file)
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= self.early_stopping_rounds:
                    break  # stop training...

        # restore best weights before evaluation
        self.model.load_state_dict(torch.load("hybrid_best.pt"))

    # helper to evaluate loaders
    def _evaluate_loader(self, loader):
        # computes Mean Absolute Error over a given DataLoader
        self.model.eval();
        err, n = 0.0, 0
        with torch.no_grad():
            for u, i, r in loader:
                preds = self.model(u.to(self.device), i.to(self.device)).cpu()
                err += torch.abs(preds - r).sum().item();
                n += len(r)
        return err / n

    # default prediction method from base class
    def predict(
            self,
            user_id: str,
            item_id: str,
            similarity: Optional[Literal['cosine', 'pearson']] = 'cosine',
            calculation_variety: Optional[Literal['weighted', 'unweighted']] = 'weighted',
            k: Optional[int] = 3,
            second_k_value: Optional[int] = None,
    ) -> float:
        self._prepare_information(user_id, item_id, k, similarity,
                                  calculation_variety)  # ignore that this can be none... just a "simple" test recommender here

        # check if the user or item is not known
        if user_id not in self.user2idx and item_id not in self.item2idx:
            return float(self.global_mean)

        # case 1: only user is unknown so we take the item-bias
        if user_id not in self.user2idx:
            idx = self.item2idx.get(item_id)
            if idx is None:
                return float(self.global_mean)
            item_bias = self.model.bi.weight[idx].item()
            return float(np.clip(self.global_mean + item_bias, 0.5, 5.0))

        # case 2: only item is unknown so we take the user-bias
        if item_id not in self.item2idx:
            u_idx = self.user2idx[user_id]
            user_bias = self.model.bu.weight[u_idx].item()
            return float(np.clip(self.global_mean + user_bias, 0.5, 5.0))

        # case 3 - both unnown (default case)
        u = torch.tensor([self.user2idx[user_id]], device=self.device)
        i = torch.tensor([self.item2idx[item_id]], device=self.device)
        self.model.eval()
        with torch.no_grad():
            score = self.model(u, i).item()
        return float(np.clip(score, 0.5, 5.0))