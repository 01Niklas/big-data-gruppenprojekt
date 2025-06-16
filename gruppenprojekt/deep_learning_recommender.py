from typing import Optional, Literal, Dict, Any

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from recommender import Recommender


# Dataset class to handle user-item-rating data
class RatingsDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        # Convert user, item, and rating columns to tensors
        self.u = torch.tensor(data["user_idx"].values, dtype=torch.long)
        self.i = torch.tensor(data["item_idx"].values, dtype=torch.long)
        self.r = torch.tensor(data["rating"].values, dtype=torch.float32)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.r)

    def __getitem__(self, idx):
        # Return a single sample (user, item, rating) by index
        return self.u[idx], self.i[idx], self.r[idx]


# hybrid matrix factorization model
class HybridMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, item_features, dropout: float = 0.15):
        super().__init__()
        # Embedding layers for users and items (Embedding-layer is one layer in the neral network (vectors))
        self.P = nn.Embedding(num_users, embedding_dim)
        self.Q = nn.Embedding(num_items, embedding_dim)

        # Bias terms for users and items (representate individual variances... e.g one user can generally give better ratings as default)
        self.bu = nn.Embedding(num_users, 1)
        self.bi = nn.Embedding(num_items, 1)

        # Global bias term (reprentate the average variance over the complete dataset)
        self.mu = nn.Parameter(torch.zeros(1))

        # Linear layer to project item features into the latent space (to combine them with the embeddings of the items)
        self.F = nn.Linear(item_features.shape[1], embedding_dim, bias=False)

        # Register item features as a buffer (non-trainable parameter)
        self.register_buffer("item_features", item_features)

        # Dropout layer for regularization
        self.drop = nn.Dropout(dropout)

        # Initialize weights for embeddings and linear layer
        nn.init.normal_(self.P.weight, std=0.05)
        nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.xavier_uniform_(self.F.weight)

    def forward(self, u, i):
        # Compute item latent factors by combining embeddings and projected features
        q = self.Q(i) + self.F(self.item_features[i])
        q = self.drop(q)

        # Compute the predicted rating
        return (self.P(u) * q).sum(-1) + self.mu + self.bu(u).squeeze() + self.bi(i).squeeze()


class DeepLearningRecommender:
    def __init__(
            self,
            trainingdata: pd.DataFrame,
            item_profile: pd.DataFrame,
            testdata: pd.DataFrame,
            embedding_dim=64,
            batch_size=1024,
            epochs=60,
            lr=1e-3,
            weight_decay=3e-5,
            dropout_p=0.2,
            early_stopping_rounds=10
    ) -> None:

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr  # learning rate
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.early_stopping_rounds = early_stopping_rounds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prepare data and build the model
        self.train_data, self.val_data, self.feat_mat, self.global_mean = self._prepare_data(trainingdata, item_profile, testdata)
        self.model = self._build_model()
        self.fit()

    def _prepare_data(self, trainingdata, item_profile, testdata):
        # Convert user and item ids to strings
        trainingdata[["user_ID", "item_ID"]] = trainingdata[["user_ID", "item_ID"]].astype(str)
        testdata[["user_ID", "item_ID"]] = testdata[["user_ID", "item_ID"]].astype(str)

        # Create mappings from user/item ids to indices
        self.user2idx = {u: i for i, u in enumerate(trainingdata["user_ID"].unique())}
        self.item2idx = {m: j for j, m in enumerate(trainingdata["item_ID"].unique())}

        # Map user and item ids to the indices in training and test data (both needed)
        for df in [trainingdata, testdata]:
            df["user_idx"] = df["user_ID"].map(self.user2idx)
            df["item_idx"] = df["item_ID"].map(self.item2idx)

        # filter and process item features
        ip = item_profile[item_profile["item_ID"].isin(self.item2idx)].copy()
        ip["item_idx"] = ip["item_ID"].map(self.item2idx)
        ip.sort_values("item_idx", inplace=True)
        feat_df = ip.filter(regex="^Genre_")

        # scale feature or set placeholder if no feature
        if not feat_df.empty:
            feat_df = StandardScaler().fit_transform(feat_df.fillna(0))
        else:
            feat_df = np.zeros((len(self.item2idx), 1))

        # convert features to a tensor (array in a dimension you need, vgl. Skalar (5), Vektor ([1,2,3]), ...)
        feat_mat = torch.tensor(feat_df, dtype=torch.float32)

        # Compute the global mean rating... optional but whats the ase when user or item i unknown ? (vgl. cold-start-szenario)
        global_mean = trainingdata["rating"].mean()

        # Create data loaders for training and validation (Dataloaders take the work of batching, shuffle or parallel loading to improve training)
        train_loader = DataLoader(RatingsDataset(trainingdata), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(RatingsDataset(testdata), batch_size=self.batch_size)

        return train_loader, val_loader, feat_mat, global_mean

    def _build_model(self):
        # Build the hybrid matrix factorization model
        return HybridMF(
            num_users=len(self.train_data.dataset.u.unique()),
            num_items=len(self.train_data.dataset.i.unique()),
            embedding_dim=self.embedding_dim,
            item_features=self.feat_mat,
            dropout=self.dropout_p,
        ).to(self.device)

    def fit(self):
        # Initialize optimizer and learning rate scheduler
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        best_mae, epochs_no_imp = float("inf"), 0
        best_model_state = None  # Variable to store the best model state

        # Training loop ... as we discussed in the lecture
        for ep in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0
            for u, i, r in self.train_data:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                opt.zero_grad()
                loss = nn.functional.smooth_l1_loss(self.model(u, i), r, beta=1.0)  # Feed-Forward
                loss.backward()  # Backpropagation
                opt.step()
                epoch_loss += loss.item()  # collect the loss-value


            # Calculate validation-Loss (we need the smallest MAE possible)
            val_mae = self.evaluate_loader(self.val_data)
            # Print average loss and MAE of each epoch
            logger.debug(f"Epoche {ep}\t| Training Loss: {epoch_loss / len(self.train_data)} \t| Validation MAE: {val_mae}")
            scheduler.step(val_mae)

            # Save the best model
            if val_mae < best_mae:
                best_mae, epochs_no_imp = val_mae, 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= self.early_stopping_rounds:
                    break

        # Load the best model state from memory
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)


    def evaluate_loader(self, loader):
        # Evaluate the model on a data loader
        self.model.eval()
        err, n = 0.0, 0
        with torch.no_grad():
            for u, i, r in loader:
                preds = self.model(u.to(self.device), i.to(self.device)).cpu()
                err += torch.abs(preds - r).sum().item()
                n += len(r)
        return err / n

    def predict(self, user_id, item_id):
        # Convert user and item IDs to indices
        user_idx = self.user2idx.get(user_id)
        item_idx = self.item2idx.get(item_id)

        # Handle cold-start cases
        if user_idx is None and item_idx is None:
            return float(self.global_mean)

        if user_idx is None:
            item_bias = self.model.bi.weight[item_idx].item()
            return float(np.clip(self.global_mean + item_bias, 0.0, 5.0)) # clip ensures that the value is between 0 and 5

        if item_idx is None:
            user_bias = self.model.bu.weight[user_idx].item()
            return float(np.clip(self.global_mean + user_bias, 0.0, 5.0)) # clip ensures that the value is between 0 and 5

        # Compute the predicted rating
        u = torch.tensor([user_idx], device=self.device)
        i = torch.tensor([item_idx], device=self.device)
        self.model.eval()
        with torch.no_grad():
            score = self.model(u, i).item()
        return float(np.clip(score, 0.0, 5.0)) # clip ensures that the value is between 0 and 5



class HyperparamOptimizedDeepLearningRecommender(Recommender):
    def __init__(self, testdata: pd.DataFrame, item_profile: pd.DataFrame, trainingdata: pd.DataFrame, include_hyperparam_check: Optional[bool] = False):
        super().__init__()
        self.testdata = testdata
        self.item_profile = item_profile
        self.trainingdata = trainingdata

        self.include_hyperparam_check= include_hyperparam_check
        self.recommender = None


    def _preprocess_data(self):
        # this will be done in the used recommender class
        pass

    def _objective(self, trial):
        embedding_dim = trial.suggest_int("embedding_dim", 32, 128, step=16)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        dropout_p = trial.suggest_uniform("dropout_p", 0.1, 0.5)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

        optimize_recommender = DeepLearningRecommender(
            trainingdata=self.trainingdata,
            item_profile=self.item_profile,
            testdata=self.testdata,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            lr=lr,
            dropout_p=dropout_p,
            weight_decay=weight_decay,
        )

        val_mae = optimize_recommender.evaluate_loader(optimize_recommender.val_data)
        return val_mae

    def _build_recommender(self, params: dict) -> None:
        self.recommender = DeepLearningRecommender(
            trainingdata=self.trainingdata,
            item_profile=self.item_profile,
            testdata=self.testdata,
            embedding_dim=params["embedding_dim"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            dropout_p=params["dropout_p"],
            weight_decay=params["weight_decay"],
        )

    def _find_out_best_params(self) -> Dict[str, Any]:
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=50)

        print("Beste Parameter:", study.best_params)

        return study.best_params


    def predict(
            self,
            user_id: str,
            item_id: str,
            similarity: Optional[Literal['cosine', 'pearson']] = 'cosine',
            calculation_variety: Optional[Literal['weighted', 'unweighted']] = 'weighted',
            k: Optional[int] = 3,
            second_k_value: Optional[int] = None,
    ) -> float:

        if self.include_hyperparam_check:
            best_params = self._find_out_best_params()
        else:
            # values came from our test with the hyperparam check
            best_params = {
                "lr": 0.00483293357554159,
                "embedding_dim": 32,
                "dropout_p": 0.24090306736140638,
                "weight_decay": 0.00022232253823222672,
                "batch_size": 64,
            }


        if self.recommender is None:
            self._build_recommender(best_params)

        return self.recommender.predict(user_id, item_id)