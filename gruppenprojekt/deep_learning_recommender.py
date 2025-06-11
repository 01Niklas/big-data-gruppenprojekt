import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from loguru import logger


class DeepLearningRecommender(nn.Module):
    def __init__(self, embedding_dim: int = 50, data: pd.DataFrame = None, epochs: int = 10, batch_size: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = None
        self.user_mapping = None
        self.item_mapping = None
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs

        if data is not None:
            logger.info("Training model with provided data.")
            self.train_model()

    def _preprocess_data(self):
        if self.data is None:
            raise ValueError("No data available for preprocessing.")
        logger.info("Preprocessing data: Converting IDs to strings and encoding them.")
        self.data["user_ID"] = self.data["user_ID"].astype(str)
        self.data["item_ID"] = self.data["item_ID"].astype(str)
        self.user_mapping = pd.Categorical(self.data["user_ID"])
        self.item_mapping = pd.Categorical(self.data["item_ID"])
        self.data["user_ID"] = self.user_mapping.codes
        self.data["item_ID"] = self.item_mapping.codes

    def _build_model(self, num_users: int, num_items: int):

        class RecommenderModel(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.fc1 = nn.Linear(2 * embedding_dim, 128)
                self.fc2 = nn.Linear(128, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(p=0.5)

            def forward(self, user_ids, item_ids):
                user_vector = self.user_embedding(user_ids)
                item_vector = self.item_embedding(item_ids)
                combined_vector = torch.cat([user_vector, item_vector], dim=1)
                x = self.relu(self.fc1(combined_vector))
                x = self.dropout(x)
                output = self.fc2(x)
                return output

        logger.info("Building the recommendation model.")
        return RecommenderModel(num_users, num_items, self.embedding_dim)

    def train_model(self, lr: float = 0.001):
        if self.user_mapping is None or self.item_mapping is None:
            self._preprocess_data()

        num_users = len(self.user_mapping.categories)
        num_items = len(self.item_mapping.categories)
        self.model = self._build_model(num_users, num_items)

        train, _ = train_test_split(self.data, test_size=0.2, random_state=42)
        user_ids_train = torch.tensor(train["user_ID"].values, dtype=torch.long)
        item_ids_train = torch.tensor(train["item_ID"].values, dtype=torch.long)
        ratings_train = torch.tensor(train["rating"].values, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        logger.info(f"Starting training for {self.epochs} epochs.", self.epochs)
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(0, len(user_ids_train), self.batch_size):
                user_batch = user_ids_train[i:i + self.batch_size]
                item_batch = item_ids_train[i:i + self.batch_size]
                rating_batch = ratings_train[i:i + self.batch_size]

                optimizer.zero_grad()
                predictions = self.model(user_batch, item_batch).squeeze()
                loss = criterion(predictions, rating_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(user_ids_train)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self, user_id: str, item_id: str) -> float:
        if self.model is None or self.user_mapping is None or self.item_mapping is None:
            raise ValueError("The model has not been trained yet.")

        if user_id not in self.user_mapping.categories or item_id not in self.item_mapping.categories:
            raise ValueError("Invalid user or item ID.")

        user_id_encoded = self.user_mapping.categories.get_loc(user_id)
        item_id_encoded = self.item_mapping.categories.get_loc(item_id)

        user_tensor = torch.tensor([user_id_encoded], dtype=torch.long)
        item_tensor = torch.tensor([item_id_encoded], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(user_tensor, item_tensor)

        return prediction.item()
