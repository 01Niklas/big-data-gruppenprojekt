from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gruppenprojekt.recommender import Recommender


class ContentBasedRecommender(Recommender):
    def __init__(self, item_profile: pd.DataFrame, user_ratings: pd.DataFrame):
        super().__init__()
        self.item_profile = item_profile
        self.user_ratings = user_ratings
        self.k = 3
        self.feature_matrix = None
        self._preprocess_data()

        self._calculate_tfidf_matrix()

    def _preprocess_data(self):
        self.item_profile["item_ID"] = self.item_profile["item_ID"].astype(str)
        self.user_ratings["item_ID"] = self.user_ratings["item_ID"].astype(str)
        self.user_ratings["user_ID"] = self.user_ratings["user_ID"].astype(str)


    def _calculate_tfidf_matrix(self):
        self.item_profile["title"] = self.item_profile["title"].fillna("")
        title_vectorizer = TfidfVectorizer()
        title_features = title_vectorizer.fit_transform(self.item_profile["title"])

        genre_cols = [col for col in self.item_profile.columns if col.startswith("Genre_")]
        self.item_profile["genre_text"] = self.item_profile[genre_cols].astype(int).apply(
            lambda row: " ".join([col.replace("Genre_", "") for col, val in row.items() if val == 1]), axis=1
        )
        genre_vectorizer = TfidfVectorizer()
        genre_features = genre_vectorizer.fit_transform(self.item_profile["genre_text"])

        language_dummies = pd.get_dummies(self.item_profile["original_language"], prefix="lang")

        runtime_bucket = pd.qcut(self.item_profile["runtime"], q=3, labels=["kurz", "mittel", "lang"])
        runtime_dummies = pd.get_dummies(runtime_bucket, prefix="runtime")

        self.item_profile["log_budget"] = np.log1p(self.item_profile["budget"].fillna(0))
        self.item_profile["log_revenue"] = np.log1p(self.item_profile["revenue"].fillna(0))
        scaler = StandardScaler()
        scaled_numericals = scaler.fit_transform(self.item_profile[["log_budget", "log_revenue"]])

        self.feature_matrix = hstack([
            title_features,
            genre_features,
            language_dummies.values,
            runtime_dummies.values,
            scaled_numericals
        ])

        self.feature_matrix = csr_matrix(self.feature_matrix)

    def predict(
            self,
            user_id: str,
            item_id: str,
            similarity: Optional[Literal['cosine', 'pearson']] = 'cosine',  # only for collaborative filtering
            calculation_variety: Optional[Literal['weighted', 'unweighted']] = 'weighted', # only for collaborative filtering
            k: Optional[int] = 3):

        self._prepare_information(user_id=user_id, item_id=item_id, k=k)
        if user_id not in self.user_ratings["user_ID"].values:
            raise ValueError(f"User-ID {user_id} not found.")

        if item_id not in self.item_profile["item_ID"].values:
            raise ValueError(f"Item-ID {item_id} not found.")

        rated_items = self.user_ratings[self.user_ratings["user_ID"] == user_id]
        rated_item_ids = rated_items["item_ID"].values
        rated_item_indices = self.item_profile[self.item_profile["item_ID"].isin(rated_item_ids)].index

        if len(rated_item_indices) == 0:
            return 0.0

        filtered_matrix = self.feature_matrix[rated_item_indices]

        knn = NearestNeighbors(metric="cosine", algorithm="brute")
        knn.fit(filtered_matrix)

        item_index = self.item_profile[self.item_profile["item_ID"] == item_id].index[0]
        distances, indices = knn.kneighbors(self.feature_matrix[item_index], n_neighbors=self.k + 1)

        similar_items = indices.flatten()[1:]
        similar_item_indices = rated_item_indices[similar_items]

        similar_ratings = []
        for idx in similar_item_indices:
            similar_item_id = self.item_profile.iloc[idx]["item_ID"]
            user_rating = self.user_ratings[
                (self.user_ratings["user_ID"] == user_id) & (self.user_ratings["item_ID"] == similar_item_id)
            ]
            if not user_rating.empty:
                similar_ratings.append(user_rating["rating"].values[0])

        if not similar_ratings:
            return 0.0

        return sum(similar_ratings) / len(similar_ratings)