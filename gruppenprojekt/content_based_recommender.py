from typing import Optional, Literal, List

import numpy as np
import pandas as pd
from pyexpat import features
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from loguru import logger

from gruppenprojekt.recommender import Recommender


class ContentBasedRecommender(Recommender):
    def __init__(self, item_profile: pd.DataFrame, user_ratings: pd.DataFrame) -> None:
        super().__init__()
        self.item_profile = item_profile
        self.user_ratings = user_ratings
        self.k = 3
        self.feature_matrix = None
        self.knn: Optional[NearestNeighbors] = None
        self.item_index_map: dict[str, int] = {}
        self._preprocess_data()

        # check if the features "budget", "revenue", "runtime" are relevant for the item/rating correlation
        self._check_features_correlation(features=["budget", "revenue", "runtime"])
        self._calculate_tfidf_matrix()
        self._init_knn()


    def _preprocess_data(self):
        self.item_profile["item_ID"] = self.item_profile["item_ID"].astype(str)
        self.user_ratings["item_ID"] = self.user_ratings["item_ID"].astype(str)
        self.user_ratings["user_ID"] = self.user_ratings["user_ID"].astype(str)

    def _check_features_correlation(self, features: List[str]) -> None:
        irrelevant_features = []  #  list for irrelevant feature that will be removed

        for feature in features:
            if feature not in self.item_profile.columns:
                continue

            # combine item and user ratings
            merged_data = pd.merge(self.user_ratings, self.item_profile, on="item_ID")

            # convert to numeric
            feature_data = pd.to_numeric(merged_data[feature].fillna(0), errors="coerce")
            rating_data = pd.to_numeric(merged_data["rating"].fillna(0), errors="coerce")

            # calculate the correlation between the user rating and the feature
            correlation, p_value = stats.pearsonr(feature_data, rating_data)

            # check if the correlation is relevant / significant
            if abs(correlation) < 0.1 or p_value > 0.05:
                logger.debug(f"Feature '{feature}' does not have a sigificant correlation and will be ignored.")
                irrelevant_features.append(feature)
            else:
                logger.debug(f"Feature '{feature}' has a significant correlation: {correlation}")

        self.item_profile.drop(columns=irrelevant_features, inplace=True)


    def _safe_get_feature(self, feature_name):
        if feature_name in self.item_profile.columns:
            return self.item_profile[feature_name]
        else:
            return None

    def _calculate_tfidf_matrix(self) -> None:
        # optional but if the title is empty we set it as an empty string
        self.item_profile["title"] = self.item_profile["title"].fillna("")

        # use the TfidfVectorizer() to transform title into numerical feature
        title_vectorizer = TfidfVectorizer()
        title_features = title_vectorizer.fit_transform(self.item_profile["title"])

        # change genre columns in text by just extracting the word after '"Genre_"'
        genre_cols = [col for col in self.item_profile.columns if col.startswith("Genre_")]
        if genre_cols:
            self.item_profile["genre_text"] = self.item_profile[genre_cols].astype(int).apply(
                lambda row: " ".join([col.replace("Genre_", "") for col, val in row.items() if val == 1]), axis=1
            )
            genre_vectorizer = TfidfVectorizer()
            genre_features = genre_vectorizer.fit_transform(self.item_profile["genre_text"])
        else:
            genre_features = np.empty((len(self.item_profile), 0))

        # the language of the items transformed into one-hot-encoded-dummies
        language_dummies = pd.get_dummies(self.item_profile["original_language"], prefix="lang")

        # put runtime into three categories (short, medium, long)
        runtime_feature = self._safe_get_feature("runtime")
        if runtime_feature is not None:
            runtime_bucket = pd.qcut(runtime_feature, q=3, labels=["kurz", "mittel", "lang"])
            runtime_dummies = pd.get_dummies(runtime_bucket, prefix="runtime")
        else:
            runtime_dummies = pd.DataFrame(index=self.item_profile.index)

        # budget and include will be logarithmically transformed and then scaled
        numerical_features = []
        if "budget" in self.item_profile.columns:
            self.item_profile["log_budget"] = np.log1p(self.item_profile["budget"].fillna(0))
            numerical_features.append("log_budget")
        if "revenue" in self.item_profile.columns:
            self.item_profile["log_revenue"] = np.log1p(self.item_profile["revenue"].fillna(0))
            numerical_features.append("log_revenue")

        if numerical_features:
            scaler = StandardScaler()
            scaled_numericals = scaler.fit_transform(self.item_profile[numerical_features])
        else:
            scaled_numericals = np.empty((len(self.item_profile), 0))

        # create feature matrix
        self.feature_matrix = hstack([
            title_features,
            genre_features,
            language_dummies.values,
            runtime_dummies.values,
            scaled_numericals
        ])

        self.feature_matrix = csr_matrix(self.feature_matrix)

    def _init_knn(self) -> None:
        """Initialise KNN model and build index mapping."""
        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn.fit(self.feature_matrix)
        self.item_index_map = {
            str(item_id): idx for idx, item_id in enumerate(self.item_profile["item_ID"])
        }

    def _check_values(self):
        if self.user_id not in self.user_ratings["user_ID"].values:
            raise ValueError(f"User-ID {self.user_id} not found.")

        if self.item_id not in self.item_profile["item_ID"].values:
            raise ValueError(f"Item-ID {self.item_id} not found.")


    def predict(
            self,
            user_id: str,
            item_id: str,
            similarity: Optional[Literal['cosine', 'pearson']] = 'cosine',  # only for collaborative filtering
            calculation_variety: Optional[Literal['weighted', 'unweighted']] = 'weighted', # only for collaborative filtering
            k: Optional[int] = 3,
            second_k_value: Optional[int] = None) -> float:

        # default function to save all the information
        self._prepare_information(user_id=user_id, item_id=item_id, k=k)

        # check if the values included in the dataframes
        self._check_values()

        # extract only the items, the user rated
        rated_items = self.user_ratings[self.user_ratings["user_ID"] == user_id]
        rated_item_ids = rated_items["item_ID"].values

        # this case can happen when k is greater than the rated items by the user
        if self.k > len(rated_item_ids):
            self.k = len(rated_item_ids)

        rated_item_indices = [self.item_index_map[item] for item in rated_item_ids if item in self.item_index_map]

        # check if the user rated some items... if not then return 0.0
        if len(rated_item_indices) == 0:
            return 0.0

        item_index = self.item_index_map.get(item_id)
        if item_index is None:
            return 0.0

        distances, indices = self.knn.kneighbors(self.feature_matrix[item_index], n_neighbors=len(self.item_profile))

        # the similar item indices beginning with the first real neighbor
        similar_items = [idx for idx in indices.flatten() if idx in rated_item_indices and idx != item_index][: self.k]
        similar_item_indices = similar_items

        # extract for each item in the similar item indices list the rating and save it in the list
        similar_ratings = []
        for idx in similar_item_indices:
            similar_item_id = self.item_profile.iloc[idx]["item_ID"]
            user_rating = self.user_ratings[(self.user_ratings["user_ID"] == user_id) & (self.user_ratings["item_ID"] == similar_item_id)]
            if not user_rating.empty:
                similar_ratings.append(user_rating["rating"].values[0])

        # if the similar ratings is zero then we return a default 0.0
        if not similar_ratings:
            return 0.0

        # calculate the predicted rating based on the sum of ratings and len of ratings
        return sum(similar_ratings) / len(similar_ratings)
