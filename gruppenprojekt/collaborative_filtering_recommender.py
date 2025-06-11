from typing import Optional, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import NearestNeighbors

from gruppenprojekt.recommender import Recommender


class CollaborativeFilteringRecommender(Recommender):
    def __init__(self, data: pd.DataFrame, mode: Literal['user', 'item'] = 'user', display_results_for_each_step: Optional[bool] = False) -> None:
        super().__init__()
        self.display_results_for_each_step = display_results_for_each_step
        self.original_data = data
        self.mode = mode
        self.data = None
        self.knn: Optional[NearestNeighbors] = None
        self.knn_metric: Optional[str] = None
        self._preprocess_data()
        # pre-train default knn with cosine similarity
        self._init_knn(metric="cosine")

    def _init_knn(self, metric: str) -> None:
        """Initialise and fit a kNN model for the given metric."""
        self.knn = NearestNeighbors(metric=metric, algorithm="brute")
        self.knn.fit(self.data.values)
        self.knn_metric = metric


    def _preprocess_data(self) -> None:
        self.original_data = self.original_data.set_index("user_ID")
        self.original_data.index = self.original_data.index.astype(str) # convert the index to string (due to error with int values)
        if self.mode == 'item':
            self.data = self.original_data.T  # transpose for item based
        else:
            self.data = self.original_data  # original for user based


    def _calculate_distance_and_indices(self, allowed_indices: list[int]) -> (np.ndarray, np.ndarray):
        """Return distances and indices of the k nearest neighbours limited to allowed indices."""
        if self.similarity == "cosine":
            # reuse knn if possible
            if self.knn is None or self.knn_metric != "cosine":
                self._init_knn(metric="cosine")

            if self.mode == "item":
                index = self.data.index.get_loc(self.item_id)
            else:
                index = self.data.index.get_loc(self.user_id)

            distances, indices = self.knn.kneighbors(
                self.data.iloc[[index]], n_neighbors=len(self.data)
            )

            similar_distances = distances.flatten()[1:]
            similar_indices = indices.flatten()[1:]
            # restrict to allowed indices
            filtered = [(d, idx) for d, idx in zip(similar_distances, similar_indices) if idx in allowed_indices]
            if filtered:
                similar_distances, similar_indices = zip(*filtered)
                similar_distances = np.array(similar_distances)[: self.k]
                similar_indices = np.array(similar_indices, dtype=int)[: self.k]
            else:
                similar_distances = np.array([])
                similar_indices = np.array([], dtype=int)
        elif self.similarity == "pearson":
            if self.mode == "item":
                index = self.data.index.get_loc(self.item_id)
            else:
                index = self.data.index.get_loc(self.user_id)

            matrix = self.data.fillna(0.0).to_numpy(dtype=float)
            query = matrix[index]
            correlations = np.array([
                np.corrcoef(query, row)[0, 1] if np.std(row) > 0 else 0.0
                for row in matrix
            ])
            correlations[index] = -1  # ignore self
            sorted_idx = np.argsort(correlations)[::-1]
            filtered = [idx for idx in sorted_idx if idx in allowed_indices]
            similar_indices = np.array(filtered[: self.k])
            similar_distances = correlations[similar_indices]
        else:
            raise ValueError("Unsupported similarity metric.")

        return similar_distances, similar_indices

    def _calculate_similarities(self, similar_distances: np.ndarray) -> np.ndarray:
        if self.similarity == 'cosine':
            similarity = [1 - x for x in similar_distances]
            similarity = [(y + 1) / 2 for y in similarity]
            return np.array(similarity)
        elif self.similarity == 'pearson':
            # distances are already correlation coefficients
            return np.nan_to_num(similar_distances)
        else:
            raise ValueError("Unsupported similarity metric.")

    def _calculate_result(self, similarity: np.ndarray, ratings: np.ndarray) -> float:
        if self.calculation_variant == "weighted":
            if similarity.sum() == 0:
                return float(np.mean(ratings)) if len(ratings) > 0 else 0.0
            mean = np.dot(ratings, similarity) / similarity.sum()
            return float(mean)
        else:
            return float(np.mean(ratings)) if len(ratings) > 0 else 0.0


    def _check_values(self) -> None:
        if self.mode == 'user':
            if self.user_id not in self.data.index:
                raise ValueError(f"User {self.user_id} nicht in Daten.")
            if self.item_id not in self.data.columns:
                raise ValueError(f"Item {self.item_id} nicht in Daten.")
        elif self.mode == 'item':
            if self.user_id not in self.original_data.index:
                raise ValueError(
                    f"User {self.user_id} nicht in Originaldaten.")
            if self.item_id not in self.data.index:
                raise ValueError(
                    f"Item {self.item_id} nicht in transponierten Daten.")


    def _process_item_based(self) -> pd.DataFrame:
        user_ratings = self.original_data.loc[self.user_id]

        # filter based on the item. Only the users that already gave a rating are relevant
        rated_items = user_ratings[user_ratings > 0.0].index.tolist()

        if not rated_items:
            raise ValueError(f"User {self.user_id} hat keine Items bewertet!")

        return self.data.loc[rated_items + [self.item_id]]

    def _process_user_based(self) -> pd.DataFrame:
        # filter based on the item. Only the users that already gave a rating are relevant
        relevant_df = self.data[self.data[self.item_id] > 0.0]

        # add the user we are looking for (due to non-existing rating this user where filtered out)
        return pd.concat([relevant_df, self.data.loc[[self.user_id]]])

    def predict(
            self,
            user_id: str,
            item_id: str,
            similarity: Literal['cosine', 'pearson'] = 'cosine',
            calculation_variety: Literal['weighted', 'unweighted'] = 'weighted',
            k: Optional[int] = 3,
            second_k_value: Optional[int] = None) -> float:
        self._prepare_information(user_id=user_id, item_id=item_id, similarity=similarity, calculation_variant=calculation_variety, k=k)
        self._check_values()

        if self.mode == 'item':
            relevant_df = self._process_item_based()
        else:
            relevant_df = self._process_user_based()

        # make sure that there are no NaN values -> set NaN to 0.0
        relevant_df = relevant_df.fillna(0.0)
        allowed_indices = [self.data.index.get_loc(idx) for idx in relevant_df.index]
        similar_distances, similar_indices = self._calculate_distance_and_indices(allowed_indices)

        if self.mode == 'item':
            ratings = self.data.iloc[similar_indices][self.user_id].to_numpy()
        else:
            ratings = self.data.iloc[similar_indices][self.item_id].to_numpy()

        similarity = self._calculate_similarities(similar_distances)
        result = self._calculate_result(similarity, ratings)

        if self.display_results_for_each_step:
          self.explain(similar_indices, relevant_df, ratings, similarity, result)

        return result

    def explain(self, similar_indices, relevant_df, ratings, similarity, result) -> None:
        print("-" * 50)
        print(f"<mode: {self.mode}>")
        print(f"({self.calculation_variant}) Mittelwert: {result:.4f}")
        print(f"Metrik: {self.similarity}")
        print()
        print(f"k ({self.k}) ähnlichsten {'Items' if self.mode == 'item' else 'Nutzer'}:")
        df = pd.DataFrame({
            "ID": relevant_df.index[similar_indices],
            "rating": ratings,
            "similarity": similarity
        }).reset_index(drop=True)
        print(df.to_string(index=True, header=True))
        print("-" * 50)
