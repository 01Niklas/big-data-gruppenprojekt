from typing import Optional, Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from gruppenprojekt.recommender import Recommender


class CollaborativeFilteringRecommender(Recommender):
    def __init__(self, data: pd.DataFrame, mode: Literal['user', 'item'] = 'user', display_results_for_each_step: Optional[bool] = False) -> None:
        super().__init__()
        self.display_results_for_each_step = display_results_for_each_step
        self.original_data = data
        self.mode = mode
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        self.original_data = self.original_data.set_index("user_ID")
        self.original_data.index = self.original_data.index.astype(str) # convert the index to string (due to error with int values)
        if self.mode == 'item':
            self.data = self.original_data.T  # transpose for item based
        else:
            self.data = self.original_data  # original for user based


    def _calculate_distance_and_indices(self, dataframe: pd.DataFrame) -> ([], []):
        metric = "cosine" if self.similarity == 'cosine' else "euclidean"
        knn = NearestNeighbors(metric=metric, algorithm='brute')
        knn.fit(dataframe.values)
        distances, indices = knn.kneighbors(dataframe.values, n_neighbors=self.k + 1)

        if self.mode == 'item':
            index = dataframe.index.get_loc(self.item_id)
        else:
            index = dataframe.index.get_loc(self.user_id)

        similar_distances = distances[index, 1:]
        similar_indices = indices[index, 1:]

        return similar_distances, similar_indices

    @staticmethod
    def _calculate_similarities(similar_distances: np.ndarray) -> np.ndarray:
        similarity = [1 - x for x in similar_distances]
        similarity = [(y + 1) / 2 for y in similarity]
        return np.array(similarity)

    def _calculate_result(self, ratings: np.ndarray) -> float:
        if self.calculation_variant == "weighted":
            return float(np.mean(ratings))
        else:
            raise ValueError(f"This calculation method is not implemented yet due to the dataset which is already weighted.")

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

    @staticmethod
    def _normalize_for_pearson(relevant_df: pd.DataFrame) -> pd.DataFrame:
        mean_values = relevant_df.mean(axis=1).to_numpy()
        relevant_df = relevant_df.sub(mean_values, axis=0)
        return relevant_df

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

        if similarity == 'pearson':
            self._normalize_for_pearson(relevant_df)

        # make sure that there are no NaN values -> set NaN to 0.0
        relevant_df = relevant_df.fillna(0.0)
        similar_distances, similar_indices = self._calculate_distance_and_indices(dataframe=relevant_df)

        if self.mode == 'item':
            ratings = relevant_df.iloc[similar_indices][self.user_id].to_numpy()
        else:
            ratings = relevant_df.iloc[similar_indices][self.item_id].to_numpy()

        similarity = self._calculate_similarities(similar_distances)
        result = self._calculate_result(ratings)

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
