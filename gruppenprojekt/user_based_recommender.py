from typing import Optional, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import NearestNeighbors


class UserBasedRecommender:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._prepocess_data()
        self.user_id = None
        self.item_id = None
        self.k = 3  # default
        self.similarity = "cosine"  # default
        self.calculation_variant = "gewichtet"  # default

    def _prepocess_data(self):
        self.data = self.data.set_index("user_ID")
        self.data.index = self.data.index.astype(str)  # convert the index to string (due to error with int values)

    def _prepare_information(self, user_id: str, item_id: str, similarity: str, calculation_variant: str, k: int):
        self.user_id = user_id
        self.item_id = item_id
        self.similarity = similarity
        self.calculation_variant = calculation_variant
        self.k = k

    def _calculate_distance_and_indices(self, dataframe: pd.DataFrame) -> ([], []):
        knn = NearestNeighbors(metric=self.similarity, algorithm='brute')
        knn.fit(dataframe.values)
        distances, indices = knn.kneighbors(dataframe.values, n_neighbors=self.k + 1)

        user_index = self.data.index.get_loc(self.user_id)
        logger.debug(f"Index of user: {self.user_id} is {user_index}")

        similar_users_distances = distances[user_index, 1:]
        logger.debug(f"Similar users (distances): {similar_users_distances}")

        similar_users_indices = indices[user_index, 1:]
        logger.debug(f"Similar users (indices): {similar_users_indices}")

        return similar_users_distances, similar_users_indices

    def _calculate_similarities(self, similar_users_distances) -> np.ndarray:
        if self.similarity == 'cosine':
            user_similarity = [1 - x for x in similar_users_distances]
            user_similarity = [(y + 1) / 2 for y in user_similarity]
            user_similarity = np.array(user_similarity)
            return user_similarity
        else:
            # TODO: add pearson?
            raise ValueError("Unsupported similarity metric.")

    def _calculate_result(self, user_similarity, similarUser_ratings) -> float:
        if self.calculation_variant == "gewichtet":
            mean = np.dot(similarUser_ratings, user_similarity) / user_similarity.sum()
            logger.debug(f"{self.calculation_variant} Mittelwert: {mean}")
            return mean
        else:
            return np.mean(similarUser_ratings)

    def predict(self, user_id=None, item_id=None, similarity: Literal['cosine', 'pearson'] = 'cosine', calculation_variety: Literal['gewichtet', 'ungewichtet'] = 'gewichtet', k: Optional[int] = 3):
        self._prepare_information(user_id, item_id, similarity, calculation_variety, k)
        logger.debug(f"User ID: {self.user_id}, Item ID: {self.item_id}")

        # check if these values (user_id and item_id) are valid and in the dataset
        if self.user_id not in self.data.index:
            raise ValueError(f"User {self.user_id} nicht in Daten!")

        if self.item_id not in self.data.columns:
            raise ValueError(f"Item {self.item_id} nicht in Daten!")


        # filter based on the item. Only the users that already gave a rating are relevant
        relevant_users_df = self.data[self.data[self.item_id] > 0.0]

        # add the user we are looking for (due to non-existing rating this user where filtered out)
        relevant_users_df = pd.concat([relevant_users_df, self.data.loc[[self.user_id]]])

        # make sure that there are no NaN values -> set NaN to 0.0
        relevant_users_df = relevant_users_df.fillna(0.0)

        # calculate the distances and indices to the k-nearest neighbors
        similar_users_distances, similar_users_indices = self._calculate_distance_and_indices(dataframe=relevant_users_df)

        # extract the k-nearest users from the dataset and calculate the similarity (cosine, pearson)
        similarUser_ratings = relevant_users_df.iloc[similar_users_indices][self.item_id].to_numpy()
        user_similarity = self._calculate_similarities(similar_users_distances)

        # calculate the mean / result
        result = self._calculate_result(user_similarity, similarUser_ratings)

        # add explanation to display result and k-most similar users in console
        self.explain(similar_users_indices, relevant_users_df, similarUser_ratings, user_similarity, result)

        return result

    def explain(self, similar_users_indices, relevant_users_df, similarUser_ratings, user_similarity, result):
        print("-" * 50)
        print(f"({self.calculation_variant}) Mittelwert: {result:.4f}")
        print(f"Metrik: {self.similarity}")
        print()
        print(f"k ({self.k}) ähnlichsten Nutzer:")
        df = pd.DataFrame({
            "userId": relevant_users_df.index[similar_users_indices],
            "rating": similarUser_ratings,
            "similarity": user_similarity
        }).reset_index(drop=True)
        print(df.to_string(index=True, header=True))
        print("-" * 50)

    def required_parameters(self):
        return ['user_id', 'item_id']
