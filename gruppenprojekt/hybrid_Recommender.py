from typing import Literal, Optional

import numpy as np

import pandas as pd

from gruppenprojekt.collaborative_filtering_recommender import CollaborativeFilteringRecommender
from gruppenprojekt.content_based_recommender import ContentBasedRecommender
from gruppenprojekt.recommender import Recommender


class HybridRecommender(Recommender):
    def __init__(self, data: pd.DataFrame, item_profile: pd.DataFrame, user_ratings: pd.DataFrame, mode: Literal['user', 'item'] = 'user', alpha: float = 0.5):
        super().__init__()
        self.collaborative_recommender = CollaborativeFilteringRecommender(data=data, mode=mode)
        self.content_based_recommender = ContentBasedRecommender(item_profile=item_profile, user_ratings=user_ratings)
        self.alpha = alpha

    def optimize_alpha(self, validation_data: pd.DataFrame, alphas: list[float]) -> float:
        """Find the best alpha on validation data based on MAE."""
        best_alpha = self.alpha
        best_mae = float("inf")
        for a in alphas:
            self.alpha = a
            preds = []
            actuals = []
            for _, row in validation_data.iterrows():
                preds.append(
                    self.predict(
                        user_id=str(row["user_ID"]),
                        item_id=str(row["item_ID"]),
                    )
                )
                actuals.append(row["rating"])
            mae = float(np.mean(np.abs(np.array(preds) - np.array(actuals))))
            if mae < best_mae:
                best_mae = mae
                best_alpha = a
        self.alpha = best_alpha
        return best_alpha

    def predict(
            self,
            user_id: str,
            item_id: str,
            similarity: Optional[Literal['cosine', 'pearson']] = 'cosine',  # only for collaborative filtering
            calculation_variety: Optional[Literal['weighted', 'unweighted']] = 'weighted', # only for collaborative filtering
            k: Optional[int] = 3,
            second_k_value: Optional[int] = 3):

        collaborative_prediction = self.collaborative_recommender.predict(
            user_id=user_id,
            item_id=item_id,
            similarity=similarity, # ignore that it can be NONE
            calculation_variety=calculation_variety, # ignore that it can be NONE
            k=k
        )

        content_based_prediction = self.content_based_recommender.predict(
            user_id=user_id,
            item_id=item_id,
            similarity=similarity,
            calculation_variety=calculation_variety,
            k=second_k_value
        )

        # combine both with alpha as weight
        combined_prediction = (self.alpha * collaborative_prediction) + ((1 - self.alpha) * content_based_prediction)
        return combined_prediction
