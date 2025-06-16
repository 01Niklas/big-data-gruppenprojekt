from typing import Literal, Optional

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
