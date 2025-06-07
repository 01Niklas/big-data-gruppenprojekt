from abc import abstractmethod
from typing import Optional, Literal


class Recommender:
    def __init__(self):
        self.k = 3 # default
        self.user_id = None
        self.item_id = None
        self.similarity: Literal["cosine", "pearson"] = "cosine"  # default
        self.calculation_variant: Literal["weighted", "unweighted"] = "weighted"  # default


    @abstractmethod
    def _preprocess_data(self):
        ...


    def _prepare_information(self, user_id: str, item_id: str, k: int, similarity: Literal["cosine", "pearson"] = "cosine", calculation_variant: Literal["weighted", "unweighted"] = "weighted") -> None:
        self.user_id = user_id
        self.item_id = item_id
        self.similarity = similarity
        self.calculation_variant = calculation_variant
        self.k = k


    @abstractmethod
    def predict(
            self,
            user_id: str,
            item_id: str,
            similarity: Optional[Literal['cosine', 'pearson']] = 'cosine',   # only for collaborative filtering
            calculation_variety: Optional[Literal['weighted', 'unweighted']] = 'weighted',  # only for collaborative filtering
            k: Optional[int] = 3,
            second_k_value: Optional[int] = None):
        ...