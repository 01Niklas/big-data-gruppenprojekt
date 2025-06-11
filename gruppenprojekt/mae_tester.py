from typing import List, Literal, Optional
from pydantic import BaseModel
from loguru import logger
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from gruppenprojekt.hybrid_Recommender import HybridRecommender
from gruppenprojekt.collaborative_filtering_recommender import CollaborativeFilteringRecommender
from gruppenprojekt.content_based_recommender import ContentBasedRecommender
from gruppenprojekt.deep_learning_recommender import DeepLearningRecommender

class Test(BaseModel):
    name: str
    type: Literal["collaborative_filtering", "content_based", "hybrid", "deep_learning"]
    mode: Optional[Literal["user", "item"]] = "item"
    k_value: Optional[int] = 3
    second_k_value: Optional[int] = 3
    metric: Optional[Literal["cosine", "pearson"]] = 'cosine'
    calculation_variety: Optional[Literal["weighted", "unweighted"]] = 'weighted'
    batch_size: Optional[int] = 0
    embedding_dim: Optional[int] = 0
    epochs: Optional[int] = 0
    alpha: Optional[float] = 0.5


class TestResult(BaseModel):
    name: str
    type: Literal["collaborative_filtering", "content_based", "hybrid", "deep_learning"]
    mode: Literal["user", "item"] | None
    k_value: int | None
    second_k_value: int | None
    metric: Literal["cosine", "pearson"] | None
    calculation_variety: Literal["weighted", "unweighted"] | None
    alpha: float | None
    epochs: int | None
    batch_size: int | None
    embedding_dim: int | None
    mae: float


class TestResults(BaseModel): # just for saving in a "pretty" form
    date: str
    num_tests: int
    best_test: TestResult
    results: List[TestResult]



class MAETester:
    def __init__(self, tests: List[Test], test_data_path: str, data_path: str, item_profile_path: str, user_ratings: str):
        self.tests = tests
        self.testdata = pd.read_csv(test_data_path)  # testdata (for evaluaton)
        self.item_profile = pd.read_csv(item_profile_path)
        self.user_ratings = pd.read_csv(user_ratings)
        self._prepare_data()
        self.data = pd.read_csv(data_path)  # trainings-data
        self.results: List[TestResult] = []


    def _prepare_data(self):
        self.testdata["user_ID"] = self.testdata["user_ID"].astype(str)
        self.testdata["item_ID"] = self.testdata["item_ID"].astype(str)

    # because we only have 0.5 steps in testdata
    def _round_to_nearest_half(self, value: float):
        return round(value * 2) / 2

    def _run_deep_learning_recommender(self, test: Test) -> TestResult:
        logger.info(f"Running Deep Learning Recommender test: {test.name}")

        recommender = DeepLearningRecommender(
            embedding_dim=test.embedding_dim,
            data=self.user_ratings,
            batch_size=test.batch_size,
            epochs=test.epochs
        )

        predictions = []
        actuals = []

        testdata_list = self.testdata.to_numpy()

        for row in tqdm(testdata_list, desc="Vorhersagen werden berechnet"):
            user_id: str = str(row[0])
            item_id: str = str(row[1])
            actual_rating = row[2]

            try:
                predicted_rating = recommender.predict(user_id=user_id, item_id=item_id)
                predicted_rating = self._round_to_nearest_half(predicted_rating)

                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except ValueError as e:
                logger.warning(f"Fehler bei der Vorhersage: {e}")

        mae = self._mean_absolute_error(actuals, predictions)

        return TestResult(
            name="deep_learning",
            type=test.type,
            mode=None,  # Falls nicht relevant, auf `None` setzen
            k_value=None,  # Falls nicht relevant, auf `None` setzen
            metric=None,  # Falls nicht relevant, auf `None` setzen
            calculation_variety=None,  # Falls nicht relevant, auf `None` setzen
            alpha=None,  # Falls nicht relevant, auf `None` setzen
            mae=mae,
            second_k_value=None,
            epochs=test.epochs,
            batch_size=test.batch_size,
            embedding_dim=test.embedding_dim
        )


    def run_tests(self) -> pd.DataFrame:
        for test in self.tests:
            if test.type == "deep_learning":
                result = self._run_deep_learning_recommender(test)
            else:
                result = self._run_test(test)
            self.results.append(result)
            logger.success(f"Test abgeschlossen: {test.name}, MAE: {result.mae:.4f}\n")

        # display final results
        result_df = self._summarize_test_results()

        # save final results to file
        self._save_to_file()

        return result_df

    def _run_test(self, test: Test) -> TestResult:
        logger.info(f"Running test: {test.name}")

        if test.type == "content_based":
            recommender = ContentBasedRecommender(
                item_profile=self.item_profile,
                user_ratings=self.user_ratings,
            )
        elif test.type == "collaborative_filtering":
            recommender = CollaborativeFilteringRecommender(
                mode=test.mode, # ignore type (that this can be NONE)
                data=self.data,
            )
        elif test.type == "hybrid":
            recommender = HybridRecommender(
                data=self.data,
                item_profile=self.item_profile,
                user_ratings=self.user_ratings,
                mode=test.mode,  # ignore type (that this can be NONE)
                alpha=test.alpha,
            )
        else:
            raise ValueError(f"Unbekannter Recomendertyp: {test.type}")

        predictions = []
        actuals = []

        testdata_list = self.testdata.to_numpy()

        for row in tqdm(testdata_list, desc="Vorhersagen werden berechnet"):
            user_id: str = str(row[0])
            item_id: str = str(row[1])
            actual_rating = row[2]

            try:
                predicted_rating = recommender.predict(
                    user_id=user_id,
                    item_id=item_id,
                    similarity=test.metric,
                    calculation_variety=test.calculation_variety,
                    k=test.k_value,
                    second_k_value=test.second_k_value,
                )

                predicted_rating = self._round_to_nearest_half(predicted_rating)

                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except ValueError as e:
                logger.warning(f"Fehler bei der Vorhersage: {e}")

        mae = self._mean_absolute_error(actuals, predictions)

        return TestResult(
            name=test.name,
            type=test.type,
            mode=test.mode,
            k_value=test.k_value,
            metric=test.metric,
            calculation_variety=test.calculation_variety,
            alpha=test.alpha,
            batch_size=test.batch_size,
            embedding_dim=test.embedding_dim,
            epochs=test.epochs,
            second_k_value=test.second_k_value,
            mae=mae,
        )

    @staticmethod
    def _mean_absolute_error(actuals: List[float], predictions: List[float]) -> float:
        if not actuals or not predictions or len(actuals) != len(predictions):
            raise ValueError("Listen für tatsächliche und vorhergesagte Werte müssen gleich lang und nicht leer sein.")

        absolute_errors = [abs(a - p) for a, p in zip(actuals, predictions)]
        mae = sum(absolute_errors) / len(absolute_errors)
        return mae

    def _summarize_test_results(self) -> pd.DataFrame:
        if not self.results:
            logger.info("Keine Testergebnisse vorhanden.")
            return

        summary_df = pd.DataFrame([{
            "Testname": result.name,
            "Recomendertyp": result.type,
            "Modus": result.mode if result.type == "collaborative_filtering" else "/",
            "k-Wert": result.k_value,
            "Metrik": result.metric if result.type == "collaborative_filtering" else "/",
            "Berechnungsvariante": result.calculation_variety if result.type == "collaborative_filtering" else "/",
            "Alpha (weight)" : result.alpha if result.type == "hybrid" else "/",
            "MAE": result.mae
        } for result in self.results])

        print("-" * 50)
        print("Zusammenfassung der Testergebnisse:")
        print(summary_df.to_string(index=False))
        print("-" * 50)

        return summary_df


    def _save_to_file(self) -> None:
        if not self.results:
            logger.info("Keine Testergebnisse vorhanden, nichts zu speichern.")
            return
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_test = min(self.results, key=lambda result: result.mae)  # best results based on mae
        test_results = TestResults(
            date=date,
            num_tests=len(self.results),
            best_test=best_test,
            results=self.results
        )

        file_path = f"./outputs/testergebnis_{date.replace(':', '-')}.json"

        with open(file_path, "w", encoding="utf-8") as json_file:
            json_file.write(test_results.model_dump_json(indent=4))

        logger.success(f"Testergebnisse erfolgreich gespeichert.")


