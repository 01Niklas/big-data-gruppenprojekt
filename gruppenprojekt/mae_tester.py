from typing import List, Literal, Optional
from pydantic import BaseModel
from loguru import logger
import pandas as pd
from datetime import datetime

from gruppenprojekt.hybrid_Recommender import HybridRecommender
from gruppenprojekt.collaborative_filtering_recommender import CollaborativeFilteringRecommender
from gruppenprojekt.content_based_recommender import ContentBasedRecommender

class Test(BaseModel):
    name: str
    type: Literal["collaborative_filtering", "content_based", "hybrid"]
    mode: Optional[Literal["user", "item"]] = "item"
    first_k_value: int
    second_k_value: Optional[int] = None
    metric: Optional[Literal["cosine", "pearson"]] = 'cosine'
    calculation_variety: Optional[Literal["weighted", "unweighted"]] = 'weighted'
    alpha: Optional[float] = 0.5


class TestResult(BaseModel):
    name: str
    type: Literal["collaborative_filtering", "content_based", "hybrid"]
    mode: Literal["user", "item"]
    k_value: int
    metric: Literal["cosine", "pearson"]
    calculation_variety: Literal["weighted", "unweighted"]
    alpha: float
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


    def run_tests(self) -> pd.DataFrame:
        for test in self.tests:
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

        for row in testdata_list:
            user_id: str = str(row[0])
            item_id: str = str(row[1])
            actual_rating = row[2]

            try:
                predicted_rating = recommender.predict(
                    user_id=user_id,
                    item_id=item_id,
                    similarity=test.metric,
                    calculation_variety=test.calculation_variety,
                    k=test.first_k_value
                )
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except ValueError as e:
                logger.warning(f"Fehler bei der Vorhersage: {e}")

        mae = self._mean_absolute_error(actuals, predictions)

        return TestResult(
            name=test.name,
            type=test.type,
            mode=test.mode,
            k_value=test.first_k_value,
            metric=test.metric,
            calculation_variety=test.calculation_variety,
            alpha=test.alpha,
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


