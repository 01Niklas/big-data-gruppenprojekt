from typing import List
from pydantic import BaseModel
from loguru import logger
import pandas as pd
from datetime import datetime

from gruppenprojekt.collaborative_filtering_recommender import CollaborativeFilteringRecommender

class Test(BaseModel):
    name: str
    mode: str
    k_value: int
    metric: str
    calculation_variety: str


class TestResult(BaseModel):
    testname: str
    mode: str
    k_value: int
    metric: str
    calculation_variety: str
    mae: float


class TestResults(BaseModel): # just for saving in a "pretty" form
    date: str
    num_tests: int
    best_test: TestResult
    results: List[TestResult]



class MAETester:
    def __init__(self, tests: List[Test], test_data_path: str, data_path: str):
        self.tests = tests
        self.testdata = pd.read_csv(test_data_path)  # testdata (for evaluaton)
        self._prepare_data()
        self.data = pd.read_csv(data_path)  # trainingsdata
        self.results: List[TestResult] = []


    def _prepare_data(self):
        self.testdata["user_ID"] = self.testdata["user_ID"].astype(str)
        self.testdata["item_ID"] = self.testdata["item_ID"].astype(str)


    def run_tests(self):
        for test in self.tests:
            result = self._run_test(test)
            self.results.append(result)
            logger.success(f"Test abgeschlossen: {test.name}, MAE: {result.mae:.4f}\n")

        # display final results
        self._summarize_test_results()

        # save final results to file
        self._save_to_file()

    def _run_test(self, test: Test) -> TestResult:
        logger.info(f"Running test: {test.name}")

        recommender = CollaborativeFilteringRecommender(
            mode=test.mode, # ignore warnings that this is not the correct type - this is just a testing class...
            data=self.data,  # trainingdata
        )

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
                    similarity=test.metric, # ignore warnings that this is not the correct type - this is just a testing class...
                    calculation_variety=test.calculation_variety, # ignore warnings that this is not the correct type - this is just a testing class...
                    k=test.k_value,
                )
                predictions.append(predicted_rating) # save the predicted rating
                actuals.append(actual_rating)  # save the actual rating from testdata
            except ValueError as e:
                logger.warning(f"Fehler bei der Vorhersage: {e}")

        mae = self._mean_absolute_error(actuals, predictions)

        return TestResult(
            testname=test.name,
            mode=test.mode,
            k_value=test.k_value,
            metric=test.calculation_variety,
            calculation_variety=test.calculation_variety,
            mae=mae,
        )


    def _mean_absolute_error(self, actuals: List[float], predictions: List[float]) -> float:
        if not actuals or not predictions or len(actuals) != len(predictions):
            raise ValueError("Listen für tatsächliche und vorhergesagte Werte müssen gleich lang und nicht leer sein.")

        absolute_errors = [abs(a - p) for a, p in zip(actuals, predictions)]
        mae = sum(absolute_errors) / len(absolute_errors)
        return mae

    def _summarize_test_results(self):
        if not self.results:
            logger.info("Keine Testergebnisse vorhanden.")
            return

        summary_df = pd.DataFrame([{
            "Testname": result.testname,
            "Modus": result.mode,
            "k-Wert": result.k_value,
            "Metrik": result.metric,
            "Berechnungsvariante": result.calculation_variety,
            "MAE": result.mae
        } for result in self.results])

        print("-" * 50)
        print("Zusammenfassung der Testergebnisse:")
        print(summary_df.to_string(index=False))
        print("-" * 50)


    def _save_to_file(self):
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


