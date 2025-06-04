# Gruppenprojekt - Big Data

## Project Description
This repository contains an implementation of a user-based recommendation system in Python. The system predicts the rating of an item by a user based on the ratings of similar users.


## Project Setup
For dependency management, the project uses Poetry. As explained in [the Poetry documentation](https://python-poetry.org/docs/), you can install Poetry in a specific virtual environment or globally. 
After Poetry is installed, you can use it to download all dependencies for the application by running:

```bash
poetry install
```

in the repository root directory.

### Adding Dependencies in the Python Backend
To add new dependencies, use (e.g., for the package `pandas`):

```bash
poetry add pandas
```

This will add the package to the `pyproject.toml` file and to the `poetry.lock` file, and install it in the virtual environment.


### Running the application with simple (single) prediction
```python
bewertungsmatrix_df = pd.read_csv('data/Bewertungsmatrix_FlixNet.csv')

# running the item based recommender
item_based_recommender = CollaborativeFilteringRecommender(data=bewertungsmatrix_df, mode='item')
item_based_recommender.predict(user_id="741", item_id="1088", k=10)
item_based_recommender.predict(user_id="5378", item_id="1088", k=2)

# running the user based recommender
user_based_recommender = CollaborativeFilteringRecommender(data=bewertungsmatrix_df, mode='user')
ser_based_recommender.predict(user_id="741", item_id="104", calculation_variety="unweighted", k=3)
user_based_recommender.predict(user_id="5378", item_id="1073", k=2)
```

### Running the application with automated tests (MAE)
```python
# defining the testcases
tests = [
    Test(name="UserBased_1", mode="user", k_value=2, metric="cosine", calculation_variety="weighted"),
    ...
]

# creating and running the MAETester with the given Tests and paths to test and training data
tester = MAETester(
    tests=tests,
    test_data_path="data/Testdaten_FlixNet.csv",
    data_path="data/Bewertungsmatrix_FlixNet.csv"
)
tester.run_tests()
```

## License
This project is under the MIT License.