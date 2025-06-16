from gruppenprojekt.hybrid_Recommender import HybridRecommenderfrom gruppenprojekt.gruppenprojekt_recommender import HyperparamOptimizedDeepLearningRecommender

# Group Project - Big Data

## Project Description
This repository contains the implementation of a recommendation system in Python. The system predicts a user's rating for an item based on the ratings of similar users or items. Additionally, a deep learning-based recommendation system and automated evaluation tools are integrated.

## Project Setup
Poetry is used for dependency management. As described in the [Poetry documentation](https://python-poetry.org/docs/), Poetry can be installed in a specific virtual environment or globally.
After installing Poetry, all project dependencies can be installed with the following command:

```bash
poetry install
```
this will add the package to the pyproject.toml and poetry.lock files and install it in the virtual environment.

## Usage
### Simple (Single) Prediction
```python
# Load the ratings matrix
bewertungsmatrix_df = pd.read_csv('data/Bewertungsmatrix_FlixNet.csv')
itemprofile_df = pd.DataFrame('data/Itemprofile_FlixNet.csv')
user_ratings_df = pd.DataFrame('data/Ratings_FlixNet.csv')
testdata_df = pd.DataFrame('data/Testdaten_FlixNet.csv')

# Item-based collaborative filtering
item_based_recommender = CollaborativeFilteringRecommender(data=bewertungsmatrix_df, mode='item')
item_based_recommender.predict(user_id="741", item_id="1088", k=10)
item_based_recommender.predict(user_id="5378", item_id="1088", k=2)

# User-based collaborative filtering
user_based_recommender = CollaborativeFilteringRecommender(data=bewertungsmatrix_df, mode='user')
user_based_recommender.predict(user_id="741", item_id="104", calculation_variety="unweighted", k=3)
user_based_recommender.predict(user_id="5378", item_id="1073", k=2)

# Content-Based Recommender
content_based_recommender = ContentBasedRecommender(item_profile=itemprofile_df, user_ratings=user_ratings_df)
content_based_recommender.predict(user_id="741", item_id="104")

# Hybrid recommender
hybrid_recommender = HybridRecommender(item_profile=itemprofile_df, user_ratings=user_ratings_df)
hybrid_recommender.predict(user_id="741", item_id="1088")

# Deep learning-based recommender
dl_recommender = DeepLearningRecommender(item_profile=itemprofile_df, trainingdata=user_ratings_df, testdata=testdata_df)
dl_recommender.predict(user_id="741", item_id="1088")

# Hypertuned Deep Learning recommender
hbdl_recommender = HyperparamOptimizedDeepLearningRecommender(item_profile=itemprofile_df, trainingdata=user_ratings_df, testdata=testdata_df)
hbdl_recommender.predict(user_id="741", item_id="1088")
```

### Automated Evaluation (MAE)
```python
tests = [
    Test(name="UserBased_1_cosine", type="collaborative_filtering", mode="user", k_value=4, metric="cosine", calculation_variety="weighted"),
    Test(name="deep_learning", type="deep_learning"),
]

eval_data_path = ...

tester = MAETester(
    tests=tests,
    test_data_path="data/Testdaten_FlixNet.csv",
    data_path="data/Bewertungsmatrix_FlixNet.csv",
    eval_data_path=eval_data_path,
    ratings="data/Ratings_FlixNet.csv",
    item_profile_path="data/Itemprofile_FlixNet.csv",
)
df = tester.run_tests()
```

## License
This project is under the MIT License.