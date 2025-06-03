# this is just for testing
import pandas as pd

from gruppenprojekt.collaborative_filtering_recommender import CollaborativeFilteringRecommender

if __name__ == '__main__':

    bewertungsmatrix_df = pd.read_csv('data/Bewertungsmatrix_FlixNet.csv')

    item_based_recommender = CollaborativeFilteringRecommender(data=bewertungsmatrix_df, mode='item')
    item_based_recommender.predict(user_id="741", item_id="1088", k=10)
    item_based_recommender.predict(user_id="5378", item_id="1088", k=2)

    user_based_recommender = CollaborativeFilteringRecommender(data=bewertungsmatrix_df, mode='user')
    user_based_recommender.predict(user_id="741", item_id="104", calculation_variety="unweighted", k=3)
    user_based_recommender.predict(user_id="5378", item_id="1073", k=2)
