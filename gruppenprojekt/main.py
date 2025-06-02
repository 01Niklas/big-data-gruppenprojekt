# this is just for testing
import pandas as pd

from gruppenprojekt.user_based_recommender import UserBasedRecommender

if __name__ == '__main__':
    bewertungsmatrix_df = pd.read_csv('data/Bewertungsmatrix_FlixNet.csv')
    r = UserBasedRecommender(data=bewertungsmatrix_df)
    r.predict(user_id="741", item_id="104", calculation_variety="ungewichtet", k=3)
    r.predict(user_id="5378", item_id="1073", k=5)
