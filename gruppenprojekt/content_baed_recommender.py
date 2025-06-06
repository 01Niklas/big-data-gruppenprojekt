import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


class ContentBasedRecommender:
    def __init__(self, item_profile: pd.DataFrame, user_ratings: pd.DataFrame, id_column: str, feature_columns: list, k: int = 3):
        self.item_profile = item_profile
        self.user_ratings = user_ratings
        self.id_column = id_column
        self.feature_columns = feature_columns
        self.k = k
        self.tfidf_matrix = None
        self._preprocess_data()

    def _preprocess_data(self):
        # Ensure all IDs are strings for consistency
        self.item_profile[self.id_column] = self.item_profile[self.id_column].astype(str)
        self.user_ratings["item_ID"] = self.user_ratings["item_ID"].astype(str)
        self.user_ratings["user_ID"] = self.user_ratings["user_ID"].astype(str)

        # Convert feature columns to strings and create a TF-IDF matrix
        self.item_profile[self.feature_columns] = self.item_profile[self.feature_columns].astype(str)
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(
            self.item_profile[self.feature_columns].fillna("").agg(" ".join, axis=1)
        )

    def predict(self, user_id: str, item_id: str) -> float:
        # Check if the user exists in the ratings data
        if user_id not in self.user_ratings["user_ID"].values:
            raise ValueError(f"User-ID {user_id} not found.")

        # Check if the item exists in the item profile
        if item_id not in self.item_profile[self.id_column].values:
            raise ValueError(f"Item-ID {item_id} not found.")

        # Filter items rated by the user
        rated_items = self.user_ratings[self.user_ratings["user_ID"] == user_id]
        rated_item_ids = rated_items["item_ID"].values  # Get IDs of rated items
        rated_item_indices = self.item_profile[self.item_profile[self.id_column].isin(rated_item_ids)].index

        # Create a filtered TF-IDF matrix for the rated items
        filtered_tfidf_matrix = self.tfidf_matrix[rated_item_indices]

        # Use k-NN to find similar items
        knn = NearestNeighbors(metric="cosine", algorithm="brute")  # Initialize k-NN with cosine similarity
        knn.fit(filtered_tfidf_matrix)  # Fit the model on the filtered TF-IDF matrix
        item_index = self.item_profile[self.item_profile[self.id_column] == item_id].index[0]  # Get the index of the target item
        distances, indices = knn.kneighbors(self.tfidf_matrix[item_index], n_neighbors=self.k + 1)

        # Collect ratings for similar items
        similar_items = indices.flatten()[1:]
        similar_ratings = []
        for similar_item in similar_items:
            # Get the ID of the similar item
            similar_item_id = self.item_profile.iloc[rated_item_indices[similar_item]][self.id_column]
            # Find the user's rating for the similar item
            user_rating = self.user_ratings[
                (self.user_ratings["user_ID"] == user_id) & (self.user_ratings["item_ID"] == similar_item_id)
            ]
            if not user_rating.empty:
                similar_ratings.append(user_rating["rating"].values[0])

        return sum(similar_ratings) / len(similar_ratings)


if __name__ == '__main__':
    itemprofile = pd.read_csv("./data/Itemprofile_FlixNet.csv")
    user_ratings = pd.read_csv("./data/Ratings_FlixNet.csv")

    predictor = ContentBasedRecommender(
        item_profile=itemprofile,
        user_ratings=user_ratings,
        id_column="item_ID",
        feature_columns=["title", "original_language", "runtime"],
        k=3
    )


    predicted_rating = predictor.predict(user_id="4160", item_id="593")  # actual 5.0
    print(f"Predicted rating: {predicted_rating}")