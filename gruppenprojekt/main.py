from gruppenprojekt.mae_tester import MAETester, Test

tests = [
    Test(name="UserBased_1", type="collaborative_filtering", mode="user", k_value=2, metric="cosine", calculation_variety="weighted"),
    Test(name="UserBased_2", type="collaborative_filtering", mode="user", k_value=3, metric="cosine", calculation_variety="weighted"),
    Test(name="UserBased_3", type="collaborative_filtering", mode="user", k_value=4, metric="cosine", calculation_variety="weighted"),
    Test(name="UserBased_4", type="collaborative_filtering", mode="user", k_value=5, metric="cosine", calculation_variety="weighted"),

    Test(name="UserBased_5", type="collaborative_filtering", mode="user", k_value=2, metric="cosine", calculation_variety="unweighted"),
    Test(name="UserBased_6", type="collaborative_filtering", mode="user", k_value=3, metric="cosine", calculation_variety="unweighted"),
    Test(name="UserBased_7", type="collaborative_filtering", mode="user", k_value=4, metric="cosine", calculation_variety="unweighted"),
    Test(name="UserBased_8", type="collaborative_filtering", mode="user", k_value=5, metric="cosine", calculation_variety="unweighted"),

    Test(name="ItemBased_1", type="collaborative_filtering", mode="item", k_value=2, metric="cosine", calculation_variety="weighted"),
    Test(name="ItemBased_2", type="collaborative_filtering", mode="item", k_value=3, metric="cosine", calculation_variety="weighted"),
    Test(name="ItemBased_3", type="collaborative_filtering", mode="item", k_value=4, metric="cosine", calculation_variety="weighted"),
    Test(name="ItemBased_4", type="collaborative_filtering", mode="item", k_value=5, metric="cosine", calculation_variety="weighted"),

    Test(name="ItemBased_5", type="collaborative_filtering", mode="item", k_value=2, metric="cosine", calculation_variety="unweighted"),
    Test(name="ItemBased_6", type="collaborative_filtering", mode="item", k_value=3, metric="cosine", calculation_variety="unweighted"),
    Test(name="ItemBased_7", type="collaborative_filtering", mode="item", k_value=4, metric="cosine", calculation_variety="unweighted"),
    Test(name="ItemBased_8", type="collaborative_filtering", mode="item", k_value=5, metric="cosine", calculation_variety="unweighted"),

    Test(name="ContentBased_1", type="content_based", first_k_value=3),
    Test(name="ContentBased_2", type="content_based", first_k_value=4),
    Test(name="ContentBased_3", type="content_based", first_k_value=5),
    Test(name="ContentBased_4", type="content_based", first_k_value=6),
    Test(name="ContentBased_5", type="content_based", first_k_value=7),
    Test(name="ContentBased_6", type="content_based", first_k_value=8),
    Test(name="ContentBased_7", type="content_based", first_k_value=9),
    Test(name="ContentBased_8", type="content_based", first_k_value=10), # --> best in our tests (for only content based) based on the test data

    Test(name="Hybrid_1", type="hybrid", mode="user", k_value=3, metric="cosine", calculation_variety="weighted", alpha=0.5),
    Test(name="Hybrid_2", type="hybrid", mode="user", k_value=3, metric="cosine", calculation_variety="weighted", alpha=0.75),
    Test(name="Hybrid_3", type="hybrid", mode="user", k_value=3, metric="cosine", calculation_variety="weighted", alpha=0.25),
    Test(name="Hybrid_4", type="hybrid", mode="user", k_value=3, metric="cosine", calculation_variety="unweighted", alpha=0.5),
    Test(name="Hybrid_5", type="hybrid", mode="user", k_value=3, metric="cosine", calculation_variety="unweighted", alpha=0.75),
    Test(name="Hybrid_6", type="hybrid", mode="user", k_value=3, metric="cosine", calculation_variety="unweighted", alpha=0.25),

    Test(name="Hybrid_7", type="hybrid", mode="item", k_value=3, metric="cosine", calculation_variety="weighted", alpha=0.5),
    Test(name="Hybrid_8", type="hybrid", mode="item", k_value=3, metric="cosine", calculation_variety="weighted", alpha=0.75),
    Test(name="Hybrid_9", type="hybrid", mode="item", k_value=3, metric="cosine", calculation_variety="weighted", alpha=0.25),
    Test(name="Hybrid_10", type="hybrid", mode="item", k_value=3, metric="cosine", calculation_variety="unweighted", alpha=0.5),
    Test(name="Hybrid_11", type="hybrid", mode="item", k_value=3, metric="cosine", calculation_variety="unweighted", alpha=0.75),
    Test(name="Hybrid_12", type="hybrid", mode="item", k_value=3, metric="cosine", calculation_variety="unweighted", alpha=0.25),

    Test(name="Hybrid_13", type="hybrid", mode="user", k_value=3, second_k_value=10, metric="cosine", calculation_variety="weighted", alpha=0.5),
    Test(name="Hybrid_14", type="hybrid", mode="user", k_value=3, second_k_value=10, metric="cosine", calculation_variety="weighted", alpha=0.75),
    Test(name="Hybrid_14", type="hybrid", mode="user", k_value=3, second_k_value=10, metric="cosine", calculation_variety="weighted", alpha=0.25),
    Test(name="Hybrid_15", type="hybrid", mode="user", k_value=3, second_k_value=10, metric="cosine", calculation_variety="unweighted", alpha=0.5),
    Test(name="Hybrid_16", type="hybrid", mode="user", k_value=3, second_k_value=10, metric="cosine", calculation_variety="unweighted", alpha=0.75),
    Test(name="Hybrid_17", type="hybrid", mode="user", k_value=3, second_k_value=10, metric="cosine", calculation_variety="unweighted", alpha=0.25),

    Test(name="Hybrid_18", type="hybrid", mode="item", k_value=3, second_k_value=10, metric="cosine", calculation_variety="weighted", alpha=0.5),
    Test(name="Hybrid_19", type="hybrid", mode="item", k_value=3, second_k_value=10, metric="cosine", calculation_variety="weighted", alpha=0.75),
    Test(name="Hybrid_20", type="hybrid", mode="item", k_value=3, second_k_value=10, metric="cosine", calculation_variety="weighted", alpha=0.25),
    Test(name="Hybrid_21", type="hybrid", mode="item", k_value=3, second_k_value=10, metric="cosine", calculation_variety="unweighted", alpha=0.5),
    Test(name="Hybrid_22", type="hybrid", mode="item", k_value=3, second_k_value=10, metric="cosine", calculation_variety="unweighted", alpha=0.75),
    Test(name="Hybrid_23", type="hybrid", mode="item", k_value=3, second_k_value=10, metric="cosine", calculation_variety="unweighted", alpha=0.25),
]
if __name__ == '__main__':


    tester = MAETester(
        tests=tests,
        test_data_path="data/Testdaten_FlixNet.csv",
        data_path="data/Bewertungsmatrix_FlixNet.csv",
        user_ratings="data/Ratings_FlixNet.csv",
        item_profile_path="data/Itemprofile_FlixNet.csv",
    )
    df = tester.run_tests()

    df.head()
