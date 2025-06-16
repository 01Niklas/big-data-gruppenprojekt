from gruppenprojekt.mae_tester import MAETester, Test

if __name__ == '__main__':

    tests = [
        Test(name="UserBased_1_cosine", type="collaborative_filtering", mode="user", k_value=4, metric="cosine", calculation_variety="weighted"),
        Test(name="UserBased_2_cosine", type="collaborative_filtering", mode="user", k_value=3, metric="cosine", calculation_variety="weighted"),
        Test(name="UserBased_3_cosine", type="collaborative_filtering", mode="user", k_value=4, metric="cosine", calculation_variety="weighted"),
        Test(name="UserBased_4_cosine", type="collaborative_filtering", mode="user", k_value=5, metric="cosine", calculation_variety="weighted"),

        Test(name="ItemBased_1_cosine", type="collaborative_filtering", mode="item", k_value=4, metric="cosine", calculation_variety="weighted"),
        Test(name="ItemBased_2_cosine", type="collaborative_filtering", mode="item", k_value=3, metric="cosine", calculation_variety="weighted"),
        Test(name="ItemBased_3_cosine", type="collaborative_filtering", mode="item", k_value=4, metric="cosine", calculation_variety="weighted"),
        Test(name="ItemBased_4_cosine", type="collaborative_filtering", mode="item", k_value=5, metric="cosine", calculation_variety="weighted"),

        Test(name="UserBased_1_pearson", type="collaborative_filtering", mode="user", k_value=4, metric="pearson", calculation_variety="weighted"),
        Test(name="UserBased_2_pearson", type="collaborative_filtering", mode="user", k_value=3, metric="pearson", calculation_variety="weighted"),
        Test(name="UserBased_3_pearson", type="collaborative_filtering", mode="user", k_value=4, metric="pearson", calculation_variety="weighted"),
        Test(name="UserBased_4_pearson", type="collaborative_filtering", mode="user", k_value=5, metric="pearson", calculation_variety="weighted"),

        Test(name="ItemBased_1_pearson", type="collaborative_filtering", mode="item", k_value=4, metric="pearson", calculation_variety="weighted"),
        Test(name="ItemBased_2_pearson", type="collaborative_filtering", mode="item", k_value=3, metric="pearson", calculation_variety="weighted"),
        Test(name="ItemBased_3_pearson", type="collaborative_filtering", mode="item", k_value=4, metric="pearson", calculation_variety="weighted"),
        Test(name="ItemBased_4_pearson", type="collaborative_filtering", mode="item", k_value=5, metric="pearson", calculation_variety="weighted"),

        Test(name="ContentBased_1", type="content_based", k_value=3),
        Test(name="ContentBased_2", type="content_based", k_value=4),
        Test(name="ContentBased_3", type="content_based", k_value=5),
        Test(name="ContentBased_4", type="content_based", k_value=6),
        Test(name="ContentBased_5", type="content_based", k_value=7),
        Test(name="ContentBased_6", type="content_based", k_value=8),
        Test(name="ContentBased_7", type="content_based", k_value=9),
        Test(name="ContentBased_8", type="content_based", k_value=10),
        Test(name="ContentBased_9", type="content_based", k_value=11),
        Test(name="ContentBased_10", type="content_based", k_value=12),
        Test(name="ContentBased_11", type="content_based", k_value=13),
        Test(name="ContentBased_12", type="content_based", k_value=14),

        Test(name="Hybrid_1", type="hybrid", mode="user", k_value=5, metric="cosine", calculation_variety="weighted", alpha=0.5),
        Test(name="Hybrid_2", type="hybrid", mode="user", k_value=5, metric="cosine", calculation_variety="weighted", alpha=0.75),
        Test(name="Hybrid_3", type="hybrid", mode="user", k_value=5, metric="cosine", calculation_variety="weighted", alpha=0.25),

        Test(name="Hybrid_4", type="hybrid", mode="user", k_value=5, second_k_value=14, metric="cosine", calculation_variety="weighted", alpha=0.5),
        Test(name="Hybrid_5", type="hybrid", mode="user", k_value=5, second_k_value=14, metric="cosine", calculation_variety="weighted", alpha=0.75),
        Test(name="Hybrid_6", type="hybrid", mode="user", k_value=5, second_k_value=14, metric="cosine", calculation_variety="weighted", alpha=0.25),

        Test(name="Hybrid_7", type="hybrid", mode="user", k_value=5, metric="pearson", calculation_variety="weighted", alpha=0.5),
        Test(name="Hybrid_8", type="hybrid", mode="user", k_value=5, metric="pearson", calculation_variety="weighted", alpha=0.75),
        Test(name="Hybrid_9", type="hybrid", mode="user", k_value=5, metric="pearson", calculation_variety="weighted", alpha=0.25),

        Test(name="Hybrid_10", type="hybrid", mode="user", k_value=5, second_k_value=14, metric="pearson", calculation_variety="weighted", alpha=0.5),
        Test(name="Hybrid_11", type="hybrid", mode="user", k_value=5, second_k_value=14, metric="pearson", calculation_variety="weighted", alpha=0.75),
        Test(name="Hybrid_12", type="hybrid", mode="user", k_value=5, second_k_value=14, metric="pearson", calculation_variety="weighted", alpha=0.25),

        Test(name="deep_learning", type="deep_learning"),
    ]

    eval_data_path = "./data/Testdaten_FlixNet.csv"

    tester = MAETester(
        tests=tests,
        test_data_path="data/Testdaten_FlixNet.csv",
        data_path="data/Bewertungsmatrix_FlixNet.csv",
        eval_data_path=eval_data_path,
        ratings="data/Ratings_FlixNet.csv",
        item_profile_path="data/Itemprofile_FlixNet.csv",
    )
    df = tester.run_tests()

    df.head()
