from gruppenprojekt.mae_tester import MAETester, Test

if __name__ == '__main__':
    tests = [
        Test(name="UserBased_1", mode="user", k_value=2, metric="cosine", calculation_variety="weighted"),
        Test(name="UserBased_2", mode="user", k_value=3, metric="cosine", calculation_variety="weighted"),
        Test(name="UserBased_3", mode="user", k_value=4, metric="cosine", calculation_variety="weighted"),
        Test(name="UserBased_4", mode="user", k_value=5, metric="cosine", calculation_variety="weighted"),

        Test(name="UserBased_5", mode="user", k_value=2, metric="cosine", calculation_variety="unweighted"),
        Test(name="UserBased_6", mode="user", k_value=3, metric="cosine", calculation_variety="unweighted"),
        Test(name="UserBased_7", mode="user", k_value=4, metric="cosine", calculation_variety="unweighted"),
        Test(name="UserBased_8", mode="user", k_value=5, metric="cosine", calculation_variety="unweighted"),

        Test(name="ItemBased_1", mode="item", k_value=2, metric="cosine", calculation_variety="weighted"),
        Test(name="ItemBased_2", mode="item", k_value=3, metric="cosine", calculation_variety="weighted"),
        Test(name="ItemBased_3", mode="item", k_value=4, metric="cosine", calculation_variety="weighted"),
        Test(name="ItemBased_4", mode="item", k_value=5, metric="cosine", calculation_variety="weighted"),

        Test(name="ItemBased_5", mode="item", k_value=2, metric="cosine", calculation_variety="unweighted"),
        Test(name="ItemBased_6", mode="item", k_value=3, metric="cosine", calculation_variety="unweighted"),
        Test(name="ItemBased_7", mode="item", k_value=4, metric="cosine", calculation_variety="unweighted"),
        Test(name="ItemBased_8", mode="item", k_value=5, metric="cosine", calculation_variety="unweighted"),
    ]

    tester = MAETester(
        tests=tests,
        test_data_path="data/Testdaten_FlixNet.csv",
        data_path="data/Bewertungsmatrix_FlixNet.csv"
    )
    tester.run_tests()
