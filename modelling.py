class RunModels:

    def __init__(self, this_df):

        print()
        print("****************************************************************")
        print("@Modelling | Start")
        print("****************************************************************")

        print("\nTrain models now for best score...this will take some time to run...")
        self.cross_validate_models(this_df)

        print("\n****************************************************************")
        print("@Modelling | End")
        print("****************************************************************")


    def cross_validate_models(self, df):

        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import ElasticNet
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.preprocessing import MinMaxScaler
        import operator


        drop_list = ['Price', 'Price_LG', 'high_price', 'low_price', 'Price_no_NA', 'Date', 'Lattitude', 'Longtitude']
        X = df.drop(drop_list, axis=1)
        Y = df["Price_LG"]

        # https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=9
        scaler = MinMaxScaler().fit(X)
        scaled_X = scaler.transform(X)

        seed = 9
        test_size = 0.20

        X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size=test_size, random_state=seed)


        # Use dictionary to store each model to be trained
        models = {}
        models["Linear"] = LinearRegression()
        models["Lasso"] = Lasso()
        models["ElasticNet"] = ElasticNet()
        models["KNN"] = KNeighborsRegressor()
        models["DecisionTree"] = DecisionTreeRegressor()
        models["AdaBoost"] = AdaBoostRegressor()
        models["GradientBoost"] = GradientBoostingRegressor()
        models["RandomForest"] = RandomForestRegressor()
        models["ExtraTrees"] = ExtraTreesRegressor()

        # 10-fold cross validation for each model
        folds = 10
        metric = "neg_mean_squared_error"

        model_results = []
        model_names = []
        rate_scores = {}

        for model_name in models:
            model = models[model_name]
            k_fold = KFold(n_splits=folds, random_state=seed, shuffle=True)

            #   https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-2?ex=8
            results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)

            rate_scores[model_name] = round(results.mean(), 3)
            model_results.append(results)
            model_names.append(model_name)
            print("...{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

        # Rank scores in descending order
        ranked_model_scores = sorted(rate_scores.items(), key=operator.itemgetter(1), reverse=True)

        # Print in descending order each model and its respective scores
        print("\n{:<6} {:<15} {:<10}".format('RANK', 'MODEL', 'SCORE'))
        rank = 0

        for r in ranked_model_scores:
            rank += 1
            print("{:6} {:<15} {:<10}".format(rank, r[0], r[1]))

