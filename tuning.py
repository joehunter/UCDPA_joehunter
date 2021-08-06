
class RunModels:

    def __init__(self, this_df):

        print()
        print("****************************************************************")
        print("@Modelling | Start")
        print("****************************************************************")


    def train_model(self, df):
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
        from sklearn.metrics import mean_squared_error

        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_val_score
        from sklearn import linear_model
        import numpy as np
        import matplotlib.pyplot as plt
        import operator

        # drop_list = ["Price_LG", "Date"]
        # X = df.drop(drop_list, axis=1)

        # X = df[['Rooms', 'Distance', 'Postcode', 'Landsize', 'BuildingArea', 'YearBuilt']]
        drop_list = ['Price', 'Price_LG', 'high_price', 'low_price', 'Price_no_NA', 'Date', 'Lattitude',
                     'Longtitude']
        X = df.drop(drop_list, axis=1)
        Y = df["Price_LG"]
        print(X.shape)
        print(Y.shape)

        scaler = self.MinMaxScaler().fit(X)
        scaled_X = scaler.transform(X)

        seed = 9
        test_size = 0.20

        X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size=test_size, random_state=seed)

        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)

        # user variables to tune
        folds = 10
        metric = "neg_mean_squared_error"

        # hold different regression models in a single dictionary
        models = {}
        # models["Linear"] = LinearRegression()
        # models["Lasso"] = Lasso()
        # models["ElasticNet"] = ElasticNet()
        # models["KNN"] = KNeighborsRegressor()
        # models["DecisionTree"] = DecisionTreeRegressor()
        # models["AdaBoost"] = AdaBoostRegressor()
        models["GradientBoost"] = GradientBoostingRegressor()
        models["RandomForest"] = RandomForestRegressor()
        # models["ExtraTrees"] = ExtraTreesRegressor()

        # 10-fold cross validation for each model
        model_results = []
        model_names = []
        rate_scores = {}

        for model_name in models:
            model = models[model_name]
            #   ValueError: Setting a random_state has no effect since shuffle is False.
            #   You should leave random_state to its default (None), or set shuffle=True.
            k_fold = KFold(n_splits=folds, random_state=seed, shuffle=True)

            lasso = linear_model.Lasso()
            results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)

            rate_scores[model_name] = round(results.mean(), 3)
            model_results.append(results)
            model_names.append(model_name)
            print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

        print(sorted(rate_scores.items(), key=operator.itemgetter(1), reverse=True))

        # create and fit the best regression model
        best_model = GradientBoostingRegressor(random_state=seed)
        best_model.fit(X_train, Y_train)

        # make predictions using the model
        predictions = best_model.predict(X_test)
        print("[INFO] MSE : {}".format(round(mean_squared_error(Y_test, predictions), 3)))

        # plot model's feature importance
        feature_importance = best_model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, X.columns.values[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.savefig("feature_importance.png")
        plt.clf()
        plt.close()


