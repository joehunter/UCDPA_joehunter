class RunModels:

    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler


    def __init__(self, this_df):

        from sklearn.model_selection import train_test_split

        drop_list = ['Price', 'Price_LG', 'high_price', 'low_price', 'Price_no_NA', 'Date', 'Lattitude', 'Longtitude']
        X = this_df.drop(drop_list, axis=1)
        y = this_df['Price_LG']

        # define training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale here
        # https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=9
        scaler = self.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(self.np.float64))
        X_test_scaled = scaler.transform(X_test.astype(self.np.float64))


        print()
        print("****************************************************************")
        print("@Modelling | Start")
        print("****************************************************************")

        # self.do_ols(X_train, X_test, y_train, y_test)
        # self.do_ols_scaled(X_test_scaled, X_train_scaled, y_train, y_test)
        # self.do_random_forest_regression(X_train, X_test, y_train, y_test)
        # self.do_random_forest_regression_scaled(X_test_scaled, X_train_scaled, y_train, y_test)
        # self.do_gradient_boosting_regressor(X_train, X_test, y_train, y_test)
        # self.do_gradient_boosting_regressor_scaled(X_test_scaled, X_train_scaled, y_train, y_test)

        self.train_model(this_df)

        print("\n****************************************************************")
        print("@Modelling | End")
        print("****************************************************************")


    def do_ols(self, X_train, X_test, y_train, y_test):
        OLS = self.LinearRegression()

        OLS.fit(X_train, y_train)
        y_pred = OLS.predict(X_test)

        print('\nLinear Regression')
        print('...R-squared training set: ', OLS.score(X_train, y_train) * 100, '%')
        print('...R-squared test set: {}'.format(OLS.score(X_test, y_test)))


    def do_ols_scaled(self, X_test_scaled, X_train_scaled, y_train, y_test):
        OLS_scaled = self.LinearRegression()
        OLS_scaled.fit(X_train_scaled, y_train)
        y_pred = OLS_scaled.predict(X_test_scaled)

        print('\nLinear Regression Scaled')
        print('...R-squared training set: {}'.format(OLS_scaled.score(X_train_scaled, y_train)))
        print('...R-squared test set: {}'.format(OLS_scaled.score(X_test_scaled, y_test)))


    def do_random_forest_regression(self, X_train, X_test, y_train, y_test):
        RF = self.RandomForestRegressor(n_estimators=10)
        RF.fit(X_train, y_train)
        y_pred = RF.predict(X_test)

        print('\nRandom Forest Regression')
        print('...R-squared training set: {}'.format(RF.score(X_train, y_train)))
        print('...R-squared test set: {}'.format(RF.score(X_test, y_test)))


    def do_random_forest_regression_scaled(self, X_test_scaled, X_train_scaled, y_train, y_test):
        RF_scaled = self.RandomForestRegressor(n_estimators=10)
        RF_scaled.fit(X_train_scaled, y_train)
        y_pred = RF_scaled.predict(X_test_scaled)

        print('\nRandom Forest Regression Scaled')
        print('...R-squared training set: {}'.format(RF_scaled.score(X_train_scaled, y_train)))
        print('...R-squared test set: {}'.format(RF_scaled.score(X_test_scaled, y_test)))



    def do_gradient_boosting_regressor(self, X_train, X_test, y_train, y_test):
        from sklearn.metrics import mean_squared_error
        GBRT = self.GradientBoostingRegressor(max_depth=2, n_estimators=120)
        GBRT.fit(X_train, y_train)

        errors = [mean_squared_error(y_test, y_pred)
                  for y_pred in GBRT.staged_predict(X_test)]
        best_n_estimators = self.np.argmin(errors)

        GBRT_best = self.GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
        GBRT_best.fit(X_train, y_train)
        y_pred = GBRT_best.predict(X_test)

        print('\nGradient Boosting Regressor')
        print('...R-squared training set: {}'.format(GBRT_best.score(X_train, y_train)))
        print('...R-squared test set: {}'.format(GBRT_best.score(X_test, y_test)))



    def do_gradient_boosting_regressor_scaled(self, X_test_scaled, X_train_scaled, y_train, y_test):
        from sklearn.metrics import mean_squared_error


        GBRT_scaled = self.GradientBoostingRegressor(max_depth=2, n_estimators=120)
        GBRT_scaled.fit(X_train_scaled, y_train)

        errors = [mean_squared_error(y_test, y_pred) for y_pred in GBRT_scaled.staged_predict(X_test_scaled)]
        best_n_estimators = self.np.argmin(errors)

        GBRT_scaled_best = self.GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
        GBRT_scaled_best.fit(X_train_scaled, y_train)
        y_pred = GBRT_scaled_best.predict(X_test_scaled)

        print('\nScaled Gradient Boosting Regressor')
        print('...R-squared training set: {}'.format(GBRT_scaled_best.score(X_train_scaled, y_train)))
        print('...R-squared test set: {}'.format(GBRT_scaled_best.score(X_test_scaled, y_test)))


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

        #drop_list = ["Price_LG", "Date"]
        #X = df.drop(drop_list, axis=1)


        #X = df[['Rooms', 'Distance', 'Postcode', 'Landsize', 'BuildingArea', 'YearBuilt']]
        drop_list = ['Price', 'Price_LG', 'high_price', 'low_price', 'Price_no_NA', 'Date', 'Lattitude', 'Longtitude']
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


        print(  sorted(rate_scores.items(), key=operator.itemgetter(1),reverse=True)  )

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
