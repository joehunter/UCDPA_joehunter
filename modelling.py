class RunModels:

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor

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

        self.do_ols(X_train, X_test, y_train, y_test)
        self.do_ols_scaled(X_test_scaled, X_train_scaled, y_train, y_test)
        self.do_random_forest_regression(X_train, X_test, y_train, y_test)
        self.do_random_forest_regression_scaled(X_test_scaled, X_train_scaled, y_train, y_test)
        self.do_gradient_boosting_regressor(X_train, X_test, y_train, y_test)
        self.do_gradient_boosting_regressor_scaled(X_test_scaled, X_train_scaled, y_train, y_test)

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
