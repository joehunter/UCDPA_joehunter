

class Linear:

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor


    def __init__(self, this_df):
        # https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-2?ex=7

        from sklearn.model_selection import train_test_split

        drop_list = ['Price', 'Price_LG', 'high_price', 'low_price', 'Price_no_NA', 'Date', 'Lattitude', 'Longtitude']
        X = this_df.drop(drop_list, axis=1)
        y = this_df['Price_LG']

        # define training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.do_linear_regression(X_train, X_test, y_train, y_test)

        self.do_gradient_boosting_regressor(X_train, X_test, y_train, y_test)

    def do_linear_regression(self, X_train, X_test, y_train, y_test):
        # determine which model to use
        OLS = self.LinearRegression()


        OLS.fit(X_train, y_train)
        y_pred = OLS.predict(X_test)

        # Display.
        print('Linear Regression')
        print('\nR-squared training set:')
        print(OLS.score(X_train, y_train))

        print('\nR-squared test set:')
        print(OLS.score(X_test, y_test))


    def do_gradient_boosting_regressor(self, X_train, X_test, y_train, y_test):
        from sklearn.metrics import mean_squared_error

        # scale X_train values
        scaler = self.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(self.np.float64))
        X_test_scaled = scaler.transform(X_test.astype(self.np.float64))

        GBRT_scaled = self.GradientBoostingRegressor(max_depth=2, n_estimators=120)
        GBRT_scaled.fit(X_train_scaled, y_train)

        errors = [mean_squared_error(y_test, y_pred) for y_pred in GBRT_scaled.staged_predict(X_test_scaled)]
        best_n_estimators = self.np.argmin(errors)

        GBRT_scaled_best = self.GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
        GBRT_scaled_best.fit(X_train_scaled, y_train)
        y_pred = GBRT_scaled_best.predict(X_test_scaled)

        # Display
        print('Scaled Gradient Boosting Regressor')
        print('\nR-squared training set:')
        print(GBRT_scaled_best.score(X_train_scaled, y_train))

        print('\nR-squared test set:')
        print(GBRT_scaled_best.score(X_test_scaled, y_test))