

class Linear:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler



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

        GBRT = self.GradientBoostingRegressor(max_depth=2, n_estimators=120)
        GBRT.fit(X_train, y_train)

        sc = self.StandardScaler()
        #X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        #y_pred = GBRT.staged_predict(X_test)

        errors = self.mean_squared_error(y_test, GBRT.predict(X_test_std))
        best_n_estimators = self.np.argmin(errors)


        GBRT_best = self.GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
        GBRT_best.fit(X_train, y_train)
        y_pred = GBRT_best.predict(X_test)

        # Display
        print('Gradient Boosting Regressor')
        print('\nR-squared training set:')
        print(GBRT_best.score(X_train, y_train))

        print('\nR-squared test set:')
        print(GBRT_best.score(X_test, y_test))