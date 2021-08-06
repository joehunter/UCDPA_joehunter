
class TuneModel:

    def __init__(self, this_df):


        print("\n****************************************************************")
        print("@Tuning | Start")
        print("****************************************************************")

        self.tune_model(this_df)

        print("\n****************************************************************")
        print("@Tuning | End")
        print("****************************************************************")

    def tune_model(self, this_df):

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error

        drop_list = ['Price', 'Price_LG', 'high_price', 'low_price', 'Price_no_NA', 'Date', 'Lattitude', 'Longtitude']
        X = this_df.drop(drop_list, axis=1)
        Y = this_df["Price_LG"]


        test_size = 0.20
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

        test_size = 0.5
        X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        # define our parameter ranges
        learning_rate = [0.01]
        alpha = [0.01, 0.03, 0.05, 0.1, 0.3, 0.9]
        n_estimators = [int(x) for x in np.linspace(start=10, stop=500, num=4)]
        max_depth = [int(x) for x in np.linspace(start=3, stop=15, num=4)]
        max_depth.append(None)
        min_samples_split = [int(x) for x in np.linspace(start=2, stop=5, num=4)]
        min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=4, num=4)]
        max_features = ['auto', 'sqrt']

        # Create the random grid
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(random_grid)

        # Initialize and fit the model.
        model = RandomForestRegressor()
        model = RandomizedSearchCV(model, random_grid, cv=3)
        model.fit(X_train_sample, y_train_sample)

        # get the best parameters
        best_params = model.best_params_
        print(best_params)

        # refit model with best parameters
        model_best = RandomForestRegressor(**best_params)
        model_best.fit(X_train, y_train)
        y_pred = model_best.predict(X_test)


        feature_importance = model_best.feature_importances_

        # Make importances relative to max importance.
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5

        plt.subplot(1,2,2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')

        plt.yticks(pos, X.columns.values[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()

        # sort top features
        top_features = np.where(feature_importance > 20)
        top_features = X.columns[top_features].ravel()
        print(top_features)


        # Display.
        print('Optimized Gradient Boosting Regressor')
        print('\nR-squared training set:')
        print(model_best.score(X_train, y_train))
        print('\nMean absolute error training set: ')
        print(mean_absolute_error(y_train, model_best.predict(X_train)))
        print('\nMean squared error training set: ')
        print(mean_squared_error(y_train, model_best.predict(X_train)))

        print('\n\nR-squared test set:')
        print(model_best.score(X_test, y_test))
        print('\nMean absolute error test set: ')
        print(mean_absolute_error(y_test, y_pred))
        print('\nMean squared error test set: ')
        print(mean_squared_error(y_test, y_pred))

        # top features
        print('\nTop indicators:')
        print(top_features)