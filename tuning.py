
class TuneModel:

    """
    This class is used to hyperparameter tune the data

    Methods
    -------
    tune_model(this_df):
        Tune the model using RandomizedSearchCV.
    """

    def __init__(self, this_df):


        print("\n****************************************************************")
        print("@Tuning | Start")
        print("****************************************************************")

        print("...Please note this will take some time to run...")
        self.tune_model(this_df)

        print("\n****************************************************************")
        print("@Tuning | End")
        print("****************************************************************")


    def tune_model(self, this_df):

        '''
          Tune the model using RandomizedSearchCV.

                  Parameters:
                          this_df (dataframe): Pandas Dataframe with cleaned data to model
        '''

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
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


        #
        #   Run in advance of optimizing to determine score...
        #
        RF = RandomForestRegressor(n_estimators=10)
        RF.fit(X_train, y_train)
        y_pred = RF.predict(X_test)

        print('\nPre-Optimized Random Forest Regressor...')
        print("R-squared training set:' {}".format(RF.score(X_train, y_train)))
        print("R-squared test set:' {}".format(RF.score(X_test, y_test)))


        #
        # Define a grid of hyperparameter ranges
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        #

        # The number of trees in the forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # The number of features to consider when looking for the best split
        max_features = ['auto', 'sqrt']
        # The maximum depth of the tree.
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)

        # The minimum number of samples required to split an internal node
        min_samples_split = [2, 5, 10]
        # The minimum number of samples required to be at a leaf node
        min_samples_leaf = [1, 2, 4]
        # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}


        # Initialize and fit the model.
        #   https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/fine-tuning-your-model?ex=11
        model = RandomizedSearchCV(RandomForestRegressor(), random_grid, cv=3)
        model.fit(X_train_sample, y_train_sample)

        # get the best parameters
        best_params = model.best_params_

        # refit model with best parameters
        model_best = RandomForestRegressor(**best_params)
        model_best.fit(X_train, y_train)
        y_pred = model_best.predict(X_test)


        feature_importance = model_best.feature_importances_
        self.plot_feature_importance(feature_importance, X)


        # Print all metrics in output
        print('\nOptimized Random Forest Regressor')
        print("Best Parameters: ", best_params)
        print("\n{:<40} {:<15}".format('METRIC', 'VALUE'))
        print("{:40} {:<15}".format('R-squared training set', model_best.score(X_train, y_train)))
        print("{:40} {:<15}".format('Mean absolute error training set', mean_absolute_error(y_train, model_best.predict(X_train))))
        print("{:40} {:<15}".format('Mean squared error training set', mean_squared_error(y_train, model_best.predict(X_train))))

        print("{:40} {:<15}".format('\nR-squared test set', model_best.score(X_test, y_test)))
        print("{:40} {:<15}".format('Mean absolute error test set', mean_absolute_error(y_test, y_pred)))
        print("{:40} {:<15}".format('Mean squared error test set', mean_squared_error(y_test, y_pred)))



    def plot_feature_importance(self, feature_importance, X):
        '''
          Plots the relative importance of all features in a dataset.

                  Parameters:
                          feature_importance (ndarray):  NumPy ndarray object
                          X (dataframe): Pandas Dataframe for feature names
        '''

        import matplotlib.pyplot as plt
        import numpy as np

        # Make importance relative to max importance.
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5

        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')

        plt.yticks(pos, X.columns.values[sorted_idx])
        plt.title('Feature Importance')
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')

        plt.savefig("./Output/feature_importance.png")
        plt.clf()
        plt.close()

