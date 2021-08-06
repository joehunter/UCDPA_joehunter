
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
        from sklearn.ensemble import GradientBoostingRegressor
        import numpy as np

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
        param_grid = {'learning_rate': learning_rate,
                      'alpha': alpha,
                      'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      }

        print("Should see param grid next?")
        print(param_grid)

        # Initialize and fit the model.
        model = GradientBoostingRegressor() #RandomForestRegressor()
        model = RandomizedSearchCV(model, param_grid, cv=3)
        model.fit(X_train_sample, y_train_sample)

        # get the best parameters
        best_params = model.best_params_
        print(best_params)



