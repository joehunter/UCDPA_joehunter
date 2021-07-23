

class Lasso:

    def __init__(self, this_df):
        # Import Lasso
        from sklearn.linear_model import Lasso
        import numpy as np
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt

        # define training and test set
        X_train, X_test, y_train, y_test = train_test_split(this_df, this_df['Price'], test_size=0.2, random_state=42)


        #https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-2?ex=12

        # Create arrays for features and target variable
        y = this_df['Price']
        X = this_df['Postcode']

        # Print the dimensions of y and X before reshaping
        print("Dimensions of y before reshaping: ", y.shape)
        print("Dimensions of X before reshaping: ", X.shape)

        # Reshape X and y
        y_reshaped = y.values.reshape(-1, 1)
        X_reshaped = X.values.reshape(-1, 1)

        # Instantiate a lasso regressor: lasso
        lasso = Lasso(alpha=0.4, normalize=True)

        # Fit the regressor to the data
        lasso.fit(X_reshaped, y_reshaped)

        # Compute and print the coefficients
        lasso_coef = lasso.fit(X_reshaped, y_reshaped).coef_
        print('Lasso coefficient...')
        print(lasso_coef)

        # Plot the coefficients
        plt.plot(range(len(this_df.columns)), lasso_coef)
        plt.xticks(range(len(this_df.columns)), this_df.columns.values, rotation=60)
        plt.margins(0.02)
        plt.show()


