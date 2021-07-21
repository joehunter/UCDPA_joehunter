

class Linear:

    import pandas
    import numpy as np
    from sklearn import linear_model

    def __init__(self, this_df):

        this_df = this_df.reset_index()

        X = this_df[['Distance', 'Rooms']]
        y = this_df[['Price']]

        #print(self.np.isnan(X))
        print(self.np.where(self.np.isnan(X)))
        #print(self.np.nan_to_num(X))

        #linear_regression = self.linear_model.LinearRegression()
        #linear_regression.fit(X, y)

        # predict the price where house is 10KM from CBD and number of rooms is 5
        #predicted_price = linear_regression.predict([[10, 5]])

        #print(predicted_price)