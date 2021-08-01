
class EDA:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Import PowerTransformer
    from sklearn.preprocessing import PowerTransformer



    def __init__(self, this_df):


        print()
        print("****************************************************************")
        print("@EDA | Start")
        print("****************************************************************")

        print('FEATURES: Number of features? %s' % this_df.shape[1])
        print('ROWS: Number of rows? %s' % this_df.shape[0])


        print('\nSAMPLES: Head & Tail')
        print(this_df.head().append(this_df.tail()))

        print('\nINFO: .info()')
        print(this_df.info())

        print('\nDESCRIBE: Transpose .describe()')
        print(this_df.describe().transpose())

        print('\nCORRELATE: Create correlation heatmap')
        self.correlation_heatmap(this_df)


        print("\nPRICE: Log transformation to improve skewed data")
        #   https://campus.datacamp.com/courses/preprocessing-for-machine-learning-in-python/standardizing-data?ex=4
        self.do_log_transformation_using_np(this_df)

        print("****************************************************************")
        print("@EDA | End")
        print("****************************************************************")

    def do_log_transformation_using_power_transformer(self, this_df):
    #   https://campus.datacamp.com/courses/feature-engineering-for-machine-learning-in-python/conforming-to-statistical-assumptions?ex=8

    #   Getting a warning with this method!
    #   RuntimeWarning: divide by zero encountered in log loglike = -n_samples / 2 * np.log(x_trans.var())

        # Instantiate PowerTransformer
        pow_trans = self.PowerTransformer()

        # Train the transform on the data
        this_df['Price_no_NA'] = this_df.Price.dropna()
        pow_trans.fit(this_df[['Price_no_NA']])

        # Apply the power transform to the data
        this_df['Price_LG'] = pow_trans.transform(this_df[['Price_no_NA']])

        log_price_mean = this_df['Price_LG'].mean()
        log_price_std = this_df['Price_LG'].std()

        # view log(price) data
        this_df[['Price_no_NA', 'Price_LG']].hist(bins=20)
        self.plt.axvline((log_price_mean + log_price_std), color='k', linestyle='--')
        self.plt.axvline((log_price_mean - log_price_std), color='k', linestyle='--')
        self.plt.axvline(log_price_mean, color='k', linestyle='-')
        self.plt.title('log(Price)')
        self.plt.show()


    def correlation_heatmap(self, this_df):
        f, ax = self.plt.subplots(figsize=(10, 10))
        self.sns.heatmap(this_df.corr(), annot=True, linewidths=5, fmt='.1f', ax=ax, cmap='Reds')
        self.plt.savefig("./Output/Correlation_plot.png")
        #self.plt.show()


    def do_log_transformation_using_np(self, this_df):
        #   Caveat limitations of this approach
        #   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120293/

        this_df['Price_no_NA'] = this_df.Price.dropna()
        this_df['Price_LG'] = self.np.log1p(this_df.Price.dropna())

        log_price_mean = this_df['Price_LG'].mean()
        log_price_std = this_df['Price_LG'].std()

        # contrast before and after log transform
        this_df[['Price_no_NA', 'Price_LG']].hist(bins=20)
        self.plt.axvline((log_price_mean + log_price_std), color='k', linestyle='--')
        self.plt.axvline((log_price_mean - log_price_std), color='k', linestyle='--')
        self.plt.axvline(log_price_mean, color='k', linestyle='-')
        self.plt.title('Log Transform of Price Feature')
        self.plt.savefig("./Output/log_transformation_using_np.png")
        #self.plt.show()
