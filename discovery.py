
class EDA:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Import PowerTransformer
    from sklearn.preprocessing import PowerTransformer

    def __init__(self, this_df):


        print()
        print("****************************************************************")
        print("EDA | Start")
        print("****************************************************************")

        print("PRICE: Log transformation to improve skewed data")
        #   https://campus.datacamp.com/courses/preprocessing-for-machine-learning-in-python/standardizing-data?ex=4
        self.do_log_transformation_using_np(this_df)

        print("****************************************************************")
        print("EDA | End")
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
            # use the pands .corr() function to compute pairwise correlations for the dataframe
            corr = this_df.corr()
            # visualise the data with seaborn
            mask = self.np.triu(self.np.ones_like(corr, dtype=self.np.bool))
            self.sns.set_style(style='white')
            f, ax = self.plt.subplots(figsize=(11, 9))
            cmap = self.sns.diverging_palette(10, 250, as_cmap=True)
            self.sns.heatmap(corr, mask=mask, cmap=cmap,
                    square=True,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
            self.plt.savefig("./Output/Correlation_plot.png")
            self.plt.show()


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
        self.plt.show()
