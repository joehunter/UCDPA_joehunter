
class EDA:

    """
        Use this class to perform exploratory data analysis on data frame.

        Methods
        -------
        do_log_transformation_using_power_transformer(self, this_df):
            Do log transformation on Price column using PowerTransformer

        correlation_heatmap(self, this_df):
            Make correlation heatmap using Seaborn

        do_log_transformation_using_np(self, this_df):
            Do log transformation using Numpy

        drop_features(self, this_df):
            Drop features from dataframe

    """

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
        print('...Which feature(s) are highly correlated (> 0.3) to target feature Price?')
        cor = this_df.corr()
        # Correlation with output variable
        cor_target = abs(cor["Price"])
        # Selecting highly correlated features
        relevant_features = cor_target[cor_target > 0.3].sort_values(ascending=False)
        relevant_features = relevant_features.drop(["Price"])
        print(relevant_features)

        print('\nFEATURE SELECTION: Check for features that have a high correlation with each other?')
        self.drop_features(this_df)


        print("\nSKEWNESS & KURTOSIS: Check for each in target feature Price")
        sr = this_df['Price']
        pre_log_kurtosis = sr.kurtosis(skipna=True)
        print("PRICE KURTOSIS: What is the kurtosis of the target feature? : {}".format(pre_log_kurtosis))
        pre_log_skew = sr.skew(skipna=True)
        print("PRICE SKEW: What is the skew of the target feature? : {}".format(pre_log_skew))

        #   If the skewness is less than -1 or greater than 1, the data are highly skewed
        if pre_log_skew > 1 or pre_log_skew < -1:
            print("PRICE: Price data are highly skewed => do Log transformation to improve")
            #   https://campus.datacamp.com/courses/preprocessing-for-machine-learning-in-python/standardizing-data?ex=4
            self.do_log_transformation_using_np(this_df)
            sr = this_df['Price_LG']
            post_log_kurtosis = sr.kurtosis(skipna=True)
            print("...KURTOSIS: What is the KURTOSIS of LOG transform of Price? : {}".format(post_log_kurtosis))
            post_log_skew = sr.skew(skipna=True)
            print("...SKEW: What is the SKEW LOG transform of Price? : {}".format(post_log_skew))


        print("****************************************************************")
        print("@EDA | End")
        print("****************************************************************")



    def do_log_transformation_using_power_transformer(self, this_df):
        '''
        Do log transformation on Price column using PowerTransformer
        #   https://campus.datacamp.com/courses/feature-engineering-for-machine-learning-in-python/conforming-to-statistical-assumptions?ex=8

            Parameters:
            this_df (dataframe): Pandas dataframe
        '''


    #   Getting a warning with this method!
    #   RuntimeWarning: divide by zero encountered in log loglike = -n_samples / 2 * np.log(x_trans.var())

        # Instantiate PowerTransformer
        pow_trans = self.PowerTransformer()

        # Train the transform on the data
        this_df['Price_no_NA'] = this_df.Price.dropna()
        pow_trans.fit(this_df[['Price_no_NA']])

        # Apply the power transform to the data
        this_df['Price_LG'] = pow_trans.transform(this_df[['Price_no_NA']])
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
        '''
        Make correlation heatmap using Seaborn

            Parameters:
            this_df (dataframe): Pandas dataframe
        '''
        f, ax = self.plt.subplots(figsize=(10, 10))
        self.sns.heatmap(this_df.corr(), annot=True, linewidths=5, fmt='.1f', ax=ax, cmap='Reds')
        self.plt.savefig("./Output/Correlation_plot.png")
        self.plt.clf()
        self.plt.close()

    def do_log_transformation_using_np(self, this_df):
        '''
            Do log transformation using Numpy

            Parameters:
                this_df (dataframe): Pandas dataframe
        '''

        #   Caveat limitations of this approach
        #   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120293/

        print("...LOG_TRANSFORM: In function do_log_transformation_using_np()")
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
        self.plt.clf()
        self.plt.close()



    def drop_features(self, this_df):
        '''
            Drop features from dataframe

            Parameters:
                this_df (dataframe): Pandas dataframe
        '''

        print('...Rooms and Bedroom2 highly correlated so drop Bedroom2 as it has NaNs')
        print(this_df[["Rooms", "Bedroom2"]].corr())

        #   'Suburb' and 'SellerG' have predominantly unique values so drop these
        #   There is a direct correlation between CouncilArea and RegionName feature so adds
        #   nothing to the data => drop

        print('...Suburb,SellerG have predominantly unique values so drop these too')
        drop_list = ['Suburb', 'SellerG', 'CouncilArea', 'Bedroom2']
        this_df = this_df.drop(drop_list, axis=1, inplace=True)

