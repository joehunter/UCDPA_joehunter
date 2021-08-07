

class CleanData:

    import pandas as pd
    # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    pd.options.mode.chained_assignment = None
    import numpy as np
    from sklearn.impute import SimpleImputer


    def __init__(self, this_df):

        self.this_df = this_df

        print()
        print("****************************************************************")
        print("@Cleaning | Pre Clean")
        print("****************************************************************")
        print("PRE Dataset Dimensions... : {}".format(this_df.shape))

        print("EMPTY: Are there any rows entirely EMPTY? : {}".format(self.are_there_empty_rows(this_df)))

        print("EMPTY: Are there any features entirely EMPTY? : {}".format(self.are_there_empty_features(this_df)))

        self.does_df_have_duplicates = self.are_there_duplicates(this_df)
        print("DUPLICATES: Are there duplicate rows (based on ['Address', 'Date'] features)? : {}".format(self.does_df_have_duplicates))

        if self.does_df_have_duplicates:
            print(" ...Going to remove duplicates now...")
            this_df = self.drop_duplicates(this_df)
            print(" ...Any duplicates remain? : {}".format(self.are_there_duplicates(this_df)))


        list_check_for_NaNs = ['Type', 'Method', 'Regionname']
        self.does_df_have_NaNs = self.are_there_NaNs(this_df, list_check_for_NaNs)
        print("NANS: Are there any NaNs? : {}".format(self.does_df_have_NaNs))

        if self.does_df_have_NaNs:
            print(" ...Going to remove NaNs now...from this list ['Type', 'Method', 'Regionname']")
            pre_num_rows_in_df = len(this_df.index)
            this_df = self.drop_rows_with_NaNs(this_df, list_check_for_NaNs)
            print(" ...NaNs Removed -> How many rows were dropped? : {}".format(pre_num_rows_in_df - len(this_df.index)))




    #   convert Date feature to proper datetime64
        print("CONVERT: Date feature to native datetime64")
        this_df['Date'] = self.pd.to_datetime(this_df['Date'], errors='raise', dayfirst=1)

        print("MISSING: Price is Target column => Drop rows missing Price values")
        pre_num_rows_in_df = len(this_df.index)
        self.drop_rows_with_no_price_values(this_df)
        print(" ...Missing Price -> How many rows were dropped? : {}".format(pre_num_rows_in_df - len(this_df.index)))

        print("CATEGORISE: Price into high/low binary values")
        self.categorise_price(this_df)

        print("IMPUTE: Populate NaNs with median values")
        this_df = self.impute_with_median_using_price_category(this_df)
        #this_df = self.impute_missing_data(this_df)


        self.this_df = this_df

        # Show the dimensions of the data now?
        print("POST Dataset Dimensions... : {}".format(this_df.shape))
        print("****************************************************************")
        print("@Cleaning | Post Clean")
        print("****************************************************************")





    def are_there_nulls(self, this_df):
        return bool(this_df.isnull().sum().sum() > 0)

    def are_there_NaNs(self, this_df, list_check_for_NaNs):
        return bool(this_df[list_check_for_NaNs].isna().sum().sum() > 0)

    def are_there_duplicates(self, this_df):
        return bool(this_df[["Address", "Date"]].duplicated().any())

    def are_there_empty_rows(self, this_df):
        return bool(this_df.isnull().all(axis=1).any())

    def are_there_empty_features(self, this_df):
        return bool(this_df.isnull().all(axis=0).any())

    def drop_duplicates(self, this_df):
        pre_num_rows_in_df = len(this_df.index)
        list_cols_used_to_identify_duplicates = ["Address", "Date"]
        this_df.drop_duplicates(subset=list_cols_used_to_identify_duplicates, keep='first', inplace=True)
        num_rows_deleted = pre_num_rows_in_df - len(this_df.index)
        print(" ...How many duplicate rows were dropped? : {}".format(num_rows_deleted))
        return this_df

    def drop_rows_with_NaNs(self, this_df, list_check_for_NaNs):
        this_df = this_df.dropna(subset=list_check_for_NaNs, axis=0)
        return this_df

    def drop_features_with_most_nulls(self, this_df):
        array_features = this_df.isnull().sum().sort_values(ascending=False).head(3).index.values
        return this_df.drop(array_features, axis=1, inplace=True)

    def drop_rows_with_no_price_values(self, this_df):
        this_df = this_df.dropna(subset=['Price'], inplace=True)

    def drop_rows_with_no_distance_values(self, this_df):
        this_df = this_df.dropna(subset=['Distance'], inplace=True)


    def categorise_price(self, this_df):
        log_price_mean = this_df['Price_LG'].mean()
        log_price_std = this_df['Price_LG'].std()

        this_df['high_price'] = self.np.where(
            this_df['Price_LG'] > (log_price_mean+log_price_std), 1, 0
        )

        this_df['low_price'] = self.np.where(
            this_df['Price_LG'] < (log_price_mean-log_price_std), 1, 0
        )


    def impute_with_median_using_price_category(self, this_df):

        this_df = this_df.reset_index()

        # get list of all columns with NaN
        cols = this_df.columns[this_df.isna().any()].tolist()

        # remove latitude and longitude
        cols.remove("Lattitude")
        cols.remove("Longtitude")

        # Add high and low price
        cols.append("high_price")
        cols.append("low_price")


        # filter for these columns into a new dataframe
        df_with_NaNs = this_df[cols]

        df_with_NaNs.loc[df_with_NaNs['high_price'] == 1] = df_with_NaNs.loc[
             df_with_NaNs['high_price'] == 1].apply(lambda x: x.fillna(x.median()), axis=0)

        df_with_NaNs.loc[df_with_NaNs['low_price'] == 1] = df_with_NaNs.loc[
            df_with_NaNs['low_price'] == 1].apply(lambda x: x.fillna(x.median()), axis=0)

        df_with_NaNs.loc[df_with_NaNs['high_price' and 'low_price'] == 0] = df_with_NaNs.loc[
            df_with_NaNs['high_price' and 'low_price'] == 0].apply(lambda x: x.fillna(x.median()), axis=0)

        this_df = self.pd.concat([this_df.drop(cols, axis=1), df_with_NaNs], axis=1)

        return this_df



    def impute_missing_data(self, this_df):
        #   https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=5

        this_df = this_df.reset_index()

        # get list of all columns with NaN
        cols = this_df.columns[this_df.isna().any()].tolist()

        # remove latitude and longitude
        cols.remove("Lattitude")
        cols.remove("Longtitude")

        # filter for these columns into a new dataframe
        df_with_NaNs = this_df[cols]

        imp = self.SimpleImputer(missing_values=self.np.nan, strategy='mean')
        imp.fit(df_with_NaNs)
        new_df = self.pd.DataFrame(imp.transform(df_with_NaNs), columns=cols)

        this_df = this_df.drop(cols, axis=1)

        this_df = self.pd.concat([this_df, new_df], axis=1)

        return this_df


    def return_df(self):
        return self.this_df
