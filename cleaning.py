

class CleanData:

    def __init__(self, this_df):
        print(this_df.head())

        # Get the dimensions of the data
        print("****************************************************************")
        print("Pre-Clean Dataset Dimensions : {}".format(this_df.shape))
        print("****************************************************************")

        self.does_df_have_duplicates = self.are_there_duplicates(this_df)
        print("Are there duplicate rows? : {}".format(self.does_df_have_duplicates))

        if self.does_df_have_duplicates:
            print(" ...Going to remove duplicates now...")
            self.drop_duplicates(this_df)
            print(" ...Any duplicates remain? : {}".format(self.are_there_duplicates(this_df)))

        self.does_df_have_NULLS = self.are_there_nulls(this_df)
        print("Are there any NULLs? : {}".format(self.does_df_have_NULLS))

        if self.does_df_have_NULLS:
            print(" ...Going to remove the 3 features with most NULLS now...")
            self.drop_features_with_most_nulls(this_df)

        print("Are there any NaN's? : {}".format(self.are_there_NaNs(this_df)))

        print("Are there any empty rows? : {}".format(self.are_there_empty_rows(this_df)))

        print("Are there any empty features? : {}".format(self.are_there_empty_features(this_df)))

        print("Drop rows with missing price values")
        this_df = self.drop_rows_with_no_price_values(this_df)

        pre_num_rows_in_df = len(this_df.index)
        this_df = this_df[this_df['Distance'].notnull()]
        num_rows_deleted = pre_num_rows_in_df - len(this_df.index)
        print(" ...Dropped number of rows with #N/A in Distance feature? : {}".format(num_rows_deleted))

        print("****************************************************************")
        print("Post-Clean Dataset Dimensions... : {}".format(this_df.shape))
        print("****************************************************************")


    def are_there_nulls(self, this_df):
        return bool(this_df.isnull().sum().sum() > 0)

    def are_there_NaNs(self, this_df):
        return bool(this_df.isna().sum().sum() > 0)

    def are_there_duplicates(self, this_df):
        return bool(this_df.duplicated().any())

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

    def drop_features_with_most_nulls(self, this_df):
        array_features = this_df.isnull().sum().sort_values(ascending=False).head(3).index.values
        return this_df.drop(array_features, axis=1, inplace=True)


    def drop_rows_with_no_price_values(self, this_df):
        pre_num_rows_in_df = len(this_df.index)
        this_df = this_df[this_df['Price'].notnull()]
        num_rows_deleted = pre_num_rows_in_df - len(this_df.index)
        print(" ...How many rows with no price value dropped? : {}".format(num_rows_deleted))
        return this_df

