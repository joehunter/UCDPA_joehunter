

class EDA:

    def __init__(self, this_df):
        print(this_df.head())

        # Get the dimensions of the data
        print("These are the data set dimensions : {}".format(this_df.shape))

        self.does_df_have_duplicates = self.are_there_duplicates(this_df)
        print("Are there duplicate rows? : {}".format(self.does_df_have_duplicates))

        if self.does_df_have_duplicates:
            print("Going to remove duplicates now...")
            self.drop_duplicates(this_df)
            print("Any duplicates remain? : {}".format(self.are_there_duplicates(this_df)))

        print("Are there any NULLs? : {}".format(self.are_there_nulls(this_df)))
        self.drop_features_with_most_nulls(this_df)

        print("Are there any NaN's? : {}".format(self.are_there_NaNs(this_df)))

        print("Date set dimensions now... : {}".format(this_df.shape))


    def are_there_nulls(self, this_df):
        return bool(this_df.isnull().sum().sum() > 0)

    def are_there_NaNs(self, this_df):
        return bool(this_df.isna().sum().sum() > 0)

    def are_there_duplicates(self, this_df):
        return bool(this_df.duplicated().any())

    def drop_duplicates(self, this_df):
        pre_num_rows_in_df = len(this_df.index)
        list_cols_used_to_identify_duplicates = ["Address", "Date"]
        this_df.drop_duplicates(subset=list_cols_used_to_identify_duplicates, keep='first', inplace=True)
        num_rows_deleted = pre_num_rows_in_df - len(this_df.index)
        print("Dropped duplicate rows? : {}".format(num_rows_deleted))
        return this_df

    def drop_features_with_most_nulls(self, this_df):
        array_features = this_df.isnull().sum().sort_values(ascending=False).head(3).index.values
        return this_df.drop(array_features, axis=1, inplace=True)
