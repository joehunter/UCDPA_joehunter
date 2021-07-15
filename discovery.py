

class EDA:
    def __init__(self, this_df):
        print(this_df.head())
        # Get the dimensions of the data
        print("These are the data set dimensions : {}".format(this_df.shape))
        print("Are there any NULLs? : {}".format(self.are_there_nulls(this_df)))
        print("Are there any NaN's? : {}".format(self.are_there_NaNs(this_df)))
        print("Are there duplicate rows? : {}".format(self.are_there_duplicates(this_df)))

    def are_there_nulls(self, this_df):
        return bool(this_df.isnull().sum().sum() > 0)

    def are_there_NaNs(self, this_df):
        return bool(this_df.isna().sum().sum() > 0)

    def are_there_duplicates(self, this_df):
        return bool(this_df.duplicated().any())
