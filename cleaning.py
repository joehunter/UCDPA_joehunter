

class CleanData:

    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    def __init__(self, this_df):
        print(this_df.head())

        # Get the dimensions of the data
        print()
        print("****************************************************************")
        print("Pre-Clean Dataset Dimensions : {}".format(this_df.shape))
        print("****************************************************************")


        print("Are there any rows entirely EMPTY? : {}".format(self.are_there_empty_rows(this_df)))

        print("Are there any features entirely EMPTY? : {}".format(self.are_there_empty_features(this_df)))

        self.does_df_have_duplicates = self.are_there_duplicates(this_df)
        print("Are there duplicate rows (based on ['Address', 'Date'] features)? : {}".format(self.does_df_have_duplicates))

        if self.does_df_have_duplicates:
            print(" ...Going to remove duplicates now...")
            this_df = self.drop_duplicates(this_df)
            print(" ...Any duplicates remain? : {}".format(self.are_there_duplicates(this_df)))

        print("Drop 4 features not required: ['Suburb', 'Address', 'SellerG', 'CouncilArea']")
        self.drop_features(this_df)

        # self.does_df_have_NULLS = self.are_there_nulls(this_df)
        # print("Are there any NULLs? : {}".format(self.does_df_have_NULLS))
        # if self.does_df_have_NULLS:
        #    print(" ...Going to remove the 3 features with most NULLS now...")
        #    self.drop_features_with_most_nulls(this_df)


        list_check_for_NaNs = ['Type', 'Method', 'Regionname']
        self.does_df_have_NaNs = self.are_there_NaNs(this_df, list_check_for_NaNs)
        print("Are there any NaNs? : {}".format(self.does_df_have_NaNs))

        if self.does_df_have_NaNs:
            print(" ...Going to remove NaNs now...from this list ['Type', 'Method', 'Regionname']")
            pre_num_rows_in_df = len(this_df.index)
            this_df = self.drop_rows_with_NaNs(this_df, list_check_for_NaNs)
            print(" ...NaNs Removed -> How many rows were dropped? : {}".format(pre_num_rows_in_df - len(this_df.index)))

        print("Start encoding using OneHotEncoder...")
        print(" ...encoding feature: Type")
        this_df_one_hot_encoded = self.do_one_hot_encoder(this_df, 'Type')
        this_df = this_df.reset_index().drop('index', axis=1)
        this_df = self.pd.concat([this_df, this_df_one_hot_encoded], axis=1)

        print(" ...encoding feature: Method")
        this_df_one_hot_encoded = self.do_one_hot_encoder(this_df, 'Method')
        this_df = this_df.reset_index().drop('index', axis=1)
        this_df = self.pd.concat([this_df, this_df_one_hot_encoded], axis=1)

        print(" ...encoding feature: Regionname")
        this_df_one_hot_encoded = self.do_one_hot_encoder(this_df, 'Regionname')
        this_df = this_df.reset_index().drop('index', axis=1)
        this_df = self.pd.concat([this_df, this_df_one_hot_encoded], axis=1)

        print(this_df.head())


        # print("Drop rows missing Price values")
        # pre_num_rows_in_df = len(this_df.index)
        # self.drop_rows_with_no_price_values(this_df)
        # print(" ...Missing Price -> How many rows were dropped? : {}".format(pre_num_rows_in_df - len(this_df.index)))
        #
        # print("Drop rows missing Distance values")
        # pre_num_rows_in_df = len(this_df.index)
        # self.drop_rows_with_no_distance_values(this_df)
        # print(" ...Missing Distance -> How many rows were dropped? : {}".format(pre_num_rows_in_df - len(this_df.index)))

        # Show the dimensions of the data now?
        print("****************************************************************")
        print("Post-Clean Dataset Dimensions... : {}".format(this_df.shape))
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

    def drop_features(self, this_df):
    #   'Suburb', 'Address', and 'SellerG' have predominantly unique values to drop these
    #   There is a direct correlation between CouncilArea and RegionName feature so adds
    #   nothing to the data => drop
        drop_list = ['Suburb', 'Address', 'SellerG', 'CouncilArea']
        this_df = this_df.drop(drop_list, axis=1, inplace=True)

    def do_one_hot_encoder(self, this_df, feature_name_to_encode):
        cat_encoder = self.OneHotEncoder(sparse=False)
        this_df_type_reshaped = this_df[feature_name_to_encode].values.reshape(-1, 1)
        this_df_type_one_hot_encoded = cat_encoder.fit_transform(this_df_type_reshaped)
        categories = cat_encoder.categories_
        this_df_type_one_hot_encoded = self.pd.DataFrame(this_df_type_one_hot_encoded, columns=categories)
        return this_df_type_one_hot_encoded
