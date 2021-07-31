

class Encode:
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    def __init__(self, this_df):

        self.this_df = this_df

        #
        #   Type, Method, RegionName
        #   As these are not numerical values, the scikit-learn API will not accept
        #   them and you will have to preprocess these features into the correct format.
        #   Our goal is to convert these features so that they are numerical.
        #   https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=1
        #

        print()
        print("****************************************************************")
        print("@Preprocessing | Start Encoding")
        print("****************************************************************")

        print("ENCODING: Start encoding using OneHotEncoder...")
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


        objects = []
        for i in this_df.columns.values:
            if this_df[i].dtype == 'O':
                objects.append(str(i))
        print("OBJECTS: Drop features with object data type: {}".format(objects))
        self.this_df = this_df.drop(objects, axis=1)

        print("****************************************************************")
        print("@Preprocessing | End Encoding")
        print("****************************************************************")



    def do_one_hot_encoder(self, this_df, feature_name_to_encode):
        this_one_hot_encoder = self.OneHotEncoder(sparse=False)
        this_df_type_reshaped = this_df[feature_name_to_encode].values.reshape(-1, 1)
        this_df_type_one_hot_encoded = this_one_hot_encoder.fit_transform(this_df_type_reshaped)
        categories = this_one_hot_encoder.get_feature_names([feature_name_to_encode])
        this_df_type_one_hot_encoded = self.pd.DataFrame(this_df_type_one_hot_encoded, columns=categories)
        return this_df_type_one_hot_encoded

    def return_df(self):
        return self.this_df