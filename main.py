import file_handling
import cleaning
import discovery
import regression
import regularization

import pandas as pd

choose_data = int(input("Press 1 to choose local data already downloaded OR press 2 to download latest data from Kaggle(requires an API Token)?"))

if choose_data == 1:
    print('Using local data now...please wait')
    fh = file_handling.Import(1)
else:
    print('Downloading from Kaggle...please wait')
    fh = file_handling.Import(0)

df = fh.return_df()
print(df.shape)
print()
print()

discovery.EDA(df)
class_cleaning = cleaning.CleanData(df)
df = class_cleaning.return_df()

#print(df.columns)
#print(df.isna().sum())

#print(df.columns[df.isnull().all(0)])




#regression.Linear(df)
#regularization.Lasso(df)

