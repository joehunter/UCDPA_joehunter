import file_handling
import cleaning
import discovery
import preprocessing
import modelling


choose_data = int(input("Press 1 to choose local data already downloaded OR press 2 to download latest data from Kaggle(requires an API Token)?"))

if choose_data == 1:
    print('Using local data now...please wait')
    fh = file_handling.Import(1)
else:
    print('Downloading from Kaggle...please wait')
    fh = file_handling.Import(0)

df = fh.return_df()

data_train_raw = df.loc[~df.Price.isnull(), :]
data_test_raw = df.loc[df.Price.isnull(), :]

print('Training samples = {}\nTesting samples = {}\nTrain-test ratio = {}'.format(len(data_train_raw), len(data_test_raw), round(len(data_train_raw)/len(data_test_raw), 1)))

discovery.EDA(df)

data_cleaning = cleaning.CleanData(df)
df = data_cleaning.return_df()

pre_processing = preprocessing.Encode(df)
df = pre_processing.return_df()

modelling.RunModels(df)

