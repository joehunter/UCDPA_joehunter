import importing
import cleaning
import discovery
import preprocessing
import modelling
import tuning
import insights

choose_data = int(input("Press 1 to choose local data already downloaded OR press 2 to download latest data from Kaggle(requires an API Token)?"))

if choose_data == 1:
    print('Using local data now...please wait')
    fh = importing.Import(1)
else:
    print('Downloading from Kaggle...please wait')
    fh = importing.Import(0)

df = fh.return_df()



discovery.EDA(df)

data_cleaning = cleaning.CleanData(df)
df = data_cleaning.return_df()

# pre_processing = preprocessing.Encode(df)
# df = pre_processing.return_df()
#
# modelling.RunModels(df)
#
# tuning.TuneModel(df)

insights.Visualize(df)
