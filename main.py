import importing
import cleaning
import discovery
import preprocessing
import modelling
import tuning
import insights

choose_data = int(input("Press 1 to choose local data already downloaded OR press 2 to download latest data from Kaggle(requires an API Token)?"))

# fetch data
if choose_data == 1:
    print('Using local data now...please wait')
    fh = importing.Import(1)
else:
    print('Downloading from Kaggle...please wait')
    fh = importing.Import(0)

raw_df = fh.return_df()

# perform EDA
discovery.EDA(raw_df)

# clean anomalies
data_cleaning = cleaning.CleanData(raw_df)
cleaned_df = data_cleaning.return_df()

# encode
pre_processing = preprocessing.Encode(cleaned_df)
cleaned_df = pre_processing.return_df()

# start modelling
modelling.RunModels(cleaned_df)

# Tune best regression model
tuning.TuneModel(cleaned_df)

#   Perform final analysis on dataset
insights.Visualize(raw_df)
