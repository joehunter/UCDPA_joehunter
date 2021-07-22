
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import zipfile
import glob
import numpy as np


import cleaning
import discovery
import regression
import regularization


def use_local_copy_of_data():
    """
    Reads a CSV file stored in local folder called 'Data\local\' relative to the main code.

    Returns: DataFrame
    """
    return pd.read_csv(".\Data\Local\Melbourne_housing_v2021-07-15.csv")


def delete_all_files_from_this_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def download_data_from_kaggle_api(this_path_name):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    delete_all_files_from_this_folder(this_path_name)

    api.dataset_download_files('anthonypino/melbourne-housing-market', path=this_path_name)

    for name in glob.glob(this_path_name + '*.zip'):
        dataset_zip_name = name
    return dataset_zip_name


def unzip_this_zip_file(this_path_and_zip_file_name, extract_to):
    with zipfile.ZipFile(this_path_and_zip_file_name, 'r') as zipref:
        zipref.extractall(extract_to)


def use_downloaded_copy_of_data():
    path_name = './Data/From_api/'
    path_and_zip_file_name = download_data_from_kaggle_api(path_name)
    unzip_this_zip_file(path_and_zip_file_name, path_name)
    for name in glob.glob(path_name+'*.csv'):
        csv_file_name = name
    return pd.read_csv(csv_file_name)


choose_data = int(input("Press 1 to choose local data already downloaded OR press 2 to download latest data from Kaggle(requires an API Token)?"))


if choose_data == 1:
    print('Using local data now...please wait')
    df = use_local_copy_of_data()
else:
    print('Downloading from Kaggle...please wait')
    df = use_downloaded_copy_of_data()


print()
print()

cleaning.CleanData(df)
#discovery.EDA(df)
#regression.Linear(df)
regularization.Lasso(df)



