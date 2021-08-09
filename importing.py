import pandas as pd
import os
import shutil
import zipfile
import glob


class Import:

    """
     Import data from either API or local folder depending on end user choice.

     Methods
     -------
    return_df(self):
        Returns this instance of the dataframe.

    use_local_copy_of_data(self):
        Reads a CSV file stored in local folder called 'Data\local\' relative to the main code.

    use_downloaded_copy_of_data(self):
        Unpacks data downloaded from API.

    download_data_from_kaggle_api(self, this_path_name):
        Download data downloaded from API.

    unzip_this_zip_file(self, this_path_and_zip_file_name, extract_to):
        Unzip zip file passed into it.

    delete_all_files_from_this_folder(self, folder):
        Purge all files in a folder.

    """

    def __init__(self, use_local_data):

        if use_local_data == 1:
            self.main_df = self.use_local_copy_of_data()
        else:
            self.main_df = self.use_downloaded_copy_of_data()


    def return_df(self):
        '''
        Returns this instance of the dataframe.

            Parameters:
            -------
            None

            Returns:
                this_df (Dataframe): A dataframe
        '''
        return self.main_df


    def use_local_copy_of_data(self):
        '''
        Reads a CSV file stored in local folder called 'Data\local\' relative to the main code.

            Returns: DataFrame
        '''
        return pd.read_csv(".\Data\Local\Melbourne_housing_v2021-07-15.csv")


    def use_downloaded_copy_of_data(self):
        '''
        Unpacks data downloaded from API.

            Returns: DataFrame
        '''
        path_name = './Data/From_api/'
        path_and_zip_file_name = self.download_data_from_kaggle_api(path_name)
        self.unzip_this_zip_file(path_and_zip_file_name, path_name)
        for name in glob.glob(path_name + '*.csv'):
            csv_file_name = name
        return pd.read_csv(csv_file_name)


    def download_data_from_kaggle_api(self, this_path_name):
        '''
        Download data downloaded from API.

            Parameters:
                this_path_name (string): Full path name.

            Returns: zip file name
        '''
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        self.delete_all_files_from_this_folder(this_path_name)

        api.dataset_download_files('anthonypino/melbourne-housing-market', path=this_path_name)

        for name in glob.glob(this_path_name + '*.zip'):
            dataset_zip_name = name
        return dataset_zip_name

    def unzip_this_zip_file(self, this_path_and_zip_file_name, extract_to):
        '''
        Unzip zip file passed into it.

            Parameters:
                this_path_and_zip_file_name (string): Full path and zip file name.
                extract_to (string): Location to extract to.
        '''
        with zipfile.ZipFile(this_path_and_zip_file_name, 'r') as zipref:
            zipref.extractall(extract_to)


    def delete_all_files_from_this_folder(self, folder):
        '''
        Purge all files in a folder.

            Parameters:
                folder (string): Folder location.
        '''

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))



