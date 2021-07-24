import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import webbrowser
import os


import file_handling
import cleaning
import discovery
import regression
import regularization

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

df['log_Price'] = np.log1p(df.Price.dropna())

log_price_mean = df['log_Price'].mean()
log_price_std = df['log_Price'].std()

# view log(price) data
df.log_Price.hist(bins=20)
plt.axvline((log_price_mean+log_price_std), color='k', linestyle='--')
plt.axvline((log_price_mean-log_price_std), color='k', linestyle='--')
plt.axvline(log_price_mean, color='k', linestyle='-')
plt.title('log(Price)')
plt.show()

print(df.shape)

#cleaning.CleanData(df)
#discovery.EDA(df)
#regression.Linear(df)
#regularization.Lasso(df)

new = 2 # open in a new tab, if possible
url = "file://c:/UCDPA_joehunter/Output/report.htm"
#webbrowser.open(url, new=new)

webbrowser.get("C:/Program Files/Google/Chrome/Application/chrome.exe %s").open(url)