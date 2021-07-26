
class EDA:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    def __init__(self, this_df):

        self.correlation_heatmap(this_df)

        this_df['log_Price'] = np.log1p(this_df.Price.dropna())

        log_price_mean = this_df['log_Price'].mean()
        log_price_std = this_df['log_Price'].std()

        # view log(price) data
        this_df.log_Price.hist(bins=20)
        self.plt.axvline((log_price_mean + log_price_std), color='k', linestyle='--')
        self.plt.axvline((log_price_mean - log_price_std), color='k', linestyle='--')
        self.plt.axvline(log_price_mean, color='k', linestyle='-')
        self.plt.title('log(Price)')
        self.plt.show()

        print(df.shape)

    def correlation_heatmap(self, this_df):
            # use the pands .corr() function to compute pairwise correlations for the dataframe
            corr = this_df.corr()
            # visualise the data with seaborn
            mask = self.np.triu(self.np.ones_like(corr, dtype=self.np.bool))
            self.sns.set_style(style='white')
            f, ax = self.plt.subplots(figsize=(11, 9))
            cmap = self.sns.diverging_palette(10, 250, as_cmap=True)
            self.sns.heatmap(corr, mask=mask, cmap=cmap,
                    square=True,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
            self.plt.savefig("./Output/Correlation_plot.png")
            self.plt.show()