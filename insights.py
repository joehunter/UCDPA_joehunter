
class Visualize:


    def __init__(self, this_df):

        this_df = this_df[this_df['Longtitude'].notna()]
        this_df = this_df[this_df['Lattitude'].notna()]
        this_df = this_df[this_df['Price_LG'].notna()]
        this_df['Property_Type'] = this_df.apply(lambda row: self.fetch_type_description(row), axis=1)
        this_df['Region'] = this_df.apply(lambda row: self.shorten_region_name(row), axis=1)

        self.map_prices(this_df)
        self.scatter_plot_distance_from_cbd(this_df)
        self.violin_plot_price_per_region(this_df)


    def map_prices(self, this_df):
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        import matplotlib.pyplot as plt

        # import street map
        street_map = gpd.read_file("./Maps/Metropolitan region_region.shp")

        # zip x and y coordinates into single feature
        geometry = [Point(xy) for xy in zip(this_df['Longtitude'], this_df['Lattitude'])]
        # create GeoPandas dataframe
        geo_df = gpd.GeoDataFrame(this_df, crs="EPSG:3112", geometry=geometry)

        # create figure and axes, assign to subplot
        fig, ax = plt.subplots(figsize=(15, 15))
        # add .shp mapfile to axes
        street_map.plot(ax=ax, alpha=0.4, color='grey')

        # add geodataframe to axes
        # assign ‘price’ variable to represent coordinates on graph
        # add legend
        # make datapoints transparent using alpha
        # assign size of points using markersize
        geo_df.plot(column='Price_LG', ax = ax, alpha = 0.5, legend = True, markersize = 1, legend_kwds={'label': "Price"})
        # add title to graph
        plt.title('Melbourne Property Prices', fontsize = 15, fontweight ='bold')

        # set latitiude and longitude boundaries for map display
        plt.xlim(144.5, 145.50)
        plt.ylim(-38.2, -37.6)

        plt.savefig("./Output/Melbourne_Property_Price_Map.png")
        plt.clf()
        plt.close()


    def scatter_plot_distance_from_cbd(self, this_df):
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set(style='whitegrid')
        sns.scatterplot(x="Distance", y="Price", hue="Property_Type", data=this_df)
        plt.savefig("./Output/Distance_From_CBD.png")
        plt.clf()
        plt.close()

    def violin_plot_price_per_region(self, this_df):
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set(style='whitegrid')

        sns.boxplot(x="Region", y="Price_LG", data=this_df)

        plt.savefig("./Output/Violin_plot_per_region.png")
        plt.clf()
        plt.close()

    def fetch_type_description(self, row):
        if row['Type'] == 'h':
            return 'House'
        if row['Type'] == 't':
            return 'Townhouse'
        if row['Type'] == 'u':
            return 'Apartment'
        return 'Other'


    def shorten_region_name(self, row):
        if row['Regionname'] == 'Northern Metropolitan':
            return 'N-Met'
        if row['Regionname'] == 'Southern Metropolitan':
            return 'S-Met'
        if row['Regionname'] == 'Eastern Metropolitan':
            return 'E-Met'
        if row['Regionname'] == 'Western Metropolitan':
            return 'W-Met'
        if row['Regionname'] == 'South-Eastern Metropolitan':
            return 'S.E.-Met'
        if row['Regionname'] == 'Northern Victoria':
            return 'N-Vic'
        if row['Regionname'] == 'Eastern Victoria':
            return 'E-Vic'
        if row['Regionname'] == 'Western Victoria':
            return 'W-Vic'
        return 'Other'

