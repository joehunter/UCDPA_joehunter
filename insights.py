
class Visualize:


    def __init__(self, this_df):

        self.map_prices(this_df)


    def map_prices(self, this_df):
        # import libraries
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        import matplotlib.pyplot as plt

        df = this_df[this_df['Longtitude'].notna()]
        df = this_df[this_df['Lattitude'].notna()]
        df = this_df[this_df['Price'].notna()]

        # import street map
        street_map = gpd.read_file("./Maps/Metropolitan region_region.shp")

        #zip x and y coordinates into single feature
        geometry = [Point(xy) for xy in zip(df['Longtitude'], df['Lattitude'])]
        # create GeoPandas dataframe
        geo_df = gpd.GeoDataFrame(df, crs="EPSG:3112", geometry=geometry)
        print(geo_df.head())

        # create figure and axes, assign to subplot
        fig, ax = plt.subplots(figsize=(15, 15))
        # add .shp mapfile to axes
        street_map.plot(ax=ax, alpha=0.4, color='grey')

        # add geodataframe to axes
        # assign ‘price’ variable to represent coordinates on graph
        # add legend
        # make datapoints transparent using alpha
        # assign size of points using markersize
        geo_df.plot(column='Price_LG', ax = ax, alpha = 0.5, legend = True, markersize = 1, legend_kwds={'label': "Prices", 'orientation': "horizontal"})
        # add title to graph
        plt.title('Melbourne Property Prices', fontsize = 15, fontweight ='bold')

        # set latitiude and longitude boundaries for map display
        plt.xlim(144.5, 145.50)
        plt.ylim(-38.2, -37.6)
        # show map
        plt.show()



    def scatter_plot_distance_from_cbd(self, this_df):
        import seaborn as sns
        import matplotlib.pyplot as plt

        this_df['Property_Type'] = this_df.apply(lambda row: self.fetch_type_description(row), axis=1)

        sns.set(style='whitegrid')
        sns.scatterplot(x="Distance",
                            y="Price",
                            hue="Property_Type",
                            data=this_df)

        plt.savefig("./Output/Distance_From_CBD.png")
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


    # def use_plotly(self):
    #     import plotly.express as px
    #
    #     fig = px.scatter_geo(this_df, lat='Lattitude', lon='Longtitude', hover_name="Address")
    #     fig.update_layout(title='World map', title_x=0.5)
    #     fig.show()
    #
    #
    #     fig = px.density_mapbox(this_df, lat='Lattitude', lon='Longtitude', z='Propertycount', radius=10,
    #                             center=dict(lat=0, lon=180), zoom=0,
    #                             mapbox_style="stamen-terrain")
    #     fig.show()