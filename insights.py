
class Visualize:


    def __init__(self, this_df):


        map_prices()


    def map_prices(self):
        # import libraries
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        import matplotlib.pyplot as plt

        # import street map
        street_map = gpd.read_file("./Maps/Metropolitan region_region.shp")


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