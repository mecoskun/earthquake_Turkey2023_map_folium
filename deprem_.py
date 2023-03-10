#!/usr/bin/env python
# coding: utf-8

# # Earthquakes in Turkey since February 5, 2023

# This notebook first scrapes the last 500 earthquakes from Bogazici University's (BOUN) KOERI website and adds them to a dataframe of earthquakes recorded since February 5, 2023. It then uses this data to visualize earthquakes on a map using Folium library.

# In[1]:


#import necessary packages
import pandas as pd
import requests
from bs4 import BeautifulSoup
import folium
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd


# ## Scraping the earthquakes data from web

# In[2]:


# scraping function scrapes the website and saves the last 500 earthquakes in a list

def get_earthquakes():
    earthquakes = []
    response = requests.get("http://www.koeri.boun.edu.tr/scripts/lst0.asp")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        response_text = soup.pre.text
        result = response_text.split("\n")[7:-2]

        for element in result:
            earthquake_info = element.split()
            #print(earthquake_info)
            date = earthquake_info[0]
            time = earthquake_info[1]
            latitude = earthquake_info[2]
            longitude = earthquake_info[3]
            depth = earthquake_info[4]
            magnitude = earthquake_info[6]
            location = earthquake_info[8]
            city = earthquake_info[9]

            earthquake = {
                "date": date,
                "time": time,
                "latitude": latitude,
                "longitude": longitude,
                "depth": depth,
                "magnitude": magnitude,
                "location": location,
                "city": city
            }
            earthquakes.append(earthquake)

        print(f"Earthquake scan completed. Number of earthquakes: {len(earthquakes)}")

    return earthquakes


# In[122]:


# call the scrape function and save the earthquakes
earthquakes_500 = get_earthquakes()

# check the first 5 records (latest occurred quakes)
earthquakes_500[0:5]


# In[123]:


# print the latest 10 earthquakes in an expression:
for i,element in enumerate(earthquakes_500):
    print("On the date of",element['date'],"in", element['location'],"region an earthquake with magnitude of", element['magnitude']," occurred.")

    if i==10:
        break


# ## Data Management

# In[124]:


# convert the earthquakes list into a dataframe using Pandas library
df_500 = pd.DataFrame(earthquakes_500)
df_500.head()


# In[125]:


# read the csv file that contains all the earthquakes that occurred since February 5, 2023
df_old = pd.read_csv('deprems_last.csv', dtype = 'str', index_col=0)
df_old


# In[126]:


# check magnitude of the February 6th morning earthquake that struct on 4:17 am local time (it was a huge 7.8)
df_old[(df_old.date == '2023.02.06') & (df_old.time == '04:17:31')]


# In[127]:


# Use Concat to add the latest 500 earthquakes to the dataframe of previous quakes while dropping duplicate records, and save it to into the csv file to update records
df_updated = pd.concat([df_500,df_old], ignore_index=True).drop_duplicates().reset_index(drop=True)

df_updated.to_csv('deprems_last.csv')
df_updated


# In[128]:


# check that currently the whole data is in object type (string)
df_updated.info()


# In[129]:


# Add a new datetime column to the dataframe by combining the date and time columns and converting them into 'datetime' format
df_updated['datetime'] = pd.to_datetime(df_updated.date+' '+df_updated.time)
df_updated.info()


# Convert resulting dataset to geodata because some spatial analysis require geodata format.

# In[130]:


# Use GeoPandas library to convert the dataframe into a geodata frame by assigning longitude and latitude values into geometry column:
geo_data = gpd.GeoDataFrame(df_updated, geometry=gpd.points_from_xy(df_updated.longitude, df_updated.latitude))
geo_data


# In[131]:


# Convert numerical columns into float data type:
geo_data[['latitude','longitude','magnitude','depth']] = geo_data[['latitude','longitude','magnitude','depth']].astype(float)
geo_data.info()


# In[132]:


# Get summary for numerical columns
geo_data.describe()


# A historgram of magnitudes of all previous earthquakes recorded in the data

# In[119]:


geo_data.magnitude.hist()


# In[133]:


# How many earthquakes with magnitude greater than 3 was recorded:
geo_data[geo_data.magnitude > 3].shape[0]


# In[134]:


# How many earthquakes with magnitude of 5 or greater occurred:
geo_data[geo_data.magnitude >= 5].shape[0]


# ## Mapping Part

# Let's draw a map of Turkey using folium library. We set Turkey's center latitude and longitude as the base map and add a circle mark for each earthquake that has a magnitude greater than 3 by iterating over the rows of the filtered data. Each circle's radius is set to exponent of the magnitude values times a constant for scaling.

# In[93]:


#Draw folium map

tr_lat = 38.9597594
tr_lon = 34.9249653

zoom_start = 6
eq_map = folium.Map(location=[tr_lat, tr_lon], zoom_start=zoom_start) 

for row in geo_data[geo_data.magnitude > 3].itertuples():
    folium.CircleMarker([row.latitude, row.longitude], radius=np.exp(0.5*float(row.magnitude)),
        color='#C30303', #if float(row.depth) <=5 else 'orange' if float(row.depth) <10 else 'yellow' if float(row.depth) <20 else 'green',
        popup='Location:{}, Magnitude:{}, Depth:{}'.format(row.location,row.magnitude,row.depth),
        fill=True, fill_color='#C30303' #if float(row.depth) <=5 else 'orange' if float(row.depth) <10 else 'yellow' if float(row.depth) <20 else 'green'
        , fill_opacity=0.3
    ).add_to(eq_map)

eq_map.save("large_map.html")
eq_map


# Now draw the same map but also add some colors. The colors represent the depth of each earthquake such that those closer to the ground are colored in red, then orange, then yellow and the deepest one are in green. Also notice the minimap added to the corner.

# In[56]:


#Draw folium map
from folium import plugins
minimap = plugins.MiniMap()

tr_lat = 38.9597594
tr_lon = 34.9249653

zoom_start = 7
eq_map = folium.Map(location=[tr_lat, tr_lon], zoom_start=zoom_start) 

for row in geo_data[geo_data.magnitude > 3].itertuples():
    folium.CircleMarker([row.latitude, row.longitude], radius=np.exp(0.5*float(row.magnitude)),
        color='#C30303' if float(row.depth) <=5 else 'orange' if float(row.depth) <10 else 'yellow' if float(row.depth) <20 else 'green',
        popup='Location:{}, Magnitude:{}, Depth:{}'.format(row.location,row.magnitude,row.depth),
        fill=True, fill_color='#C30303' if float(row.depth) <=5 else 'orange' if float(row.depth) <10 else 'yellow' if float(row.depth) <20 else 'green', fill_opacity=0.3
    ).add_to(eq_map)

eq_map.add_child(minimap)
eq_map


# Now, lets focus on the earthquake region (South mid-Turkey). And, instead of using exponents of magnitude for the circle size we use powers of 10 to reflect the true size scale of each earthquake.

# In[57]:


#Draw folium map
from folium import plugins
minimap = plugins.MiniMap()

kmaras_lat = 37.9773
kmaras_lon = 37.4153

zoom_start = 7
eq_map = folium.Map(location=[kmaras_lat, kmaras_lon], zoom_start=zoom_start) 

# select the earthquakes with larger than 3 magnitude
geo_data_big = geo_data[geo_data.magnitude > 3]

for row in geo_data_big.itertuples():
    folium.CircleMarker([row.latitude, row.longitude], radius=0.000005*(10**float(row.magnitude)), 
        color='#C30303' if float(row.depth) <=5 else 'orange' if float(row.depth) <10 else 'yellow' if float(row.depth) <20 else 'green',
        popup='Location:{}, Magnitude:{}, Depth:{}'.format(row.location,row.magnitude,row.depth),
        fill=True, fill_color='#C30303' if float(row.depth) <=5 else 'orange' if float(row.depth) <10 else 'yellow' if float(row.depth) <20 else 'green', fill_opacity=0.2
    ).add_to(eq_map)

eq_map.add_child(minimap)
eq_map


# ## Animated Map

# Next, we use TimestampedGeoJson plugin from the folium library to add a time animation to our map. To use this functionality we need to save out data in a specificly structured geojson format. The below function `create_geojson_features` does that. Then we apply the `TimestampedGeoJson` function with `period = 'PT1H'` argument which means the animation proceeds with 1 frame per hour over the whole data range. Since the time range is fairly long the animation takes several minutes to finish. We can limit the time range by filtering our data to only quakes occurred between certain dates if we want.

# In[58]:


#Draw folium animated map

from folium.plugins import TimestampedGeoJson

kmaras_lat = 37.9773
kmaras_lon = 37.4153

zoom_start = 7
eq_map = folium.Map(location=[kmaras_lat, kmaras_lon], zoom_start=zoom_start) 


def create_geojson_features(df):
    features = []
    
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row['longitude'],row['latitude']]
            },
            'properties': {
                'time': row['datetime'].__str__().replace('-','/'),
                'popup' : 'Location:{}, Magnitude:{}, Depth:{}'.format(row.location,row.magnitude,row.depth),
                'style': {'color' : '#C30303' if row['depth'] <=5 else 'orange' if row['depth'] < 10 else 'yellow' if row['depth'] <20 else 'green'},
                'icon': 'circle',
                    'iconstyle':{
                    'fillColor': '#C30303' if row['depth'] <=5 else 'orange' if row['depth'] < 10 else 'yellow' if row['depth'] <20 else 'green',
                    'fillOpacity': 0.2,
                    'stroke': 'true',
                    'radius': np.exp(0.7 * row['magnitude'])
                }
            }
        }
        features.append(feature)
    return features


geojsondata = create_geojson_features(geo_data)

data2 = {'type': 'FeatureCollection', 'features': geojsondata}


TimestampedGeoJson(data2, period = 'PT1H', duration = None,transition_time=300).add_to(eq_map)
eq_map


# Below is an example of an animated map using filtered data to a specific date range (Speficically to February 6th, when the two major shocks hit the region).
# 
# Also the period is set to 1 frame per 15 minutes.

# In[78]:


#Draw folium animated map

from folium.plugins import TimestampedGeoJson

kmaras_lat = 37.9773
kmaras_lon = 37.4153

zoom_start = 7
eq_map = folium.Map(location=[kmaras_lat, kmaras_lon], zoom_start=zoom_start) 

geojsondata = create_geojson_features(geo_data[(geo_data.magnitude > 3) & (geo_data.datetime > '2023-02-06') & (geo_data.datetime < '2023-02-07')])

data3 = {'type': 'FeatureCollection', 'features': geojsondata}

TimestampedGeoJson(data3, period = 'PT15M', duration = None,transition_time=300).add_to(eq_map)
eq_map


# ## Earthquake Scale Graphs

# In this final section we draw plots to compare true size of earthquakes with different magnitudes using the 10-powered rule. In fact, a magnitude 7 earthquake is 10 times larger in size than a 6 magnitude earthquake.

# In[59]:


import matplotlib.pyplot as plt


# Set the size of the circles based on the logarithmic scale of earthquake magnitudes
circle_size_7p5 = 10**((7.5 - 3)) 
circle_size_7p8 = 10**((7.8 - 3)) 
circle_size_6p6 = 10**((6.6 - 3))
circle_size_5p5 = 10**((5.5 - 3))  

# Plot the circles
plt.scatter([0], [0], s=circle_size_7p8, color='#FF6666', alpha=0.9)
plt.scatter([0], [0], s=circle_size_7p5, color='#FFE88B', alpha=0.9)
plt.scatter([0], [0], s=circle_size_6p6, color='lightgreen', alpha=1)
plt.scatter([0], [0], s=circle_size_5p5, color='green', alpha=1)

# Set the x and y limits to ensure both circles are visible
plt.xlim(-1 * circle_size_7p8, 1 * circle_size_7p8)
plt.ylim(-1 * circle_size_7p8, 1 * circle_size_7p8)

# Add a legend to indicate the magnitude of each circle
plt.legend([plt.Rectangle((0,0), 1,1, fc="green"),
            plt.Rectangle((0,0), 1,1, fc="lightgreen"),
            plt.Rectangle((0,0), 1,1, fc="#FFE88B"),
            plt.Rectangle((0,0), 1,1, fc="#FF6666")],
           ["5.5", "6.6","7.5", "7.8"])

ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

# Display the plot
plt.show()
circle_size_7p8


# Moreover, energy levels of earthquakes increase by a multiple of 32 (approximately) by each increase in magnitude. In another words a magnitude 7 earthquake is factually 32 times stronger (releases that much more energy) than a magnitude 6. The below plot shows difference of energy released (strengts) by four earthquakes with magnitudes 5.5, 6.6, 7.5, 7.8.

# In[60]:


import matplotlib.pyplot as plt
import numpy as np

# Set the size of the circles based on the logarithmic scale of earthquake magnitudes
circle_size_7p5 = 0.0000002*10 ** (1.5 * 7.5 )
circle_size_7p8 = 0.0000002*10 ** (1.5 * 7.8 )
circle_size_6p6 = 0.0000002*10 ** (1.5 * 6.6 )
circle_size_5p5 = 0.0000002*10 ** (1.5 * 5.5 )

# Plot the circles
plt.scatter([0], [0], s=circle_size_7p8, color='#FF6666', alpha=0.9)
plt.scatter([0], [0], s=circle_size_7p5, color='#FFE88B', alpha=0.9)
plt.scatter([0], [0], s=circle_size_6p6, color='lightgreen', alpha=1)
plt.scatter([0], [0], s=circle_size_5p5, color='green', alpha=1)

# Set the x and y limits to ensure both circles are visible
#plt.xlim(-1e19, 1e19)
#plt.ylim(-1e19, 1e19)

plt.xlim(-10 * circle_size_7p8, 10 * circle_size_7p8)
plt.ylim(-10 * circle_size_7p8, 10 * circle_size_7p8)

# Add a legend to indicate the magnitude of each circle
plt.legend([plt.Rectangle((0,0), 1,1, fc="green"),
            plt.Rectangle((0,0), 1,1, fc="lightgreen"),
            plt.Rectangle((0,0), 1,1, fc="#FFE88B"),
            plt.Rectangle((0,0), 1,1, fc="#FF6666")],
           ["5.5", "6.6","7.5", "7.8"])

# Display the plot
plt.show()
circle_size_7p8


# ## Bargraph of earthquakes on a timeline

# In[62]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Sort the data by date
df = geo_data[geo_data.magnitude > 4]#.sort_values(by='datetime')

# Create a bar plot
plt.bar(df['datetime'], df['magnitude'], color= df['magnitude'].apply(lambda x: 'red' if  x>5 else 'blue'), width=0.04, snap=False)

# Set the x-axis labels to be the dates in the data
plt.xticks(df['datetime'], rotation=90)

# Set the x-axis label format to show dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

# Set the x-axis label interval to 4 hours
plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 8)))

# Label the y-axis
plt.ylabel('Magnitude')
#plt.ylim(0, max(df['magnitude'])*1.01)

#plt.set_size_inches(11,8.5)

#plt.figure(figsize=(18, 11), dpi=200)
#plt.figure(figsize=(18, 11), dpi=200)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('deprembar.png', dpi=100)

# Show the plot
plt.show()

