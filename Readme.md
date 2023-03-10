# Python Script for the Earthquake Mapping Project

## This project maps out the devastating earthquakes occurred on February 6th 2023 in Turkey

### Description
This project was born out of an aspiration to raise awareness about the severe destruction and loss of lives caused by the earthquakes that struck Turkey in February 2023. The project consists of two parts. The first part loads the existing earthquake data (of earthquakes recorded since February 5, 2023) and adds on it the 500 most recently occurred earthquakes data scraped from the Bogazici University's (BOUN) KOERI website. 

The second part uses the folium library to visualize earthquakes on a map of Turkey. This part utilizes a variety of features of the folium library to demonstrate different ways to display the data on the map, including a time-animated version. Additionally, simple circle plots are used to educate the reader about the true size and strenght of earthquakes.

NOTE: notice that depending on when you run the script the last 500 earthquakes will be further away from dates of the earthquakes saved in the existing csv file. There will be gaps because the csv file is not being updated periodically.  


### Requirements
The following python libraries need to be installed to run the script file (see requirements.txt file):  
pandas  
numpy  
requests  
beautifulsoup4  
folium  
matplotlib  
geopandas  
  
Also note that the script file is provided both in .py and jupyter notebook formats.


