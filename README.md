# Predicting crime with Deep Learning

## Data flow process

### Original (as downloaded from the internet)

- `Crimes_Chicago_2001_to_2019.csv` Original crime data source from Chicago open data website
    - Only being used in `simple_dataset`.py
- `openweatherdata.json` Unprocessed weather data
    - Not incorporated with any of the models

### Raw (preprocessed data)

A lot of this data was processed under ../meesters/Chicago where census, weather and crime data explored to discover any
patterns.

### Processes

Crime data is processed with generate_data.py into a grid format according to the config file specified in
./config/generate_data.json. Distribution maps and heatmaps are also produced to get an overview of the data. Grid data
includes:

- Crime Data
- Google Street View Data
- Census/Demographic Data The data is rasterised in time and space. Giving us grids with the dimensions: (N,C,H,W),
  where: N is the number of time steps, C the number of channels, H the height of the grid in number of cells and W the width of the
  grid in the number of cells.

[//]: # (todo: add readme that makes sense )


    


