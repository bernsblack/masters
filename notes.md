# Notes

Regarding training and testing loops, rather use the indices as it makes access coherent daata easier,
although this techniques might create and issue when we start adding extra, other date types
like time/weather/demographic data

When we concat various vectors into one vector - these, vectors should all be 

We might want to have a certain number of coordinates (t,x,y) or (t,l) and
a list of these that we can use for our.

We need a list of time indices or just flat_index?

we define flat_index as the entire data grid flattened.


`flat_index = flatten(t,x,y) or flatten(t,l)`

and

`t, x, y = unflatten(flat_index, shape(T,X,Y))`
`t, l = unflatten(flat_index, shape(T,L))`


where `T,L,X,Y` represents the size of their respective axis.


For comparing the models we want a single list flat_indices
and their shapes (2dim or 3dim). The dataset loaders should
be able to handle this for us seeing that we only have to 
types of datasets - flat_dataset (2dim) and grid_dataset (3dim). 

We can add something to the testing batch loader - where we do not subsample ont sample all of the grids, 
and take the evaluations per time index.


Thus we save each prediction for the model under the data folder - with that models name. Then all results are in one
 place and it is easy to compare the results - should be in a time index format - flattened (N,L) for all predictions.  


## Data Shapes

	crime_feature_indices shape (C,)
    crime_types_grids shape (N, C, H, W)
	crime_grids shape (N, 1, H, W)
	demog_grid shape (1, 37, H, W)
	street_grid shape (1, 512, H, W)
	time_vectors shape (N + 1, 44)
	weather_vectors shape (N, C)
	x_range shape (W,)
	y_range shape (H,)
	t_range shape (N + 1,)
	
## Defining a data pipeline
Do the data clean, reshaping and normalisation first - because each model can 
have different training loops they do not always use the same loaders - so might be a waste of time.
Focus on cleaning and normalising data. We use data-groups to normalise all data using only. 

#### Important note:
1. all models should be comparing the same target indices
2. we might need to average the results to avoid getting biased results.  
 
- Raw data
- Cleaned data
- Reshaped data
- Normalised Data
- Batch Load Data
- Model Transform Data
- Evaluated Data 

We should not touch the indices outside of the batch loaders - this can lead to issues,
if and when the datasets are changes and indices have not changed on different spots