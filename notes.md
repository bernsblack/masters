# Notes

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