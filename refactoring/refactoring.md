# Refactoring of Total Crime Predictions

Evaluations of the total crime is done on GRUFNN and other rolling mean methods like HA, EWM, just to display the
differences. The HA and EWM models are never seen as models, so the estimates are calculated on the fly. But instead we
want to wrap everything in its own model directory with logs metrics and saved models.

Create a function for each model that will train_eval_fn_for_<MODEL_NAME>(data_loaders). The function should:

- Take in SequenceDataLoader
    - This dataloader will have the train, validation and test data loaders grouped together.
    - The sequence length will determine the amount of data the models can have when training.
    - The test set size will always be constant and be required. This will ensure that all models are evaluated on the
      same amount of data.
    - The train, validation and test sets can overlap with sequence length, because the models are only evaluated on the
      final target of the sequence, and those values do not overlap in the various datasets.
- Run hyper parameter optimisation on the training and validation sets to determine valid hyper parameters.
- Run training and validation loops with optimal hyper parameters to determine the number of epochs before over-fitting.
- Train model on training and validation set for the number of epochs determined in the previous step.
    - Training with the validation set included will hopefully ensure better generalisation of the model.
- Evaluation of model using the test set. This includes:
    - Table with various metrics (MASE)

We should then import each metric evaluation for the model directories, e.g.:

- GRUFNN
- Hawkes Process
- CNN1D
- HA
- EWM

### File parts

1. Setup
    1. display and plotting settings
    2. conf loading
    3. logging setup
    4. data path selection (selects time scale)
    5. conf.freq is set from select path - this determines the plot title and axis names
    6.

## Function blocks

Large blocks of code used as closures:

- train_evaluate_hyper
- run_trials
- compare_time_series_metrics

## Data

### Loaders

## Models

- Ground truth
- GRUFNN: (GRU and FNN)
- HA: historic average (need to rename this to periodic average)
- EWM: Exponential weighted mean.

## Metrics

- MASE: mean absolute scaled error
- MAE: mean absolute error (not scaled and can be confusing when comparing experiments on various scales)
- RMSE: root mean square error (values squared before calculating mean - penalising larger errors more than in MAE)
- MSE: mean square error

## Plots

All plots generated:

- plot_time_series_anom_wc
- plot_mi_curves_wc
- plot_crime_count_distributions
- plot_correlation (pearson/spearman/partial)
- plot_time_vectors
- plot_interactive_epoch_losses

pio and rcParams used to set plots templates for plotly and matplotlib.

