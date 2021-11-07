# Refactoring of Total Crime Predictions

Evaluations of the total crime is done on GRUFNN and other rolling mean methods like HA, EWM, just to display the
differences. The HA and EWM models are never seen as models, so the estimates are calculated on the fly. But instead we
want to wrap everything in its own model directory with logs metrics and saved models.

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

## Plots

All plots generated:

- plot_time_series_anom_wc
- plot_mi_curves_wc
- plot_crime_count_distributions
- plot_correlation (pearson/spearman/partial)
- plot_time_vectors
- plot_interactive_epoch_losses

pio and rcParams used to set plots templates for plotly and matplotlib.

