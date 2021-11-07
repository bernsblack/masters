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

## Data

### Loaders

## Models

- Ground truth
- GRUFNN: (GRU and FNN)
- HA: historic average (need to rename this to periodic average)
- EWM: Exponentional weighted mean.

## Metrics

## Plots

pio and rcParams used to set plots templates for plotly and matplotlib.

