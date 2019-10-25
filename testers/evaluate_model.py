import logging as log
from models.model_result import ModelMetrics, ModelResult
from utils.metrics import PRCurvePlotter, ROCCurvePlotter


def evaluate_model(model, batch_loader, eval_fn, shaper, conf):
    """
    Training the model for a single epoch
    """
    y_true, y_pred, probas_pred = eval_fn(model, batch_loader, conf)

    # save result
    # only saves the result of the metrics not the predicted values
    model_metrics = ModelMetrics(model_name=conf.model_name,
                                 y_true=y_true,
                                 y_pred=y_pred,
                                 probas_pred=probas_pred)
    log.info(model_metrics)

    # saves the actual target and predicted values to be visualised later on
    model_result = ModelResult(model_name=conf.model_name,
                               y_true=y_true,
                               y_pred=y_pred,
                               probas_pred=probas_pred,
                               t_range=batch_loader.dataset.t_range,
                               shaper=shaper)
    log.info(model_result)

    # do result plotting and saving
    pr_plotter = PRCurvePlotter()
    pr_plotter.add_curve(y_true.flatten(), probas_pred.flatten(), label_name=conf.model_name)
    pr_plotter.savefig(conf.model_path + "plot_pr_curve.png")

    roc_plotter = ROCCurvePlotter()
    roc_plotter.add_curve(y_true.flatten(), probas_pred.flatten(), label_name=conf.model_name)
    roc_plotter.savefig(conf.model_path + "plot_roc_curve.png")
    return y_true, y_pred, probas_pred