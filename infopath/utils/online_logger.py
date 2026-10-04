import os
from types import SimpleNamespace


class OnlineLogger:
    """Stream training metrics to Comet (https://www.comet.com) while the job runs.

    Active only if `opt.online_logging` is true, `comet_ml` is importable and a Comet API key is
    available (env COMET_API_KEY or ~/.comet.config). Otherwise every method is a no-op, so training
    never depends on the service.
    """

    def __init__(self, opt, name=None):
        self.exp = None
        if not getattr(opt, "online_logging", False):
            return
        try:
            import comet_ml

            self.exp = comet_ml.Experiment(
                project_name=getattr(opt, "online_project", "snn_wm"),
                auto_output_logging=False,
                auto_metric_logging=False,
                auto_param_logging=False,
                log_code=False,
                log_graph=False,
            )
        except Exception as e:  # missing package, no key, no network
            print(f"[online_logger] disabled: {type(e).__name__}: {e}")
            return
        if name:
            self.exp.set_name(name)
        params = {}
        for k, v in vars(opt).items():
            params[k] = v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)
        self.exp.log_parameters(params)

    def log(self, metrics, step):
        if self.exp is None:
            return
        self.exp.log_metrics(
            {k: v for k, v in metrics.items() if v is not None}, step=step
        )

    def log_eval(self, step, model, model_spikes, data_spikes, stims, trial_type, model_perc, data_perc):
        """Per-evaluation statistics: firing rates (model vs data), trial classification, confusion matrix."""
        if self.exp is None:
            return
        try:
            from infopath.utils.eval_stats import TYPE_NAMES, eval_statistics

            metrics, confusion, type_table = eval_statistics(
                model, model_spikes, data_spikes, stims, trial_type, model_perc, data_perc
            )
            self.exp.log_metrics(metrics, step=step)
            self.exp.log_confusion_matrix(
                matrix=confusion.tolist(),
                labels=["NO-GO", "GO"],
                row_label="Stimulus",
                column_label="Assigned class",
                title="Stimulus vs assigned trial class",
                file_name=f"confusion_stimulus_class_{step}.json",
                step=step,
            )
            self.exp.log_table(
                f"assigned_trial_type_{step}.csv",
                tabular_data=[["stimulus"] + TYPE_NAMES]
                + [["NO-GO"] + type_table[0].tolist(), ["GO"] + type_table[1].tolist()],
                headers=True,
                step=step,
            )
        except Exception as e:  # never let logging stop training
            print(f"[online_logger] log_eval failed: {type(e).__name__}: {e}")

    def close(self):
        if self.exp is not None:
            self.exp.end()
