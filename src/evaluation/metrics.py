from pathlib import Path
import pandas as pd


class TrainingMetrics:
    """
    Extract metrics from YOLO training results.
    """

    def __init__(
        self,
        results_file="models/runs/train/results.csv"
    ):
        self.results_file = Path(
            results_file
        )


    def exists(self):
        """
        Check if results file exists.
        """
        return self.results_file.exists()


    def load(self):
        """
        Load YOLO results CSV.
        """

        if not self.exists():
            return None

        return pd.read_csv(
            self.results_file
        )


    def get_final_metrics(self):
        """
        Return last training metrics.
        """

        data = self.load()

        if data is None:
            return {}

        last = data.iloc[-1]

        return {
            "precision": last.get(
                "metrics/precision(B)"
            ),

            "recall": last.get(
                "metrics/recall(B)"
            ),

            "mAP50": last.get(
                "metrics/mAP50(B)"
            ),

            "mAP50_95": last.get(
                "metrics/mAP50-95(B)"
            )
        }