from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import COVID_START, COVID_PEAK_END
from .metrics import calculate_metrics, calculate_metrics_by_segment


class ErrorAnalyzer:
    """Breaks down forecast errors by route, time period, season, etc."""

    def __init__(self, df, y_true, y_pred):
        self.df = df.copy()
        self.df["y_true"] = y_true
        self.df["y_pred"] = y_pred
        self.df["error"] = y_pred - y_true
        self.df["abs_error"] = np.abs(self.df["error"])

        # pct error blows up near zero, so exclude those
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = self.df["error"] / self.df["y_true"] * 100
            pct = np.where(np.abs(self.df["y_true"]) < 1, np.nan, pct)
            self.df["pct_error"] = pct

        self._add_segments()

    def _add_segments(self):
        """Adds time period, day type, and season columns for slicing."""
        self.df["date"] = pd.to_datetime(self.df["date"])

        covid_start = pd.Timestamp(COVID_START)
        covid_end = pd.Timestamp(COVID_PEAK_END)

        self.df["time_period"] = "Post-COVID"
        self.df.loc[self.df["date"] < covid_start, "time_period"] = "Pre-COVID"
        self.df.loc[
            (self.df["date"] >= covid_start) & (self.df["date"] < covid_end),
            "time_period"
        ] = "COVID"

        self.df["day_type"] = self.df["date"].dt.dayofweek.apply(
            lambda x: "Weekend" if x >= 5 else "Weekday"
        )

        self.df["season"] = self.df["date"].dt.quarter.map({
            1: "Q1 (Winter)",
            2: "Q2 (Spring)",
            3: "Q3 (Summer)",
            4: "Q4 (Fall)"
        })

    def overall_metrics(self):
        """Returns aggregate metrics across all data."""
        return calculate_metrics(
            self.df["y_true"].values,
            self.df["y_pred"].values
        )

    def metrics_by_route(self):
        """Metrics broken down by route."""
        results = []
        for route in self.df["route"].unique():
            mask = self.df["route"] == route
            metrics = calculate_metrics(
                self.df.loc[mask, "y_true"].values,
                self.df.loc[mask, "y_pred"].values
            )
            metrics["route"] = route
            results.append(metrics)

        return pd.DataFrame(results).set_index("route")

    def metrics_by_time_period(self):
        """Metrics split by pre-covid, covid, and post-covid."""
        return pd.DataFrame(
            calculate_metrics_by_segment(
                self.df["y_true"].values,
                self.df["y_pred"].values,
                self.df["time_period"].values
            )
        ).T

    def metrics_by_day_type(self):
        """Metrics split by weekday vs weekend."""
        return pd.DataFrame(
            calculate_metrics_by_segment(
                self.df["y_true"].values,
                self.df["y_pred"].values,
                self.df["day_type"].values
            )
        ).T

    def metrics_by_season(self):
        """Metrics split by quarter/season."""
        return pd.DataFrame(
            calculate_metrics_by_segment(
                self.df["y_true"].values,
                self.df["y_pred"].values,
                self.df["season"].values
            )
        ).T

    def worst_forecasts(self, n=10):
        """Returns the n predictions with the highest absolute error."""
        return self.df.nlargest(n, "abs_error")[
            ["date", "route", "y_true", "y_pred", "error"]
        ]

    def plot_residuals_over_time(self, figsize=(14, 5), save_path=None):
        """Scatter of daily mean error with 30-day rolling average line."""
        fig, ax = plt.subplots(figsize=figsize)

        daily_error = self.df.groupby("date")["error"].mean()

        ax.scatter(daily_error.index, daily_error.values, alpha=0.3, s=10)

        rolling = daily_error.rolling(30, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values, color="red", linewidth=2,
                label="30-day rolling mean")

        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Error (Predicted - Actual)")
        ax.set_title("Residuals Over Time")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_error_by_segment(self, segment_col, figsize=(10, 6), save_path=None):
        """Box plot of errors grouped by the given segment column."""
        fig, ax = plt.subplots(figsize=figsize)

        order = self.df.groupby(segment_col)["abs_error"].median().sort_values().index

        sns.boxplot(
            data=self.df,
            x=segment_col,
            y="error",
            order=order,
            ax=ax
        )

        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel(segment_col.replace("_", " ").title())
        ax.set_ylabel("Error (Predicted - Actual)")
        ax.set_title(f"Error Distribution by {segment_col.replace('_', ' ').title()}")

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_predicted_vs_actual(self, figsize=(8, 8), save_path=None):
        """Scatter plot of predicted vs actual with a perfect prediction line."""
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            self.df["y_true"],
            self.df["y_pred"],
            alpha=0.3,
            s=10
        )

        min_val = min(self.df["y_true"].min(), self.df["y_pred"].min())
        max_val = max(self.df["y_true"].max(), self.df["y_pred"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2,
                label="Perfect prediction")

        ax.set_xlabel("Actual Delay (min)")
        ax.set_ylabel("Forecasted Avg Delay (min)")
        ax.set_title("Forecasted vs Actual Daily Avg Delay")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_error_heatmap(self, figsize=(12, 8), save_path=None):
        """MAE heatmap by route x season."""
        pivot_data = []

        for route in self.df["route"].unique():
            for season in self.df["season"].unique():
                mask = (self.df["route"] == route) & (self.df["season"] == season)
                if mask.sum() > 0:
                    mae = np.mean(self.df.loc[mask, "abs_error"])
                    pivot_data.append({
                        "route": route,
                        "season": season,
                        "mae": mae
                    })

        if not pivot_data:
            return None

        pivot_df = pd.DataFrame(pivot_data).pivot(
            index="route", columns="season", values="mae"
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            ax=ax,
            cbar_kws={"label": "MAE (min)"}
        )

        ax.set_title("MAE by Route and Season")
        ax.set_xlabel("Season")
        ax.set_ylabel("Route")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_report(self, output_dir=None):
        """Writes a text report and saves all plots to output_dir if given."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        lines = ["Error Analysis Report", ""]

        lines.append("\nOVERALL METRICS:")
        overall = self.overall_metrics()
        for name, value in overall.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.2f}")
            else:
                lines.append(f"  {name}: {value}")

        lines.append("\n\nMETRICS BY ROUTE:")
        route_metrics = self.metrics_by_route()
        lines.append(route_metrics[["rmse", "mae", "mape"]].round(2).to_string())

        lines.append("\n\nMETRICS BY DAY TYPE:")
        day_metrics = self.metrics_by_day_type()
        lines.append(day_metrics[["rmse", "mae", "mape"]].round(2).to_string())

        lines.append("\n\nMETRICS BY SEASON:")
        season_metrics = self.metrics_by_season()
        lines.append(season_metrics[["rmse", "mae", "mape"]].round(2).to_string())

        lines.append("\n\nWORST FORECASTS:")
        worst = self.worst_forecasts(10)
        lines.append(worst.to_string())

        report_text = "\n".join(lines)

        if output_dir:
            with open(output_dir / "error_analysis_report.txt", "w") as f:
                f.write(report_text)

            for plot_fn, args, filename in [
                (self.plot_residuals_over_time, {}, "residuals_over_time.png"),
                (self.plot_error_by_segment, {"segment_col": "route"}, "error_by_route.png"),
                (self.plot_error_by_segment, {"segment_col": "season"}, "error_by_season.png"),
                (self.plot_predicted_vs_actual, {}, "predicted_vs_actual.png"),
                (self.plot_error_heatmap, {}, "error_heatmap.png"),
            ]:
                fig = plot_fn(**args, save_path=output_dir / filename)
                if fig is not None:
                    plt.close(fig)

        return report_text
