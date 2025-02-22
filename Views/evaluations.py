import glob
import os
from math import ceil, sqrt

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Div, HoverTool, Range1d
from bokeh.palettes import Category10
from bokeh.plotting import figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Configs.config_schema import Config  # Ensure this import is correct

class Evaluations:
    """Main application class to orchestrate the dashboard visualization."""
    def __init__(self, config: Config):
        # Load configuration
        self.config_loader = ConfigLoader(config)

        # Load data
        self.data_loader = DataLoader(self.config_loader.results_folder)

        # Calculate metrics
        self.metrics_calculator = MetricsCalculator(self.data_loader.csv_files)

        # Save evaluation metrics to CSV file in the Results folder
        results_csv_path = self.config_loader.evaluation_path

        # Ensure the results folder exists
        os.makedirs(self.config_loader.results_folder, exist_ok=True)
        self.metrics_calculator.loss_df.to_csv(results_csv_path, index=False)
        print(f"Saved evaluation metrics to {results_csv_path}")

        # Create figures
        self.figure_creator = FigureCreator(
            self.metrics_calculator.loss_df,
            self.config_loader.metrics
        )

        # Arrange layout
        self.layout_manager = LayoutManager(
            self.figure_creator.figures,
            self.config_loader.title_text,
            self.config_loader.ncols
        )


class ConfigLoader:
    """Load configuration settings."""
    def __init__(self, config: Config):
        self.config = config
        self.load_config()

    def load_config(self):
        self.results_folder = os.path.dirname(self.config.data.out_path)
        self.evaluation_path = self.config.data.evaluation_path
        self.metrics = ['MSE', 'RMSE', 'MAE', 'R-squared', 'ASE', 'MBD']
        self.ncols = self.config.evaluation_visualization.n_cols
        symbol = self.config.data.name.split('/')[-1]
        self.title_text = f"{symbol} Volume Prediction Loss Functions"


class DataLoader:
    """Load data from CSV files."""
    def __init__(self, results_folder):
        self.results_folder = results_folder
        self.csv_files = self.get_csv_files()

    def get_csv_files(self):
        return glob.glob(os.path.join(self.results_folder, '*.csv'))


class MetricsCalculator:
    """Calculate loss metrics from data."""
    def __init__(self, csv_files):
        self.csv_files = csv_files
        self.loss_df = self.calculate_metrics()

    @staticmethod
    def absolute_scaled_error(y_true, y_pred):
        y_mean = np.mean(y_true)
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_mean))

    @staticmethod
    def mean_bias_deviation(y_true, y_pred):
        non_zero_indices = y_true > 2  # Adjust threshold as needed
        return np.mean((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])

    def calculate_metrics(self):
        loss_measures = []

        for file_path in self.csv_files:
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            model_name = model_name.split('.')[-1]

            df = pd.read_csv(file_path)

            if {'Volume', 'VolumeForecast'}.issubset(df.columns):
                mse = mean_squared_error(df['Volume'], df['VolumeForecast'])
                rmse = sqrt(mse)
                mae = mean_absolute_error(df['Volume'], df['VolumeForecast'])
                r2 = r2_score(df['Volume'], df['VolumeForecast'])
                ase = self.absolute_scaled_error(df['Volume'], df['VolumeForecast'])
                mbd = self.mean_bias_deviation(df['Volume'], df['VolumeForecast'])

                loss_measures.append({
                    'Model': model_name,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R-squared': r2,
                    'ASE': ase,
                    'MBD': mbd
                })
            else:
                print(f"Columns 'Volume' and 'VolumeForecast' not found in {file_path}")

        loss_df = pd.DataFrame(loss_measures)
        return loss_df


class FigureCreator:
    """Create Bokeh figures for each loss metric."""
    def __init__(self, loss_df, metrics):
        self.loss_df = loss_df
        self.metrics = metrics
        self.figures = self.create_figures()

    def create_figures(self):
        figures = []
        colors = Category10[10]  # Use Category10 palette for colors

        for i, metric in enumerate(self.metrics):
            data = {'models': self.loss_df['Model'], 'values': self.loss_df[metric]}
            source = ColumnDataSource(data=data)

            p = figure(
                x_range=self.loss_df['Model'],
                y_axis_label=metric,
                title=f"Model {metric}",
                height=350,
                sizing_mode='stretch_width'
            )

            p.vbar(
                x='models',
                top='values',
                width=0.5,
                source=source,
                color=colors[i % len(colors)]
            )
            p.xgrid.grid_line_color = None
            p.y_range.start = 0
            p.title.text_font_size = '14pt'
            p.xaxis.major_label_orientation = 1  # Rotate x-axis labels if necessary

            # Special case for MBD to handle negative values
            if metric == 'MBD':
                y_min = self.loss_df[metric].min()
                y_max = self.loss_df[metric].max()
                y_range_span = max(abs(y_min), abs(y_max))
                p.y_range = Range1d(-y_range_span * 1.1, y_range_span * 1.1)

            # Add hover tool for tooltips
            hover = HoverTool(
                tooltips=[
                    ("Model", "@models"),
                    (f"{metric}", "@values{0.000}")  # Format as needed
                ]
            )
            p.add_tools(hover)

            figures.append(p)

        return figures


class LayoutManager:
    """Arrange the layout and add to the document."""
    def __init__(self, figures, title_text, ncols):
        self.figures = figures
        self.title_text = title_text
        self.ncols = ncols
        self.create_layout()

    def create_layout(self):
        nrows = ceil(len(self.figures) / self.ncols)
        grid_figures = []
        for i in range(nrows):
            start = i * self.ncols
            end = start + self.ncols
            row_figures = self.figures[start:end]
            grid_figures.append(row_figures)

        grid = gridplot(grid_figures, sizing_mode='stretch_width')

        # Create the title (centered)
        title = Div(
            text=f"<h1 style='text-align:center;'>{self.title_text}</h1>",
            styles={'text-align': 'center'},
            sizing_mode='stretch_width'
        )

        # Combine the title and grid into a layout
        layout = column(title, grid, sizing_mode='stretch_both')
        curdoc().add_root(layout)
        curdoc().title = self.title_text
