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

from Configs.ConfigSchema import Config  # Ensure this import is correct


class Dashboard:
    def __init__(self, config: Config):
        """Initialize the Dashboard with configuration and prepare data."""
        # Load configuration
        self.load_config(config)

        # Load and prepare data
        self.loss_df = self.load_data()

        # Initialize figures
        self.figures = self.create_figures()
    
    def run(self):
        """Run the Dashboard application."""
        self.create_layout()

    def load_config(self, config: Config):
        """Load configuration settings."""
        self.results_folder = f"Results/{config.data.name}"  # Example config parameter
        self.metrics = ['MSE', 'RMSE', 'MAE', 'R-squared', 'ASE', 'MBD']
        self.ncols = config.dashboard_visualization.n_cols  # Number of columns in grid layout
        symbol = config.data.name.split('/')[-1]
        self.title_text = f"{symbol} Volume Prediction Loss Functions"  # Title for the dashboard

    def absolute_scaled_error(self, y_true, y_pred):
        """Calculate ASE (Absolute Scaled Error)."""
        y_mean = np.mean(y_true)
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_mean))

    def mean_bias_deviation(self, y_true, y_pred):
        """Calculate MBD (Mean Bias Deviation), ignoring zero values in y_true."""
        non_zero_indices = y_true > 2  # Adjust threshold as needed
        return np.mean((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])

    def load_data(self):
        """Load and calculate loss measures from CSV files."""
        # Get list of CSV files in the results folder
        csv_files = glob.glob(os.path.join(self.results_folder, '*.csv'))

        # List to store loss measures
        loss_measures = []

        # Iterate over each CSV file
        for file_path in csv_files:
            # Get model name from file name
            model_name = os.path.splitext(os.path.basename(file_path))[0]  # e.g., 'NeuralNetworks.LSTM'
            model_name = model_name.split('.')[-1]  # e.g., 'LSTM'

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Ensure 'Volume' and 'VolumeForecast' columns are present
            if {'Volume', 'VolumeForecast'}.issubset(df.columns):
                # Calculate loss metrics
                mse = mean_squared_error(df['Volume'], df['VolumeForecast'])
                rmse = sqrt(mse)
                mae = mean_absolute_error(df['Volume'], df['VolumeForecast'])
                r2 = r2_score(df['Volume'], df['VolumeForecast'])
                ase = self.absolute_scaled_error(df['Volume'], df['VolumeForecast'])
                mbd = self.mean_bias_deviation(df['Volume'], df['VolumeForecast'])

                # Append results to the list
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

        # Create a DataFrame from the loss measures
        loss_df = pd.DataFrame(loss_measures)
        return loss_df

    def create_figures(self):
        """Create Bokeh figures for each loss metric."""
        figures = []
        colors = Category10[len(self.metrics)]  # Assign different colors to each metric

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

            p.vbar(x='models', top='values', width=0.5, source=source, color=colors[i])
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

    def create_layout(self):
        """Arrange the layout and add to the document."""
        # Define number of columns in the grid layout
        ncols = self.ncols if hasattr(self, 'ncols') else 3  # Default to 3 if not set

        # Calculate number of rows needed
        nrows = ceil(len(self.figures) / ncols)

        # Organize figures into rows
        grid_figures = []
        for i in range(nrows):
            start = i * ncols
            end = start + ncols
            row_figures = self.figures[start:end]
            grid_figures.append(row_figures)

        # Create the grid layout for the figures
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