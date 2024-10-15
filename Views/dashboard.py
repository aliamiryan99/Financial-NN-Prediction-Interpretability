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


def absolute_scaled_error(y_true, y_pred):
    """Calculate ASE (Absolute Scaled Error)"""
    y_mean = np.mean(y_true)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_mean))

def mean_bias_deviation(y_true, y_pred):
    """Calculate MBD (Mean Bias Deviation), ignoring zero values in y_true"""
    non_zero_indices = y_true > 2  # Get indices where y_true is not zero
    return np.mean((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])

def load_data():
    # Path to the Results folder
    results_folder = 'Results/ForexData/XAUUSD_H1'  # Ensure this path is correct

    # Get list of CSV files in the results folder
    csv_files = glob.glob(os.path.join(results_folder, '*.csv'))

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
        if 'Volume' in df.columns and 'VolumeForecast' in df.columns:
            # Calculate MSE
            mse = mean_squared_error(df['Volume'], df['VolumeForecast'])
            # Calculate RMSE
            rmse = sqrt(mse)
            # Calculate MAE
            mae = mean_absolute_error(df['Volume'], df['VolumeForecast'])
            # Calculate R-squared
            r2 = r2_score(df['Volume'], df['VolumeForecast'])
            # Calculate ASE
            ase = absolute_scaled_error(df['Volume'], df['VolumeForecast'])
            # Calculate MBD
            mbd = mean_bias_deviation(df['Volume'], df['VolumeForecast'])
                
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

def create_figures(loss_df):
    # Prepare data for plotting
    models = list(loss_df['Model'])
    metrics = ['MSE', 'RMSE', 'MAE', 'R-squared', 'ASE', 'MBD']  # Added ASE and MBD
    figures = []
    colors = Category10[len(metrics)]  # Assign different colors to each metric

    for i, metric in enumerate(metrics):
        data = {'models': models, 'values': loss_df[metric].tolist()}
        source = ColumnDataSource(data=data)
        
        # Set the sizing mode to 'stretch_width' for full screen width
        p = figure(x_range=models, y_axis_label=metric, title=f"Model {metric}", height=350, sizing_mode='stretch_width')
        p.vbar(x='models', top='values', width=0.5, source=source, color=colors[i])
        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_orientation = 1  # Rotate x-axis labels if necessary

         # Special case for MBD to handle negative values
        if metric == 'MBD':
            # Determine the range to cover both negative and positive values
            y_range = abs(max(0, max(loss_df[metric])) - min(0, min(loss_df[metric])))
            y_min = -y_range*1.1
            y_max = y_range*1.1
            p.y_range = Range1d(y_min, y_max)  # Adjust y-range dynamically based on MBD values
        
        # Add hover tool for tooltips
        hover = HoverTool()
        hover.tooltips = [("Model", "@models"), (f"{metric}", "@values")]
        p.add_tools(hover)
        
        figures.append(p)
    
    return figures

def modify_doc(doc):
    # Load the data
    loss_df = load_data()
    
    # Create the figures
    figures = create_figures(loss_df)
    
    # Define number of columns (ncols) in the grid layout
    ncols = 3  # Set the desired number of columns to 3
    figures_per_row = ncols
    nrows = ceil(len(figures) / ncols)  # Calculate the number of rows needed
    
    grid_figures = []
    for i in range(nrows):
        start = i * figures_per_row
        end = start + figures_per_row
        row_figures = figures[start:end]
        grid_figures.append(row_figures)
    
    # Create the grid layout for the figures
    grid = gridplot(grid_figures, sizing_mode='stretch_width')  # Apply stretch_width to the grid
    
    # Create the title (centered)
    title = Div(text="<h1 style='text-align:center;'>XAUUSD Volume Prediction Loss Functions</h1>", height=50, width=800, sizing_mode='stretch_width')
    
    # Combine the title and grid into a layout
    layout = column(title, grid, sizing_mode='stretch_width')
    
    # Add the layout to the document
    doc.add_root(layout)

# For Bokeh serve
modify_doc(curdoc())
