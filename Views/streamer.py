import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (BoxZoomTool, Button, ColumnDataSource,
                          DatetimeTickFormatter, Div, HoverTool, PanTool,
                          ResetTool, SaveTool, Spinner, WheelZoomTool)
from bokeh.plotting import curdoc, figure

# ---------------------------
# Configuration and Setup
# ---------------------------

# Path to your CSV file
CSV_FILE_PATH = 'Results/ForexData/XAUUSD_H1/Statisticals.SARIMA.csv'

# Number of data points to stream at each interval
BATCH_SIZE = 1  # Streaming one data point at a time

# Interval between data updates in milliseconds
UPDATE_INTERVAL = 50

# Maximum number of data points to display
MAX_POINTS = 500  # Adjust based on your preference

# Width of candlesticks in milliseconds (for H1 data)
# 0.5 hours = 30 minutes = 1,800,000 milliseconds
CANDLE_WIDTH = pd.Timedelta('0.7H')

# ---------------------------
# Data Preparation
# ---------------------------

# Load CSV data into DataFrame
df = pd.read_csv(CSV_FILE_PATH)

# Ensure the CSV has the required columns
required_columns = {'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VolumeForecast'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain the following columns: {required_columns}")

# Convert 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'])

# Sort data by Time in ascending order
df = df.sort_values('Time').reset_index(drop=True)

# Add a column to indicate whether the candle is bullish (Close >= Open)
df['Status'] = ['bullish' if row['Close'] >= row['Open'] else 'bearish' for _, row in df.iterrows()]

# Add a column for candle color based on status
df['Color'] = ['green' if status == 'bullish' else 'red' for status in df['Status']]

# Total number of data points
TOTAL_POINTS = len(df)

# Initialize a pointer to keep track of streaming progress
current_index = 0

# ---------------------------
# Initialize ColumnDataSources
# ---------------------------

# Initialize empty data sources for candlesticks
source_price = ColumnDataSource(data=dict(
    Time=np.array([], dtype='datetime64[ns]'),
    Open=np.array([], dtype='float64'),
    High=np.array([], dtype='float64'),
    Low=np.array([], dtype='float64'),
    Close=np.array([], dtype='float64'),
    Color=np.array([], dtype='object')
))

# Initialize empty data source for volume
source_volume = ColumnDataSource(data=dict(
    x1=np.array([], dtype='datetime64[ns]'),
    x2=np.array([], dtype='datetime64[ns]'),
    Time=np.array([], dtype='datetime64[ns]'),
    Volume=np.array([], dtype='float64'),
    PredictedVolume=np.array([], dtype='float64')
))

# ---------------------------
# Create Bokeh Figures
# ---------------------------

# Candlestick (Price) Plot
candlestick_plot = figure(
    title='XAU/USD Candlestick Streaming',
    x_axis_type='datetime',
    height=450,
    width=800,
    sizing_mode='stretch_width',
    toolbar_location='above',
    tools=[],  # Initialize with no tools; we'll add custom tools below
    y_axis_label='Price (USD)'
)

# Add interactive tools to the price plot
candlestick_plot.add_tools(
    PanTool(),
    WheelZoomTool(),
    BoxZoomTool(),
    ResetTool(),
    SaveTool(),
    HoverTool(
        tooltips=[
            ("Time", "@Time{%F %H:%M}"),
            ("Open", "@Open{0.2f}"),
            ("High", "@High{0.2f}"),
            ("Low", "@Low{0.2f}"),
            ("Close", "@Close{0.2f}"),
        ],
        formatters={'@Time': 'datetime'},
        mode='vline'
    )
)

# Configure x-axis datetime format
candlestick_plot.xaxis.formatter = DatetimeTickFormatter(
    hours=["%Y-%m-%d %H:%M"],
    days=["%Y-%m-%d"],
    months=["%Y-%m"],
    years=["%Y"],
)

# Add wicks (high-low lines)
candlestick_plot.segment('Time', 'High', 'Time', 'Low', source=source_price, color='black')

# Add candle bodies (vbar: x, width, bottom, top)
candlestick_plot.vbar(
    'Time',
    CANDLE_WIDTH,
    'Open',
    'Close',
    source=source_price,
    fill_color='Color',
    line_color='black'
)

# Volume Plot (Unchanged)
volume_plot = figure(
    title='XAU/USD Volume Streaming',
    x_axis_type='datetime',
    height=200,
    width=800,
    sizing_mode='stretch_width',
    toolbar_location='above',
    x_range=candlestick_plot.x_range,  # Link x-axes
    tools=[],  # No tools for volume plot
    y_axis_label='Volume'
)

# Configure x-axis datetime format for volume plot
volume_plot.xaxis.formatter = DatetimeTickFormatter(
    hours=["%Y-%m-%d %H:%M"],
    days=["%Y-%m-%d"],
    months=["%Y-%m"],
    years=["%Y"],
)

# Add volume bars with two columns next to each other
offset = CANDLE_WIDTH / 4

# First volume bar (full volume)
volume_plot.vbar(
    x='x1',
    top='Volume',
    width=CANDLE_WIDTH * 0.4,
    source=source_volume,
    color='royalblue',
    legend_label='Volume'
)

# Second volume bar (predicted volume)
volume_plot.vbar(
    x='x2',
    top='PredictedVolume',
    width=CANDLE_WIDTH * 0.4,
    source=source_volume,
    color='crimson',
    legend_label='Predicted Volume'
)

# Add hover tool for volume plot
volume_plot.add_tools(
    HoverTool(
        tooltips=[
            ("Time", "@Time{%F %H:%M}"),
            ("Volume", "@Volume"),
        ],
        formatters={'@Time': 'datetime'},
        mode='vline'
    )
)

# ---------------------------
# Create Pause/Resume Button and Status Message
# ---------------------------

# Initialize a flag to control streaming
is_paused = False

# Create a Button widget
pause_button = Button(label="Pause", button_type="success", width=100)

# Create a Div widget for status messages
status_div = Div(text="<b>Status:</b> <span style='color:green;'>Streaming Active</span>", width=200, height=30)
    
# Define the callback function for the button
def toggle_pause():
    global is_paused
    if is_paused:
        # Resume streaming
        is_paused = False
        pause_button.label = "Pause"
        pause_button.button_type = "success"
        status_div.text = "<b>Status:</b> <span style='color:green;'>Streaming Active</span>"
        print("Streaming resumed.")
    else:
        # Pause streaming
        is_paused = True
        pause_button.label = "Resume"
        pause_button.button_type = "warning"
        status_div.text = "<b>Status:</b> <span style='color:red;'>Streaming Paused</span>"
        print("Streaming paused.")

# Assign the callback to the button
pause_button.on_click(toggle_pause)

# ---------------------------
# Create Spinner Widget for UPDATE_INTERVAL
# ---------------------------

# Create a Spinner widget for streaming speed (UPDATE_INTERVAL)
speed_spinner = Spinner(
    title="Streaming Delay (ms):",
    low=10,
    high=1000,
    step=10,
    value=UPDATE_INTERVAL,
    width=150
)

# Initialize the callback ID variable
callback_id = None
# Define the callback function for the spinner
def update_interval(attr, old, new):
    global callback_id, UPDATE_INTERVAL
    try:
        # Update the UPDATE_INTERVAL with the new value from the spinner
        new_interval = int(speed_spinner.value)
        
        # Remove the existing callback if it exists
        if callback_id is not None:
            curdoc().remove_periodic_callback(callback_id)
        
        # Update the global UPDATE_INTERVAL
        UPDATE_INTERVAL = new_interval
        
        # Add a new periodic callback with the updated interval
        callback_id = curdoc().add_periodic_callback(update, UPDATE_INTERVAL)
        
        print(f"Streaming speed updated to {UPDATE_INTERVAL} ms.")
    except Exception as e:
        print(f"Error updating streaming speed: {e}")

# Assign the callback to the spinner's 'value' property
speed_spinner.on_change('value', update_interval)

# ---------------------------
# Update Function
# ---------------------------

def update():
    global current_index
    if is_paused:
        return  # Do not stream new data if paused
    # Determine the end index for the current batch
    end_index = current_index + BATCH_SIZE
    if end_index > TOTAL_POINTS:
        end_index = TOTAL_POINTS

    # Slice the DataFrame for the current batch
    new_data = df.iloc[current_index:end_index]

    if new_data.empty:
        return  # No new data to add

    # Prepare data for candlestick plot
    new_candles = dict(
        Time=new_data['Time'],
        Open=new_data['Open'],
        High=new_data['High'],
        Low=new_data['Low'],
        Close=new_data['Close'],
        Color=new_data['Color']
    )

    # Prepare data for volume plot
    new_volume = dict(
        x1=new_data['Time'] - offset,
        x2=new_data['Time'] + offset,
        Time=new_data['Time'],
        Volume=new_data['Volume'],
        PredictedVolume=new_data['VolumeForecast']
    )

    # Stream new data into the sources
    source_price.stream(new_candles, rollover=MAX_POINTS)
    source_volume.stream(new_volume, rollover=MAX_POINTS)

    # Update the current index
    current_index = end_index

    # Optionally, stop the callback when all data has been streamed
    if current_index >= TOTAL_POINTS:
        curdoc().remove_periodic_callback(callback_id)
        print("All data has been streamed.")

# ---------------------------
# Layout and Add to Document
# ---------------------------
# Arrange the button and status message in a row
left_margin = Div(width=50, height=30)  # Adjust width as needed
button_row = row(left_margin, pause_button, speed_spinner, status_div, sizing_mode='stretch_width', width=800, css_classes=['centered-row'])

top_margin = Div(text="", height=20)  # 20 pixels top margin
layout = column(top_margin, button_row,candlestick_plot, volume_plot, sizing_mode='stretch_both')
curdoc().add_root(layout)
curdoc().title = "Historical Forex Data Streaming with Candlesticks"

# ---------------------------
# Add Periodic Callback
# ---------------------------   

# Add the periodic callback to stream data at defined intervals
callback_id = curdoc().add_periodic_callback(update, UPDATE_INTERVAL)

# Optional: Stream initial data to populate the plots quickly
update()
