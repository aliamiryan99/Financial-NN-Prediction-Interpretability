import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (BoxZoomTool, Button, ColumnDataSource,
                          DatetimeTickFormatter, Div, HoverTool, PanTool,
                          ResetTool, SaveTool, Spinner, WheelZoomTool)
from bokeh.plotting import curdoc, figure

from Configs.ConfigSchema import Config


class ForexStreamer:
    def __init__(self, config: Config):
        """Initialize the ForexStreamer with default configurations and prepare data."""
        # Load configuration
        self.load_config(config)

        # Load and prepare data
        self.df = self.load_data(self.CSV_FILE_PATH)
        self.TOTAL_POINTS = len(self.df)
        self.current_index = 0

        # Initialize data sources
        self.source_price, self.source_volume = self.initialize_sources()

        # Initialize flags and callback ID
        self.is_paused = False
        self.callback_id = None
    
    def run(self):
        """Run the ForexStreamer application."""
        # Create plots
        candlestick_plot, volume_plot, offset = self.create_plots()
        self.offset = offset

        # Setup callbacks and widgets
        self.setup_callbacks(candlestick_plot, volume_plot, offset)

        # Arrange layout and add to document
        self.create_layout()

        # Add periodic callback
        self.add_periodic_callback()

        # Optionally, stream initial data to populate the plots quickly
        self.update()

    def load_config(self, config: Config):
        """Load configuration settings."""
        self.CSV_FILE_PATH = config.data.out_path
        self.BATCH_SIZE = config.stream_visualization.batch_size  # Streaming one data point at a time
        self.UPDATE_INTERVAL = config.stream_visualization.update_interval  # Initial streaming interval in ms
        self.MAX_POINTS = config.stream_visualization.max_points  # Maximum number of points to display
        self.CANDLE_WIDTH = pd.Timedelta(f'0.7{config.stream_visualization.time_frame}')  # Width of candlesticks

    def load_data(self, csv_file_path):
        """Load and prepare data from CSV file."""
        # Load CSV data into DataFrame
        df = pd.read_csv(csv_file_path)

        # Ensure the CSV has the required columns
        required_columns = {'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VolumeForecast'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")

        # Convert 'Time' column to datetime
        df['Time'] = pd.to_datetime(df['Time'])

        # Sort data by Time in ascending order
        df = df.sort_values('Time').reset_index(drop=True)

        # Add a column to indicate whether the candle is bullish (Close >= Open)
        df['Status'] = np.where(df['Close'] >= df['Open'], 'bullish', 'bearish')

        # Add a column for candle color based on status
        df['Color'] = np.where(df['Status'] == 'bullish', 'green', 'red')

        return df

    def initialize_sources(self):
        """Initialize ColumnDataSources."""
        source_price = ColumnDataSource(data=dict(
            Time=np.array([], dtype='datetime64[ns]'),
            Open=np.array([], dtype='float64'),
            High=np.array([], dtype='float64'),
            Low=np.array([], dtype='float64'),
            Close=np.array([], dtype='float64'),
            Color=np.array([], dtype='object')
        ))

        source_volume = ColumnDataSource(data=dict(
            x1=np.array([], dtype='datetime64[ns]'),
            x2=np.array([], dtype='datetime64[ns]'),
            Time=np.array([], dtype='datetime64[ns]'),
            Volume=np.array([], dtype='float64'),
            PredictedVolume=np.array([], dtype='float64')
        ))
        return source_price, source_volume

    def create_plots(self):
        """Create Bokeh figures for the candlestick and volume plots."""
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
        candlestick_plot.segment('Time', 'High', 'Time', 'Low', source=self.source_price, color='black')

        # Add candle bodies (vbar: x, width, bottom, top)
        candlestick_plot.vbar(
            'Time',
            self.CANDLE_WIDTH,
            'Open',
            'Close',
            source=self.source_price,
            fill_color='Color',
            line_color='black'
        )

        # Volume Plot
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
        offset = self.CANDLE_WIDTH / 4

        # First volume bar (full volume)
        volume_plot.vbar(
            x='x1',
            top='Volume',
            width=self.CANDLE_WIDTH * 0.4,
            source=self.source_volume,
            color='royalblue',
            legend_label='Volume'
        )

        # Second volume bar (predicted volume)
        volume_plot.vbar(
            x='x2',
            top='PredictedVolume',
            width=self.CANDLE_WIDTH * 0.4,
            source=self.source_volume,
            color='crimson',
            legend_label='Predicted Volume'
        )

        # Add hover tool for volume plot
        volume_plot.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@Time{%F %H:%M}"),
                    ("Volume", "@Volume"),
                    ("Predicted Volume", "@PredictedVolume"),
                ],
                formatters={'@Time': 'datetime'},
                mode='vline'
            )
        )

        # Optional: Customize legend
        volume_plot.legend.location = "top_left"

        return candlestick_plot, volume_plot, offset

    def create_widgets(self):
        """Create interactive widgets (buttons, spinners, etc.)."""
        # Create a Button widget
        pause_button = Button(label="Pause", button_type="success", width=100)

        # Create a Div widget for status messages
        status_div = Div(text="<b>Status:</b> <span style='color:green;'>Streaming Active</span>", width=200, height=30)

        # Create a Spinner widget for streaming speed (UPDATE_INTERVAL)
        speed_spinner = Spinner(
            title="Streaming Delay (ms):",
            low=10,
            high=1000,
            step=10,
            value=self.UPDATE_INTERVAL,
            width=150
        )
        return pause_button, status_div, speed_spinner

    def setup_callbacks(self, candlestick_plot, volume_plot, offset):
        """Setup callbacks for widgets and events."""
        # Create widgets
        self.pause_button, self.status_div, self.speed_spinner = self.create_widgets()

        # Assign callback functions
        self.pause_button.on_click(self.toggle_pause)
        self.speed_spinner.on_change('value', self.update_interval)

        # Create plots
        self.candlestick_plot, self.volume_plot, self.offset = self.create_plots()

    def toggle_pause(self):
        """Pause or resume the data streaming."""
        if self.is_paused:
            # Resume streaming
            self.is_paused = False
            self.pause_button.label = "Pause"
            self.pause_button.button_type = "success"
            self.status_div.text = "<b>Status:</b> <span style='color:green;'>Streaming Active</span>"
            print("Streaming resumed.")
        else:
            # Pause streaming
            self.is_paused = True
            self.pause_button.label = "Resume"
            self.pause_button.button_type = "warning"
            self.status_div.text = "<b>Status:</b> <span style='color:red;'>Streaming Paused</span>"
            print("Streaming paused.")

    def update_interval(self, attr, old, new):
        """Update the streaming interval based on spinner value."""
        try:
            # Update the UPDATE_INTERVAL with the new value from the spinner
            new_interval = int(self.speed_spinner.value)

            # Remove the existing callback if it exists
            if self.callback_id is not None:
                curdoc().remove_periodic_callback(self.callback_id)

            # Update the global UPDATE_INTERVAL
            self.UPDATE_INTERVAL = new_interval

            # Add a new periodic callback with the updated interval
            self.callback_id = curdoc().add_periodic_callback(self.update, self.UPDATE_INTERVAL)

            print(f"Streaming speed updated to {self.UPDATE_INTERVAL} ms.")
        except Exception as e:
            print(f"Error updating streaming speed: {e}")

    def update(self):
        """Stream new data to the plots."""
        if self.is_paused:
            return  # Do not stream new data if paused
        # Determine the end index for the current batch
        end_index = self.current_index + self.BATCH_SIZE
        if end_index > self.TOTAL_POINTS:
            end_index = self.TOTAL_POINTS

        # Slice the DataFrame for the current batch
        new_data = self.df.iloc[self.current_index:end_index]

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
            x1=new_data['Time'] - self.offset,
            x2=new_data['Time'] + self.offset,
            Time=new_data['Time'],
            Volume=new_data['Volume'],
            PredictedVolume=new_data['VolumeForecast']
        )

        # Stream new data into the sources
        self.source_price.stream(new_candles, rollover=self.MAX_POINTS)
        self.source_volume.stream(new_volume, rollover=self.MAX_POINTS)

        # Update the current index
        self.current_index = end_index

        # Optionally, stop the callback when all data has been streamed
        if self.current_index >= self.TOTAL_POINTS:
            curdoc().remove_periodic_callback(self.callback_id)
            print("All data has been streamed.")

    def create_layout(self):
        """Arrange the layout and add to the document."""
        # Arrange the button and status message in a row
        left_margin = Div(width=50, height=30)  # Adjust width as needed
        button_row = row(left_margin, self.pause_button, self.speed_spinner, self.status_div, sizing_mode='stretch_width', width=800, css_classes=['centered-row'])

        top_margin = Div(text="", height=20)  # 20 pixels top margin
        layout = column(top_margin, button_row, self.candlestick_plot, self.volume_plot, sizing_mode='stretch_both')
        curdoc().add_root(layout)
        curdoc().title = "Historical Forex Data Streaming with Candlesticks"

    def add_periodic_callback(self):
        """Add the periodic callback to update the data."""
        self.callback_id = curdoc().add_periodic_callback(self.update, self.UPDATE_INTERVAL)

