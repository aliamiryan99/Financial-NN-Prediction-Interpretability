import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    BoxZoomTool, Button, ColumnDataSource, DatetimeTickFormatter, Div, HoverTool,
    PanTool, ResetTool, SaveTool, Spinner, WheelZoomTool
)
from bokeh.plotting import curdoc, figure

from Configs.config_schema import Config

class Streamer:
    """Main application class to orchestrate the streaming visualization."""
    def __init__(self, config: Config):
        self.config_loader = ConfigLoader(config)
        self.data_loader = DataLoader(
            self.config_loader.CSV_FILE_PATH,
            self.config_loader.show_aggregator
        )
        self.interpretability_data_loader = InterpretabilityDataLoader(
            self.config_loader.INTERPRETABILITY_PATH,
            config.model_parameters.feature_columns
        )
        self.data_source_manager = DataSourceManager(
            config.model_parameters.feature_columns,
            self.interpretability_data_loader.timestep_columns
        )
        self.plot_creator = PlotCreator(
            self.data_source_manager.source_price,
            self.data_source_manager.source_volume,
            self.config_loader.CANDLE_WIDTH,
            self.config_loader.show_aggregator
        )
        self.interpretability_plot_creator = InterpretabilityPlotCreator(
            self.data_source_manager.source_feature_importance,
            self.data_source_manager.source_timestep_importance,
            config.model_parameters.feature_columns,
            self.interpretability_data_loader.timestep_columns
        )
        self.widget_creator = WidgetCreator(
            self.config_loader.UPDATE_INTERVAL
        )
        self.stream_updater = StreamUpdater(
            self.data_loader.df,
            self.interpretability_data_loader.feature_importance_df,
            self.interpretability_data_loader.timestep_importance_df,
            self.data_source_manager.source_price,
            self.data_source_manager.source_volume,
            self.data_source_manager.source_feature_importance,
            self.data_source_manager.source_timestep_importance,
            config.model_parameters.feature_columns,
            self.interpretability_data_loader.timestep_columns,
            self.config_loader.BATCH_SIZE,
            self.config_loader.MAX_POINTS,
            self.config_loader.show_aggregator,
            self.plot_creator.offset,
            self.config_loader.UPDATE_INTERVAL,
            self.widget_creator.pause_button,
            self.widget_creator.status_div,
            self.widget_creator.speed_spinner
        )
        self.layout_manager = LayoutManager(
            self.widget_creator.pause_button,
            self.widget_creator.speed_spinner,
            self.widget_creator.status_div,
            self.plot_creator.candlestick_plot,
            self.plot_creator.volume_plot,
            self.interpretability_plot_creator.feature_importance_plot,
            self.interpretability_plot_creator.timestep_importance_plot
        )
        self.stream_updater.add_periodic_callback()

    def run(self):
        self.stream_updater.update()

class ConfigLoader:
    """Load configuration settings."""
    def __init__(self, config: Config):
        self.config = config
        self.load_config()

    def load_config(self):
        self.show_aggregator = self.config.stream_visualization.show_aggregator
        if self.show_aggregator:
            self.config.data.out_path = f"Results/{self.config.data.name}/EnsembleAggregator.csv"
        self.CSV_FILE_PATH = self.config.data.out_path
        self.BATCH_SIZE = self.config.stream_visualization.batch_size
        self.UPDATE_INTERVAL = self.config.stream_visualization.update_interval
        self.MAX_POINTS = self.config.stream_visualization.max_points
        self.CANDLE_WIDTH = pd.Timedelta(f'0.7{self.config.stream_visualization.time_frame}')
        # Changed as per your request
        self.INTERPRETABILITY_PATH = self.config.data.interpret_path

class DataLoader:
    """Load and prepare data from CSV file."""
    def __init__(self, csv_file_path, show_aggregator):
        self.csv_file_path = csv_file_path
        self.show_aggregator = show_aggregator
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_file_path)

        if self.show_aggregator:
            required_columns = {
                'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VolumeForecast',
                'VolumeForecast_Min', 'VolumeForecast_Max', 'VolumeForecast_Var'
            }
        else:
            required_columns = {'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VolumeForecast'}

        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")

        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        df['Status'] = np.where(df['Close'] >= df['Open'], 'bullish', 'bearish')
        df['Color'] = np.where(df['Status'] == 'bullish', 'green', 'red')

        return df

class InterpretabilityDataLoader:
    """Load and prepare interpretability data from a single CSV file."""
    def __init__(self, interpretability_path, feature_columns):
        self.interpretability_path = interpretability_path
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)
        self.feature_importance_df, self.timestep_importance_df, self.timestep_columns = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.interpretability_path)
        # First columns are feature importance
        feature_importance_cols = df.columns[:self.num_features]
        # Remaining columns are timestep importance
        timestep_importance_cols = df.columns[self.num_features:]
        # Generate timestep column names if not present
        if len(timestep_importance_cols) == 0:
            timestep_importance_cols = [f"Timestep_{i+1}" for i in range(df.shape[1] - self.num_features)]
            df.columns = list(feature_importance_cols) + timestep_importance_cols

        feature_importance_df = df[feature_importance_cols]
        timestep_importance_df = df[timestep_importance_cols]
        return feature_importance_df, timestep_importance_df, timestep_importance_cols.tolist()

class DataSourceManager:
    """Initialize ColumnDataSources for price, volume, and interpretability data."""
    def __init__(self, feature_columns, timestep_columns):
        self.source_price = self.initialize_price_source()
        self.source_volume = self.initialize_volume_source()
        self.source_feature_importance = self.initialize_feature_importance_source(feature_columns)
        self.source_timestep_importance = self.initialize_timestep_importance_source(timestep_columns)

    @staticmethod
    def initialize_price_source():
        return ColumnDataSource(data=dict(
            Time=np.array([], dtype='datetime64[ns]'),
            Open=np.array([], dtype='float64'),
            High=np.array([], dtype='float64'),
            Low=np.array([], dtype='float64'),
            Close=np.array([], dtype='float64'),
            Color=np.array([], dtype='object')
        ))

    @staticmethod
    def initialize_volume_source():
        return ColumnDataSource(data=dict(
            x1=np.array([], dtype='datetime64[ns]'),
            x2=np.array([], dtype='datetime64[ns]'),
            Time=np.array([], dtype='datetime64[ns]'),
            Volume=np.array([], dtype='float64'),
            PredictedVolume=np.array([], dtype='float64'),
            PredictedVolume_Min=np.array([], dtype='float64'),
            PredictedVolume_Max=np.array([], dtype='float64')
        ))

    @staticmethod
    def initialize_feature_importance_source(feature_columns):
        return ColumnDataSource(data=dict(
            Feature=feature_columns,
            Importance=[0]*len(feature_columns)
        ))

    @staticmethod
    def initialize_timestep_importance_source(timestep_columns):
        return ColumnDataSource(data=dict(
            Timestep=timestep_columns,
            Importance=[0]*len(timestep_columns)
        ))

class PlotCreator:
    """Create Bokeh figures for the candlestick and volume plots."""
    def __init__(self, source_price, source_volume, candle_width, show_aggregator):
        self.source_price = source_price
        self.source_volume = source_volume
        self.candle_width = candle_width
        self.show_aggregator = show_aggregator
        self.candlestick_plot, self.volume_plot, self.offset = self.create_plots()

    def create_plots(self):
        # Candlestick Plot
        candlestick_plot = figure(
            title='XAU/USD Candlestick Streaming',
            x_axis_type='datetime',
            height=500,
            width=800,
            sizing_mode='stretch_width',
            toolbar_location='above',
            tools=[],
            y_axis_label='Price (USD)'
        )

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

        candlestick_plot.xaxis.formatter = DatetimeTickFormatter(
            hours=["%Y-%m-%d %H:%M"],
            days=["%Y-%m-%d"],
            months=["%Y-%m"],
            years=["%Y"],
        )

        candlestick_plot.segment('Time', 'High', 'Time', 'Low', source=self.source_price, color='black')

        candlestick_plot.vbar(
            'Time',
            self.candle_width,
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
            x_range=candlestick_plot.x_range,
            tools=[],
            y_axis_label='Volume'
        )

        volume_plot.add_tools(
            PanTool(),
            WheelZoomTool(),
            BoxZoomTool(),
            ResetTool(),
            SaveTool(),
        )

        volume_plot.xaxis.formatter = DatetimeTickFormatter(
            hours=["%Y-%m-%d %H:%M"],
            days=["%Y-%m-%d"],
            months=["%Y-%m"],
            years=["%Y"],
        )

        offset = self.candle_width / 4

        if self.show_aggregator:
            actual_volume_vbar = volume_plot.vbar(
                x='x1',
                top='Volume',
                width=self.candle_width * 0.6,
                source=self.source_volume,
                color='royalblue',
                legend_label='Actual Volume'
            )
            volume_plot.vbar(
                x='x2',
                top='PredictedVolume_Max',
                bottom='PredictedVolume_Min',
                width=self.candle_width * 0.8,
                source=self.source_volume,
                fill_color=None,
                line_color='crimson',
                legend_label='Predicted Volume Range'
            )
            volume_plot.add_tools(
                HoverTool(
                    renderers=[actual_volume_vbar],
                    tooltips=[
                        ("Time", "@Time{%F %H:%M}"),
                        ("Volume", "@Volume"),
                        ("Predicted Volume", "@PredictedVolume"),
                        ("Predicted Volume Min", "@PredictedVolume_Min"),
                        ("Predicted Volume Max", "@PredictedVolume_Max"),
                    ],
                    formatters={'@Time': 'datetime'},
                    mode='vline'
                )
            )
        else:
            actual_volume_vbar = volume_plot.vbar(
                x='x1',
                top='Volume',
                width=self.candle_width * 0.4,
                source=self.source_volume,
                color='royalblue',
                legend_label='Actual Volume'
            )
            volume_plot.vbar(
                x='x2',
                top='PredictedVolume',
                width=self.candle_width * 0.4,
                source=self.source_volume,
                color='crimson',
                legend_label='Predicted Volume'
            )
            volume_plot.add_tools(
                HoverTool(
                    renderers=[actual_volume_vbar],
                    tooltips=[
                        ("Time", "@Time{%F %H:%M}"),
                        ("Volume", "@Volume"),
                        ("Predicted Volume", "@PredictedVolume"),
                    ],
                    formatters={'@Time': 'datetime'},
                    mode='vline'
                )
            )

        volume_plot.legend.location = "top_left"

        return candlestick_plot, volume_plot, offset

class InterpretabilityPlotCreator:
    """Create Bokeh figures for the interpretability plots."""
    def __init__(self, source_feature_importance, source_timestep_importance, feature_columns, timestep_columns):
        self.source_feature_importance = source_feature_importance
        self.source_timestep_importance = source_timestep_importance
        self.feature_columns = feature_columns
        self.timestep_columns = timestep_columns
        self.feature_importance_plot = self.create_feature_importance_plot()
        self.timestep_importance_plot = self.create_timestep_importance_plot()

    def create_feature_importance_plot(self):
        # Create a vertical bar chart for feature importance
        plot = figure(
            x_range=self.feature_columns,
            height=200,
            width=300,
            title="Feature Importance",
            toolbar_location=None,
            tools="",
            sizing_mode='fixed'
        )
        plot.vbar(
            x='Feature',
            top='Importance',
            width=0.9,
            source=self.source_feature_importance,
            color='navy'
        )
        plot.xgrid.grid_line_color = None
        plot.xaxis.major_label_orientation = 1
        plot.xaxis.axis_label = "Feature"
        plot.yaxis.axis_label = "Importance"
        return plot

    def create_timestep_importance_plot(self):
        # Create a horizontal bar chart for timestep importance
        plot = figure(
            y_range=self.timestep_columns[::-1],
            height=500,
            width=300,
            title="Timestep Importance",
            toolbar_location=None,
            tools="",
            sizing_mode='fixed'
        )
        plot.hbar(
            y='Timestep',
            right='Importance',
            height=0.8,
            source=self.source_timestep_importance,
            color='navy'
        )
        plot.ygrid.grid_line_color = None
        plot.xaxis.axis_label = "Importance"
        plot.yaxis.axis_label = "Timestep"
        return plot

class WidgetCreator:
    """Create interactive widgets (buttons, spinners, etc.)."""
    def __init__(self, update_interval):
        self.update_interval = update_interval
        self.pause_button, self.status_div, self.speed_spinner = self.create_widgets()

    def create_widgets(self):
        pause_button = Button(label="Pause", button_type="success", width=100)
        status_div = Div(text="<b>Status:</b> <span style='color:green;'>Streaming Active</span>")
        speed_spinner = Spinner(
            title="Streaming Delay (ms):",
            low=10,
            high=1000,
            step=10,
            value=self.update_interval,
            width=150
        )
        return pause_button, status_div, speed_spinner

class StreamUpdater:
    """Manage data streaming and updates."""
    def __init__(
        self, df, feature_importance_df, timestep_importance_df,
        source_price, source_volume, source_feature_importance, source_timestep_importance,
        feature_columns, timestep_columns,
        batch_size, max_points,
        show_aggregator, offset, update_interval, pause_button, status_div, speed_spinner
    ):
        self.df = df
        self.feature_importance_df = feature_importance_df
        self.timestep_importance_df = timestep_importance_df
        self.source_price = source_price
        self.source_volume = source_volume
        self.source_feature_importance = source_feature_importance
        self.source_timestep_importance = source_timestep_importance
        self.feature_columns = feature_columns
        self.timestep_columns = timestep_columns
        self.BATCH_SIZE = batch_size
        self.MAX_POINTS = max_points
        self.show_aggregator = show_aggregator
        self.offset = offset
        self.UPDATE_INTERVAL = update_interval
        self.pause_button = pause_button
        self.status_div = status_div
        self.speed_spinner = speed_spinner

        self.TOTAL_POINTS = len(self.df)
        self.current_index = 0
        self.is_paused = False
        self.callback_id = None

        self.pause_button.on_click(self.toggle_pause)
        self.speed_spinner.on_change('value', self.update_interval_callback)

    def update(self):
        if self.is_paused:
            return

        end_index = self.current_index + self.BATCH_SIZE
        if end_index > self.TOTAL_POINTS:
            end_index = self.TOTAL_POINTS

        new_data = self.df.iloc[self.current_index:end_index]
        new_feature_importance = self.feature_importance_df.iloc[self.current_index:end_index]
        new_timestep_importance = self.timestep_importance_df.iloc[self.current_index:end_index]

        if new_data.empty:
            return

        new_candles = dict(
            Time=new_data['Time'],
            Open=new_data['Open'],
            High=new_data['High'],
            Low=new_data['Low'],
            Close=new_data['Close'],
            Color=new_data['Color']
        )

        new_volume = dict(
            x1=new_data['Time'] - self.offset,
            x2=new_data['Time'] + self.offset,
            Time=new_data['Time'],
            Volume=new_data['Volume'],
            PredictedVolume=new_data['VolumeForecast'],
            PredictedVolume_Min=new_data['VolumeForecast'],
            PredictedVolume_Max=new_data['VolumeForecast']
        )

        if self.show_aggregator:
            new_volume['x1'] = new_data['Time']
            new_volume['x2'] = new_data['Time']
            new_volume['PredictedVolume_Min'] = new_data['VolumeForecast_Min']
            new_volume['PredictedVolume_Max'] = new_data['VolumeForecast_Max']

        self.source_price.stream(new_candles, rollover=self.MAX_POINTS)
        self.source_volume.stream(new_volume, rollover=self.MAX_POINTS)

        # Update interpretability data sources with the latest values
        latest_feature_importance = new_feature_importance.iloc[-1]
        latest_timestep_importance = new_timestep_importance.iloc[-1]

        self.source_feature_importance.data = {
            'Feature': self.feature_columns,
            'Importance': latest_feature_importance.values
        }
        self.source_timestep_importance.data = {
            'Timestep': self.timestep_columns,
            'Importance': latest_timestep_importance.values
        }

        self.current_index = end_index

        if self.current_index >= self.TOTAL_POINTS:
            curdoc().remove_periodic_callback(self.callback_id)
            print("All data has been streamed.")

    def toggle_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.pause_button.label = "Pause"
            self.pause_button.button_type = "success"
            self.status_div.text = "<b>Status:</b> <span style='color:green;'>Streaming Active</span>"
            print("Streaming resumed.")
        else:
            self.is_paused = True
            self.pause_button.label = "Resume"
            self.pause_button.button_type = "warning"
            self.status_div.text = "<b>Status:</b> <span style='color:red;'>Streaming Paused</span>"
            print("Streaming paused.")

    def update_interval_callback(self, attr, old, new):
        try:
            new_interval = int(self.speed_spinner.value)
            if self.callback_id is not None:
                curdoc().remove_periodic_callback(self.callback_id)
            self.UPDATE_INTERVAL = new_interval
            self.callback_id = curdoc().add_periodic_callback(self.update, self.UPDATE_INTERVAL)
            print(f"Streaming speed updated to {self.UPDATE_INTERVAL} ms.")
        except Exception as e:
            print(f"Error updating streaming speed: {e}")

    def add_periodic_callback(self):
        self.callback_id = curdoc().add_periodic_callback(self.update, self.UPDATE_INTERVAL)

class LayoutManager:
    """Arrange the layout and add to the document."""
    def __init__(
        self, pause_button, speed_spinner, status_div,
        candlestick_plot, volume_plot,
        feature_importance_plot, timestep_importance_plot
    ):
        self.pause_button = pause_button
        self.speed_spinner = speed_spinner
        self.status_div = status_div
        self.candlestick_plot = candlestick_plot
        self.volume_plot = volume_plot
        self.feature_importance_plot = feature_importance_plot
        self.timestep_importance_plot = timestep_importance_plot
        self.create_layout()

    def create_layout(self):
        left_margin = Div(width=50, height=30)
        button_row = row(
            left_margin, self.pause_button, self.speed_spinner, self.status_div,
            sizing_mode='stretch_width',
            width=800,
            css_classes=['centered-row'],
            styles={'align-items': 'flex-end', 'justify-content': 'flex-start'},
        )
        top_margin = Div(text="", height=20)

        # Adjust plot sizes
        self.candlestick_plot.sizing_mode = 'stretch_both'
        self.timestep_importance_plot.sizing_mode = 'fixed'
        self.timestep_importance_plot.width = 300

        self.volume_plot.sizing_mode = 'stretch_both'
        self.feature_importance_plot.sizing_mode = 'fixed'
        self.feature_importance_plot.width = 300

        # Create rows for the plots
        candlestick_row = row(
            self.candlestick_plot, self.timestep_importance_plot,
            sizing_mode='stretch_width'
        )

        volume_row = row(
            self.volume_plot, self.feature_importance_plot,
            sizing_mode='stretch_width'
        )

        layout = column(
            top_margin, button_row, candlestick_row, volume_row,
            sizing_mode='stretch_both'
        )
        curdoc().add_root(layout)
        curdoc().title = "Historical Forex Data Streaming with Candlesticks"
