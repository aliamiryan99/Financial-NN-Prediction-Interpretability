from bokeh.models import (
    BoxZoomTool, DatetimeTickFormatter, HoverTool, Spinner,
    PanTool, ResetTool, SaveTool, WheelZoomTool, Button, Div
)
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc

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
            height=300,
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
            height=300,
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

class FrequencyPlotCreator:
    """Create Bokeh figures for the frequency plots."""
    def __init__(self, source_frequencies, source_frequency_importance, feature_columns, frequency_labels):
        self.source_frequencies = source_frequencies
        self.source_frequency_importance = source_frequency_importance
        self.feature_columns = feature_columns
        self.frequency_labels = frequency_labels
        self.frequency_line_plot = self.create_frequency_line_plot()
        self.frequency_importance_bar_plot = self.create_frequency_importance_plot()

    def create_frequency_line_plot(self):
        plot = figure(
            title="Frequencies by Feature",
            x_range=self.frequency_labels,
            height=300,
            sizing_mode='stretch_width',
            toolbar_location='above',
            tools=[PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool()],
            y_axis_label='Frequency Value'
        )

        # Add a line for each feature
        colors = ['blue', 'green', 'red', 'orange', 'purple']  # Extend if more features
        for feature, color in zip(self.feature_columns, colors):
            plot.line(
                x='Frequency',
                y=feature,
                source=self.source_frequencies,
                line_color=color,
                legend_label=feature,
                line_width=2
            )
            plot.circle(
                x='Frequency',
                y=feature,
                source=self.source_frequencies,
                fill_color=color,
                size=5
            )

        plot.legend.location = "top_right"
        plot.legend.click_policy = "hide"
        plot.xaxis.major_label_orientation = 1
        return plot

    def create_frequency_importance_plot(self):
        plot = figure(
            x_range=self.frequency_labels,
            height=300,
            title="Frequency Importance",
            toolbar_location=None,
            tools="",
            sizing_mode='stretch_width'
        )
        plot.vbar(
            x='Frequency',
            top='Importance',
            width=0.9,
            source=self.source_frequency_importance,
            color='teal'
        )
        plot.xgrid.grid_line_color = None
        plot.xaxis.major_label_orientation = 1
        plot.xaxis.axis_label = "Frequency"
        plot.yaxis.axis_label = "Importance"
        return plot

class WidgetCreator:
    """Create interactive widgets (buttons, spinners, etc.)."""
    def __init__(self, update_interval, model_name, interpretability_method):
        self.update_interval = update_interval
        self.model_name = model_name
        self.interpretability_method = interpretability_method
        self.pause_button, self.status_div, self.speed_spinner, self.info_div = self.create_widgets()

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
        info_div = Div(
            text=f"""
                <div style="
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    padding: 10px;
                    display: inline-block;
                    margin-left: 20px;
                    background-color: #f9f9f9;
                ">
                    <b>Model:</b> {self.model_name} | 
                    <b>Interpretability:</b> {self.interpretability_method}
                </div>
            """,
            width=400,
            height=50
        )
        return pause_button, status_div, speed_spinner, info_div
    
class LayoutManager:
    """Arrange the layout and add to the document."""
    def __init__(
        self, pause_button, speed_spinner, status_div,
        candlestick_plot, volume_plot,
        feature_importance_plot, timestep_importance_plot,
        frequency_line_plot, frequency_importance_bar_plot, info_div
    ):
        self.pause_button = pause_button
        self.speed_spinner = speed_spinner
        self.status_div = status_div
        self.info_div = info_div
        self.candlestick_plot = candlestick_plot
        self.volume_plot = volume_plot
        self.feature_importance_plot = feature_importance_plot
        self.timestep_importance_plot = timestep_importance_plot
        self.frequency_line_plot = frequency_line_plot
        self.frequency_importance_bar_plot = frequency_importance_bar_plot
        self.create_layout()

    def create_layout(self):
        left_margin = Div(width=50, height=30)
        button_row = row(
            left_margin, self.pause_button, self.speed_spinner, self.status_div, self.info_div,
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

        # The frequency plots at the bottom
        self.frequency_line_plot.sizing_mode = 'stretch_width'
        self.frequency_importance_bar_plot.sizing_mode = 'stretch_width'

        candlestick_row = row(
            self.candlestick_plot, self.timestep_importance_plot,
            sizing_mode='stretch_width'
        )

        volume_row = row(
            self.volume_plot, self.feature_importance_plot,
            sizing_mode='stretch_width'
        )

        frequency_row = row(
            self.frequency_line_plot,
            self.frequency_importance_bar_plot,
            sizing_mode='stretch_width'
        )

        layout = column(
            top_margin, button_row, candlestick_row, volume_row, frequency_row,
            sizing_mode='stretch_both'
        )
        curdoc().add_root(layout)
        curdoc().title = "Historical Forex Data Streaming with Candlesticks and Frequencies"