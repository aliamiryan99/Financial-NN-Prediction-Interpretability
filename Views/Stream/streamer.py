# Views/Stream/streamer.py

from bokeh.plotting import curdoc
import numpy as np

from Configs.config_schema import Config 
from Views.Stream.data_handler import (
    ConfigLoader, DataLoader, InterpretabilityDataLoader,
    FrequencyDataLoader, DataSourceManager
)
from Views.Stream.plot_handler import (
    PlotCreator, InterpretabilityPlotCreator,
    FrequencyPlotCreator, WidgetCreator, LayoutManager, ImportanceStreamPlotCreator
)

class Streamer:
    """Main application class to orchestrate the streaming visualization."""
    def __init__(self, config: Config):
        # Data Handling
        self.config_loader = ConfigLoader(config)
        self.data_loader = DataLoader(
            self.config_loader.CSV_FILE_PATH,
            self.config_loader.show_aggregator
        )
        
        # Load interpretability data
        self.interpretability_data_loader = InterpretabilityDataLoader(
            self.config_loader.INTERPRETABILITY_PATH,
            config.model_parameters.feature_columns
        )
        
        # Load frequency data
        self.frequency_data_loader = FrequencyDataLoader(
            config.data.spec_interpret_path,
            config.model_parameters.feature_columns,
            self.config_loader
        )

        # Initialize data sources
        self.data_source_manager = DataSourceManager(
            config.model_parameters.feature_columns,
            self.interpretability_data_loader.timestep_columns,
            self.frequency_data_loader.frequency_labels
        )
        
        # Create plots
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
        self.frequency_plot_creator = FrequencyPlotCreator(
            self.data_source_manager.source_frequencies,
            self.data_source_manager.source_frequency_importance,
            config.model_parameters.feature_columns,
            self.frequency_data_loader.frequency_labels
        )
        
        # Create importance whisker plots with scatter
        self.importance_stream_plot_creator = ImportanceStreamPlotCreator(
            self.data_source_manager.source_timestep_whisker,
            self.data_source_manager.source_feature_whisker,
            self.data_source_manager.source_frequency_whisker,
            self.data_source_manager.source_timestep_scatter,
            self.data_source_manager.source_feature_scatter,
            self.data_source_manager.source_frequency_scatter,
            feature_columns=config.model_parameters.feature_columns,
            timestep_column=self.interpretability_data_loader.timestep_columns,
            frequency_columns=self.frequency_data_loader.frequency_labels,
        )
        
        # Create widgets
        self.widget_creator = WidgetCreator(
            self.config_loader.UPDATE_INTERVAL,
            config.model,  
            config.time_interpretability_class,
            config.spectral_interpretability_class
        )
        
        # Initialize stream updater
        self.stream_updater = StreamUpdater(
            df=self.data_loader.df,
            feature_importance_df=self.interpretability_data_loader.feature_importance_df,
            timestep_importance_df=self.interpretability_data_loader.timestep_importance_df,
            frequencies_df=self.frequency_data_loader.frequencies_df,
            frequency_importance_df=self.frequency_data_loader.frequency_importance_df,
            source_price=self.data_source_manager.source_price,
            source_volume=self.data_source_manager.source_volume,
            source_feature_importance=self.data_source_manager.source_feature_importance,
            source_timestep_importance=self.data_source_manager.source_timestep_importance,
            source_frequencies=self.data_source_manager.source_frequencies,
            source_frequency_importance=self.data_source_manager.source_frequency_importance,
            source_timestep_whisker=self.data_source_manager.source_timestep_whisker,
            source_feature_whisker=self.data_source_manager.source_feature_whisker,
            source_frequency_whisker=self.data_source_manager.source_frequency_whisker,
            source_timestep_scatter=self.data_source_manager.source_timestep_scatter,
            source_feature_scatter=self.data_source_manager.source_feature_scatter,
            source_frequency_scatter=self.data_source_manager.source_frequency_scatter,
            feature_columns=config.model_parameters.feature_columns,
            timestep_columns=self.interpretability_data_loader.timestep_columns,
            frequency_labels=self.frequency_data_loader.frequency_labels,
            batch_size=self.config_loader.BATCH_SIZE,
            max_points=self.config_loader.MAX_POINTS,
            show_aggregator=self.config_loader.show_aggregator,
            offset=self.plot_creator.offset,
            update_interval=self.config_loader.UPDATE_INTERVAL,
            pause_button=self.widget_creator.pause_button,
            status_div=self.widget_creator.status_div,
            speed_spinner=self.widget_creator.speed_spinner
        )
        
        # Arrange layout
        self.layout_manager = LayoutManager(
            pause_button=self.widget_creator.pause_button,
            speed_spinner=self.widget_creator.speed_spinner,
            status_div=self.widget_creator.status_div,
            info_div=self.widget_creator.info_div,
            candlestick_plot=self.plot_creator.candlestick_plot,
            volume_plot=self.plot_creator.volume_plot,
            feature_importance_plot=self.interpretability_plot_creator.feature_importance_plot,
            timestep_importance_plot=self.interpretability_plot_creator.timestep_importance_plot,
            frequency_line_plot=self.frequency_plot_creator.frequency_line_plot,
            frequency_importance_bar_plot=self.frequency_plot_creator.frequency_importance_bar_plot,
            timestep_whisker_plot=self.importance_stream_plot_creator.timestep_whisker_plot,
            feature_whisker_plot=self.importance_stream_plot_creator.feature_whisker_plot,
            frequency_whisker_plot=self.importance_stream_plot_creator.frequency_whisker_plot
        )
        
        # Start streaming
        self.stream_updater.add_periodic_callback()

    def run(self):
        self.stream_updater.update()


class StreamUpdater:
    """Manage data streaming and updates."""
    def __init__(
        self, df, feature_importance_df, timestep_importance_df,
        frequencies_df, frequency_importance_df,
        source_price, source_volume,
        source_feature_importance, source_timestep_importance,
        source_frequencies, source_frequency_importance,
        source_timestep_whisker, source_feature_whisker, source_frequency_whisker,
        source_timestep_scatter, source_feature_scatter, source_frequency_scatter,
        feature_columns, timestep_columns, frequency_labels,
        batch_size, max_points,
        show_aggregator, offset, update_interval, pause_button, status_div, speed_spinner
    ):
        self.df = df
        self.feature_importance_df = feature_importance_df
        self.timestep_importance_df = timestep_importance_df
        self.frequencies_df = frequencies_df
        self.frequency_importance_df = frequency_importance_df

        self.source_price = source_price
        self.source_volume = source_volume
        self.source_feature_importance = source_feature_importance
        self.source_timestep_importance = source_timestep_importance
        self.source_frequencies = source_frequencies
        self.source_frequency_importance = source_frequency_importance

        self.source_timestep_whisker = source_timestep_whisker
        self.source_feature_whisker = source_feature_whisker
        self.source_frequency_whisker = source_frequency_whisker
        self.source_timestep_scatter = source_timestep_scatter
        self.source_feature_scatter = source_feature_scatter
        self.source_frequency_scatter = source_frequency_scatter

        self.feature_columns = feature_columns
        self.timestep_columns = timestep_columns
        self.frequency_labels = frequency_labels

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

        if self.current_index >= self.TOTAL_POINTS:
            if self.callback_id:
                curdoc().remove_periodic_callback(self.callback_id)
            print("All data has been streamed.")
            return

        # make new data samples 
        new_data = self.df.iloc[self.current_index:end_index]
        new_feature_importance = self.feature_importance_df.iloc[self.current_index:end_index]
        new_timestep_importance = self.timestep_importance_df.iloc[self.current_index:end_index]
        new_freq_data = self.frequencies_df.iloc[self.current_index:end_index]
        new_freq_importance = self.frequency_importance_df.iloc[self.current_index:end_index]

        if new_data.empty:
            return

        # Price and volume update
        new_candles = dict(
            Time=new_data['Time'].tolist(),
            Open=new_data['Open'].tolist(),
            High=new_data['High'].tolist(),
            Low=new_data['Low'].tolist(),
            Close=new_data['Close'].tolist(),
            Color=new_data['Color'].tolist()
        )

        new_volume = dict(
            x1=(new_data['Time'] - self.offset).tolist(),
            x2=(new_data['Time'] + self.offset).tolist(),
            Time=new_data['Time'].tolist(),
            Volume=new_data['Volume'].tolist(),
            PredictedVolume=new_data['VolumeForecast'].tolist(),
            PredictedVolume_Min=new_data.get('VolumeForecast_Min', new_data['VolumeForecast']).tolist(),
            PredictedVolume_Max=new_data.get('VolumeForecast_Max', new_data['VolumeForecast']).tolist()
        )

        if self.show_aggregator:
            new_volume['x1'] = new_data['Time'].tolist()
            new_volume['x2'] = new_data['Time'].tolist()
            new_volume['PredictedVolume_Min'] = new_data['VolumeForecast_Min'].tolist()
            new_volume['PredictedVolume_Max'] = new_data['VolumeForecast_Max'].tolist()

        self.source_price.stream(new_candles, rollover=self.MAX_POINTS)
        self.source_volume.stream(new_volume, rollover=self.MAX_POINTS)

        # Interpretability updates
        latest_feature_importance = new_feature_importance.iloc[-1].tolist()
        latest_timestep_importance = new_timestep_importance.iloc[-1].tolist()

        self.source_feature_importance.data = {
            'Feature': self.feature_columns,
            'Importance': latest_feature_importance
        }
        self.source_timestep_importance.data = {
            'Timestep': self.timestep_columns,
            'Importance': latest_timestep_importance
        }

        # Frequency updates: set the current frequency values
        freq_data = {'Frequency': self.frequency_labels}
        for feature in self.feature_columns:
            freq_col = [col for col in self.frequencies_df.columns if col.startswith(f"{feature}_freq_")]
            freq_values = new_freq_data[freq_col].iloc[-1].tolist()
            freq_data[feature] = freq_values

        self.source_frequencies.data = freq_data

        # Frequency importance: set the current importance values
        latest_freq_importance = new_freq_importance.iloc[-1].tolist()
        self.source_frequency_importance.data = {
            'Frequency': self.frequency_labels,
            'Importance': latest_freq_importance
        }

        # Update whisker charts
        # Stream scatter data
        # For each importance type, append the new values to the scatter sources
        # Stream Feature Importance Scatter
        feature_scatter_new = {'Feature': [], 'value': []}
        for index, feature in enumerate(self.feature_columns):
            value = latest_feature_importance[index]
            feature_scatter_new['Feature'].append(feature)
            feature_scatter_new['value'].append(value)
        self.source_feature_scatter.stream(feature_scatter_new, rollover=self.MAX_POINTS)

        # Stream Timestep Importance Scatter
        timestep_scatter_new = {'Feature': [], 'value': []}
        for index, feature in enumerate(self.timestep_columns):
            value = latest_timestep_importance[index]
            timestep_scatter_new['Feature'].append(feature)
            timestep_scatter_new['value'].append(value)
        self.source_timestep_scatter.stream(timestep_scatter_new, rollover=self.MAX_POINTS)

        # Stream Frequency Importance Scatter
        frequency_scatter_new = {'Feature': [], 'value': []}
        for index, feature in enumerate(self.frequency_labels):
            value = latest_freq_importance[index]
            frequency_scatter_new['Feature'].append(feature)
            frequency_scatter_new['value'].append(value)
        self.source_frequency_scatter.stream(frequency_scatter_new, rollover=self.MAX_POINTS)

    #    # Compute Quantiles and Update Whisker Data Sources
    #     def compute_and_update_quantiles(scatter_source, whisker_source):
    #         upper_quantiles = []
    #         lower_quantiles = []
    #         for feature in scatter_source.data['Feature']:
    #             # Filtering values for the current feature
    #             mask = np.array(scatter_source.data['Feature']) == feature
    #             feature_values = np.array(scatter_source.data['value'])[mask]
    #             if len(feature_values) >= 5:
    #                 upper = np.quantile(feature_values, 0.80)
    #                 lower = np.quantile(feature_values, 0.20)
    #             elif len(feature_values) > 0:
    #                 upper = np.max(feature_values)
    #                 lower = np.min(feature_values)
    #             else:
    #                 upper = 0
    #                 lower = 0
    #             upper_quantiles.append(upper)
    #             lower_quantiles.append(lower)
    #         whisker_source.data = dict(
    #             base=whisker_source.data['base'],
    #             upper=upper_quantiles,
    #             lower=lower_quantiles
    #         )

    #     # Update Feature Importance Whiskers
    #     compute_and_update_quantiles(self.source_feature_scatter, self.source_feature_whisker)

    #     # Update Timestep Importance Whiskers
    #     compute_and_update_quantiles(self.source_timestep_scatter, self.source_timestep_whisker)

    #     # Update Frequency Importance Whiskers
    #     compute_and_update_quantiles(self.source_frequency_scatter, self.source_frequency_whisker)

        self.current_index = end_index

        if self.current_index >= self.TOTAL_POINTS:
            if self.callback_id:
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
