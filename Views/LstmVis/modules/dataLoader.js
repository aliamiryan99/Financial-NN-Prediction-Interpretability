// dataLoader.js
// Responsible for loading CSV data, preparing data arrays, injecting random properties if needed.

import { generateGates, generateRandomOutputValue } from './randomUtils.js';

/**
 * Loads the input sequence from 'Assets/InputSequence.csv'.
 * Returns a Promise that resolves to an array of 24 items:
 *   [ {Open, High, Low, Close, Volume}, ... ]
 */
export function loadSequenceData(MAX_SAMPLES = 24) {
  return d3.csv('./Assets/InputSequence.csv').then(seq => {
    // Expect columns: Time, Open, High, Low, Close, Volume, VolumeForecast
    // We only take the first 24 rows and parse them
    return seq.slice(0, MAX_SAMPLES).map(d => {
      return {
        Open: +d.Open,
        High: +d.High,
        Low: +d.Low,
        Close: +d.Close,
        Volume: +d.Volume
      };
    });
  });
}

/**
 * Loads the internal (units) data from 'Assets/internal.csv'.
 * Filters for layer 1, then injects random gating and output values.
 * Returns a Promise that resolves to an array of unit objects.
 */
export function loadUnitsData() {
  return d3.csv('./Assets/internal.csv').then(units => {
    // Filter for Layer 1
    const layer1Data = units.filter(d => +d.Layer === 1);
    layer1Data.forEach(d => {
      d.MSE_Difference = +d.MSE_Difference;
    });

    const unitsData = layer1Data.map(row => {
      return {
        id: +row.Unit,
        importance: row.MSE_Difference,

        // We'll keep hidden and cell vectors as zero for now
        hiddenVector: Array(50).fill(0),
        cellVector: Array(50).fill(0),

        // Use random gating from randomUtils
        gates: generateGates(),

        // Use random output values from randomUtils
        hiddenOutputValue: generateRandomOutputValue(),
        cellOutputValue: generateRandomOutputValue()
      };
    });

    return unitsData;
  });
}
