// dataLoader.js


export function transpose(arr) {
  // Handle empty array case
  if (arr.length === 0) {
      return [];
  }
  
  // Create transposed array using map
  return arr[0].map((_, colIndex) => arr.map(row => row[colIndex]));
}

export function roundArray(arr, decimals) {
  if (!Array.isArray(arr)) {
      return typeof arr === "number" ? parseFloat(arr.toFixed(decimals)) : arr;
  }
  return arr.map(item => roundArray(item, decimals));
}


const ROUND_DECIMALS = 3;

/**
 * Loads up to `MAX_SAMPLES` from `Data/InputSequence.json`.
 */
export function loadSequenceData(MAX_SAMPLES = 24) {
  return d3.json('Data/InputSequence.json').then(jsonData => {

    // Return array of objects {Open, High, Low, Close, Volume}
    jsonData.NormalizedInput = jsonData.NormalizedInput.map(d => ({
      Open: +d.Open,
      High: +d.High,
      Low: +d.Low,
      Close: +d.Close,
      Volume: +d.Volume
    }));

    jsonData.OriginalInput = jsonData.OriginalInput.map(d => ({
      Open: +d.Open,
      High: +d.High,
      Low: +d.Low,
      Close: +d.Close,
      Volume: +d.Volume
    }));

    // Return array of objects {Open, High, Low, Close, Volume}
    return jsonData;
  });
}

/**
 * Loads LSTM “core” data from `Data/LSTMCore.json`.
 * We expect the JSON structure to look like:
 * 
 * {
 *   "UnitImportanceScores": {
 *       "1": [... array of {Unit, MSE_Difference} ...],
 *       "2": [...]
 *   },
 *   "Weights": [
 *       // Big arrays for (5,200), (50,200), (200), ...
 *   ]
 * }
 * 
 * This returns an object with:
 *   { unitsLayer1, weightsLayer1, ... } 
 * You can add more layers if needed.
 */
export function loadLSTMCoreData() {
  return d3.json('Data/LSTMCore.json').then(jsonData => {
    // 1) Grab unit importance for layer 1
    const layer1Importance = jsonData.UnitImportanceScores['1'] || [];
    // Potentially also for layer 2 if needed:
    // const layer2Importance = jsonData.UnitImportanceScores['2'] || [];

    // 2) Convert the "UnitImportanceScores" for layer 1
    // e.g. [ {id, importance, cell, gates, ...}, ... ]
    const unitsLayer1 = layer1Importance.map(row => {
      return {
        id: +row.Unit - 1,
        importance: +row.MSE_Difference,

        // For now, we can keep gates or outputs as placeholders
        gates: {
          forget: 0,
          input: 0,
          cell: 0,
          output: 0
        },
      };
    });

    // 3) Parse the LSTM model weights
    //    The user’s JSON includes “Weights”: an array of big matrices/vectors.
    //    Typically for a 1-layer LSTM with 4 gates (50 units, 5 inputs):
    //      kernel shape  = (5, 200)    // 200 = 4 gates * 50 units
    //      recurrent     = (50, 200)
    //      bias          = (200)
    //    Possibly repeated for layer 2, or plus some MLP layers, etc.
    //    Adjust as needed for your actual model architecture.
    const W = jsonData.Weights;
    // If your model’s first LSTM layer is at indices 0..2 (kernel, recurrent, bias):
    const layer1Weights = {
      kernel: transpose(roundArray(W[0], ROUND_DECIMALS)),    // shape (200, 5)
      recurrent: transpose(roundArray(W[1], ROUND_DECIMALS)), // shape (200, 50)
      bias: W[2]       // shape (200)
    };

    // If you have a second LSTM layer, you might do:
    // const layer2Weights = {
    //   kernel: W[3],    // shape (50, 200)
    //   recurrent: W[4], // shape (50, 200)
    //   bias: W[5]       // shape (200)
    // };

    return {
      unitsLayer1,
      layer1Weights
      // layer2Weights,
      // ...
    };
  });
}
