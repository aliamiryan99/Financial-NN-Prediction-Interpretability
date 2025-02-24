export function transpose(arr) {
    // Handle empty array case
    if (arr.length === 0) {
        return [];
    }
    
    // Create transposed array using map
    return arr[0].map((_, colIndex) => arr.map(row => row[colIndex]));
}

// Utility functions
export const sigmoid = (x) => 1.0 / (1.0 + Math.exp(-x));
export const tanh = (x) => Math.tanh(x);

export function dotProduct(arrA, arrB) {
  // arrA.length === arrB.length
  let sum = 0;
  for (let i = 0; i < arrA.length; i++) {
    sum += arrA[i] * arrB[i];
  }
  return sum;
}


  /*****************************************************
 * computeAllGateOutputs
 * ---------------------
 * Given:
 *   - inputVec:  The current input x_t (array of length inputDim, e.g. 5)
 *   - hiddenVec: The previous hidden state h_{t-1} (array of length 50)
 *   - layer1Weights: An object { kernel, recurrent, bias }
 *                    where:
 *                     kernel:   shape (200, inputDim)
 *                     recurrent: shape (200, 50)
 *                     bias:     length 200
 * Returns:
 *   - An array of length 200, containing:
 *      [ f_0..f_49, i_0..i_49, c_0..c_49, o_0..o_49 ]
 *****************************************************/
export function computeAllGateOutputs(inputVec, hiddenVec, layer1Weights) {
  const { kernel, recurrent, bias } = layer1Weights;
  const totalGates = 4;     // forget, input, cell, output
  const nUnits = layer1Weights.kernel.length/totalGates;        // number of LSTM units
  const outputSize = nUnits * totalGates;  // 50 * 4 = 200
  
  // Prepare the result array
  let gateOutputs = new Array(outputSize);

  // Offsets for each gate block in [0..199]
  //   forget indices: 0..49
  //   input indices : 50..99
  //   cell indices  : 100..149
  //   output indices: 150..199
  let forgetOffset = 0;
  let inputOffset  = nUnits;      // 50
  let cellOffset   = 2 * nUnits;  // 100
  let outputOffset = 3 * nUnits;  // 150

  // For each unit i in [0..49], compute each gate
  for (let i = 0; i < nUnits; i++) {
    // ------------------ Forget Gate f_i ------------------
    let fIndex = forgetOffset + i;  // row index for forget gate
    let fKernelRow = kernel[fIndex];        // shape = (inputDim)
    let fRecurrentRow = recurrent[fIndex];  // shape = (50)
    let fBias = bias[fIndex];

    // Weighted sum => z_f = W_f * x_t + U_f * h_{t-1} + b_f
    let forgetSum = dotProduct(fKernelRow, inputVec)
                  + dotProduct(fRecurrentRow, hiddenVec)
                  + fBias;
    // Apply sigmoid
    gateOutputs[fIndex] = sigmoid(forgetSum);

    // ------------------- Input Gate i_i -------------------
    let iIndex = inputOffset + i;  // row index for input gate
    let iKernelRow = kernel[iIndex];
    let iRecurrentRow = recurrent[iIndex];
    let iBias = bias[iIndex];

    let inputSum = dotProduct(iKernelRow, inputVec)
                 + dotProduct(iRecurrentRow, hiddenVec)
                 + iBias;
    gateOutputs[iIndex] = sigmoid(inputSum);

    // ------------------ Cell Gate c_i (candidate) ------------------
    let cIndex = cellOffset + i;   // row index for candidate gate
    let cKernelRow = kernel[cIndex];
    let cRecurrentRow = recurrent[cIndex];
    let cBias = bias[cIndex];

    let cellSum = dotProduct(cKernelRow, inputVec)
                + dotProduct(cRecurrentRow, hiddenVec)
                + cBias;
    gateOutputs[cIndex] = tanh(cellSum);

    // ------------------ Output Gate o_i ------------------
    let oIndex = outputOffset + i; // row index for output gate
    let oKernelRow = kernel[oIndex];
    let oRecurrentRow = recurrent[oIndex];
    let oBias = bias[oIndex];

    let outputSum = dotProduct(oKernelRow, inputVec)
                  + dotProduct(oRecurrentRow, hiddenVec)
                  + oBias;
    gateOutputs[oIndex] = sigmoid(outputSum);
  }

  // gateOutputs now holds the 200 gate activations
  return gateOutputs;
}

/**
 * Updates the gates property of each LSTM unit in unitsData using the computed gate activations.
 * 
 * @param {Array} unitsData - Array of LSTM unit objects. Each unit should have a `gates` property with keys: forget, input, cell, output.
 * @param {Array} gateActivations - Array of computed gate outputs (length should be 4 * number of units).
 */
export function updateUnitsGateValues(unitsData, gateActivations) {
  const nUnits = unitsData.length;
  
  // Check that gateActivations has the expected length.
  if (gateActivations.length !== nUnits * 4) {
    throw new Error(`Expected gateActivations to have length ${nUnits * 4}, but got ${gateActivations.length}`);
  }

  for (let i = 0; i < nUnits; i++) {
    unitsData[i].gates.forget = gateActivations[i];
    unitsData[i].gates.input  = gateActivations[i + nUnits];
    unitsData[i].gates.cell   = gateActivations[i + 2 * nUnits];
    unitsData[i].gates.output = gateActivations[i + 3 * nUnits];
  }
}

/**
 * Computes the next hidden state (h_t) and cell state (c_t) for an LSTM layer.
 * 
 * @param {Array} c_tminus1 - Previous cell state (length: 50)
 * @param {Array} gateActivations - List of 200 gate activations (forget, input, cell, output for 50 units)
 * @returns {Object} - Contains updated cell state `c_t` and hidden state `h_t`
 */
export function computeNextStates(c_tminus1, gateActivations) {
  const nUnits = c_tminus1.length; // Should be 50
  let c_t = new Array(nUnits);
  let h_t = new Array(nUnits);

  // Offsets in gateActivations array
  let forgetOffset = 0;
  let inputOffset = nUnits;
  let cellOffset = 2 * nUnits;
  let outputOffset = 3 * nUnits;

  // Apply LSTM update equations for each unit
  for (let i = 0; i < nUnits; i++) {
      let f_t = gateActivations[forgetOffset + i];  // Forget gate activation
      let i_t = gateActivations[inputOffset + i];   // Input gate activation
      let c_hat_t = gateActivations[cellOffset + i]; // Candidate cell activation
      let o_t = gateActivations[outputOffset + i];  // Output gate activation

      // Compute next cell state
      c_t[i] = f_t * c_tminus1[i] + i_t * c_hat_t;

      // Compute next hidden state
      h_t[i] = o_t * Math.tanh(c_t[i]);
  }

  return { c_t, h_t };
}