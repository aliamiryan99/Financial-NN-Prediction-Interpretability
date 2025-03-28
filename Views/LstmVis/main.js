// main.js
import { loadSequenceData, loadLSTMCoreData, roundArray, loadScaler } from './modules/dataLoader.js';
import {
  renderSequencePanel,
  renderStatesPanel,
  getGridPosition,
  renderUnits,
  renderConnections,
  renderGateDetails,
  zoomToUnit,
  centerAllUnits,
  repositionLayerTitle,
  renderFinalLayerInputs,
  renderFinalNeuron,
  renderFinalLayerConnections
} from './modules/visualization.js';
import { dotProduct, computeAllGateOutputs, updateUnitsGateValues, computeNextStates } from './modules/utils.js'; 

// -------------- 1) CONFIG & GLOBALS ---------------
export let NUM_COLUMNS = 3;
const RECT_WIDTH = 160;
const RECT_HEIGHT = 80;
const UNIT_SPACING_X = 360;
const UNIT_SPACING_Y = 200;
const D3TRANSITION_DURATION = 750;
const DECIMAL_PRECISION = 4;

let inputSampleLenght = 0;
let currentSampleIndex = 0;
let stepCounter = 0; // each press of "Play Next Sample" increments step
let phase = 1 // the control variable to demonstrate the phase we are in right now so we chould show the correct layer of the model

let coreData = null;
let noramlSequenceData = [];  
let originalSequenceData = [];  
let sequenceList = [];
let unitsData = [];
let statesHistory = [];
let prevLayerHistory = [];
let h_t = null, h_tminus1 = null;
let c_t = null, c_tminus1 = null;
let layerWeights = null; // The object {kernel, recurrent, bias}
let the_scaler = null;

// Container
const container = d3.select('#lstm-visualization');
const rect = container.node().getBoundingClientRect();
const width = rect.width;
const height = rect.height;

const svg = container
  .append('svg')
  .attr('width', width)
  .attr('height', height);

const mainGroup = svg.append('g').attr('class', 'main-group');

// Title: "LSTM Layer 1"
const layerTitle = mainGroup.append('text')
  .attr('class', 'layer-title')
  .attr('text-anchor', 'middle')
  .attr('font-size', 24)
  .attr('font-weight', 'bold')
  .text('LSTM Layer 1');

// Zoom
const zoom = d3.zoom()
  .scaleExtent([0.5, 8])
  .on('zoom', (event) => {
    mainGroup.attr('transform', event.transform);
  });

svg.call(zoom);

// If user clicks background, reset
svg.on('click', (event) => {
  if (event.target === svg.node()) {
    centerAllUnits(svg, mainGroup, zoom, width, height, repositionLayerTitle);
  }
});

// Tooltip
const tooltip = d3.select('body')
  .append('div')
  .attr('class', 'tooltip');

// Defs/Markers
const defs = svg.append('defs');
defs.append('marker')
  .attr('id', 'arrow')
  .attr('markerWidth', 10)
  .attr('markerHeight', 10)
  .attr('refX', 6)
  .attr('refY', 3)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,0 L0,6 L6,3 z')
  .attr('fill', '#000');

defs.append('marker')
  .attr('id', 'square')
  .attr('markerWidth', 5)
  .attr('markerHeight', 5)
  .attr('refX', 2.5)
  .attr('refY', 2.5)
  .append('rect')
  .attr('x', 0)
  .attr('y', 0)
  .attr('width', 5)
  .attr('height', 5)
  .attr('fill', '#000');

// -------------- 2) LOAD JSON DATA ---------------
Promise.all([
  loadSequenceData(),
  loadLSTMCoreData(),
  loadScaler()
]).then(([inputSequenceJson, core, scaler]) => {
  coreData = core;
  the_scaler = scaler;
  noramlSequenceData = inputSequenceJson.NormalizedInput;
  originalSequenceData = inputSequenceJson.OriginalInput;
  sequenceList = noramlSequenceData.map(item => Object.values(item));
  inputSampleLenght = sequenceList.length
  unitsData = core.unitsLayer1;
  layerWeights = core.layer1Weights;
  const lenUnits = layerWeights.kernel.length/4;
  h_tminus1 = new Array(lenUnits).fill(0);
  h_t = new Array(lenUnits).fill(0);
  c_tminus1 = new Array(lenUnits).fill(0);
  c_t = new Array(lenUnits).fill(0);

  const gatesActivations = computeAllGateOutputs(sequenceList[currentSampleIndex], h_tminus1, layerWeights);
  updateUnitsGateValues(unitsData, gatesActivations);
  ({ c_t, h_t } = computeNextStates(c_tminus1, gatesActivations));

  // Render initial left panel
  renderSequencePanel('#sequence-panel', noramlSequenceData, originalSequenceData, currentSampleIndex, prevLayerHistory, phase, container, tooltip);

  // Re-render states panel on the right
  renderStatesPanel('#states-panel', statesHistory, container, tooltip, h_t, c_t);

  // Display top-10 units
  updateVisualization(10);
}).catch(err => {
  console.error('Error loading JSON data', err);
});

// -------------- 3) VISUALIZATION UPDATE ---------------
function updateVisualization(numUnits) {
  // Sort by importance descending, then slice top N
  const topUnits = unitsData
    .slice()
    .sort((a,b) => d3.descending(a.importance, b.importance))
    .slice(0, numUnits);

  // Compute layout positions
  topUnits.forEach((u, i) => {
    const { x, y } = getGridPosition(i, NUM_COLUMNS, UNIT_SPACING_X, UNIT_SPACING_Y);
    u.x = x;
    u.y = y;
  });

  // Clear zoom highlight
  mainGroup.selectAll('.lstm-unit').classed('zoomed-in', false);
  mainGroup.selectAll('.internal-lstm-group').remove();

  // Render the units and connections
  renderUnits(mainGroup, topUnits, D3TRANSITION_DURATION, RECT_WIDTH, RECT_HEIGHT, handleUnitClick);
  renderConnections(
    mainGroup, topUnits, sequenceList[currentSampleIndex],
    container, tooltip,
    RECT_WIDTH, RECT_HEIGHT,
    D3TRANSITION_DURATION,
    c_tminus1, c_t, h_tminus1, h_t
  );

  // After rendering, center and re-title
  setTimeout(() => {
    centerAllUnits(svg, mainGroup, zoom, width, height, repositionLayerTitle);
  }, 800);
}

// -------------- 4) EVENT HANDLERS ---------------
function handleUnitClick(event, d) {
  event.stopPropagation();
  // Zoom in on the clicked LSTM unit
  zoomToUnit(svg, zoom, width, height, d);

  // Mark as zoomed-in
  d3.select(this).select('rect').classed('zoomed-in', true);

  // Render internal gate details
  renderGateDetails(mainGroup, d, container, tooltip, layerWeights, sequenceList[currentSampleIndex], h_tminus1, h_t, c_tminus1, c_t);
}

// -------------- 5) UI ---------------
// #Units slider
const unitsCountInput = document.getElementById('units-count');
const unitsLabel = document.getElementById('units-label');
unitsCountInput.addEventListener('input', function() {
  const val = +this.value;
  unitsLabel.textContent = val;
  if (phase === 3) { // Assuming phase 3 is the final layer
    renderFinalLayer();
  } else {
    updateVisualization(val); // For previous layers
  }
});
unitsLabel.textContent = unitsCountInput.value;

// #Columns input
const columnsInput = document.getElementById('columns-count');
columnsInput.addEventListener('change', function() {
  const colVal = +this.value;
  if (colVal >= 1 && colVal <= 6) {
    NUM_COLUMNS = colVal;  // reassign the global
    updateVisualization(+unitsCountInput.value);
  }
});

// Reset Zoom
document.getElementById('reset-zoom-btn').addEventListener('click', () => {
  centerAllUnits(svg, mainGroup, zoom, width, height, repositionLayerTitle);
});

// Play Next Sample
document.getElementById('play-sample-btn').addEventListener('click', () => {
  currentSampleIndex = (currentSampleIndex + 1);

  // Re-render sequence panel to highlight new current
  renderSequencePanel('#sequence-panel', noramlSequenceData, originalSequenceData, currentSampleIndex, prevLayerHistory, phase, container, tooltip);

  // Store the hidden state and cell state then uppdate them
  stepCounter++;
  statesHistory.push({
    stepIndex: stepCounter,
    hiddenState: roundArray(h_t, DECIMAL_PRECISION),
    cellState: roundArray(c_t, DECIMAL_PRECISION)
  });

  // Update states
  h_tminus1 = h_t;
  c_tminus1 = c_t;

  const gatesActivations = computeAllGateOutputs(sequenceList[currentSampleIndex], h_tminus1, layerWeights);
  updateUnitsGateValues(unitsData, gatesActivations);
  ({ c_t, h_t } = computeNextStates(c_tminus1, gatesActivations));

  // Re-render states panel on the right
  renderStatesPanel('#states-panel', statesHistory, container, tooltip, h_t, c_t);

  // Rerun update so the input connections use new sample
  updateVisualization(+unitsCountInput.value);

  if (currentSampleIndex >= inputSampleLenght - 1) {
    // Optionally, process the final sample update here if needed.
    // Then disable the play button to prevent further clicks.

    if (phase == 1){

      // Store the hidden state and cell state then uppdate them
      stepCounter++;
      statesHistory.push({
        stepIndex: stepCounter,
        hiddenState: roundArray(h_t, DECIMAL_PRECISION),
        cellState: roundArray(c_t, DECIMAL_PRECISION)
      });

      unitsData = coreData.unitsLayer2;
      layerWeights = coreData.layer2Weights;
      const lenUnits = layerWeights.kernel.length/4;
      h_tminus1 = new Array(lenUnits).fill(0);
      h_t = new Array(lenUnits).fill(0);
      c_tminus1 = new Array(lenUnits).fill(0);
      c_t = new Array(lenUnits).fill(0);

      prevLayerHistory = statesHistory;
      statesHistory = [];
      currentSampleIndex = 0;
      stepCounter = 0;

      phase = 2;

      sequenceList = prevLayerHistory.map(item => item.hiddenState);

      inputSampleLenght = sequenceList.length

      mainGroup.select('.layer-title')
      .text("LSTM Layer 2")

       // Re-render sequence panel to highlight new current
      renderSequencePanel('#sequence-panel', noramlSequenceData, originalSequenceData, currentSampleIndex, prevLayerHistory, phase, container, tooltip);

      const gatesActivations = computeAllGateOutputs(sequenceList[currentSampleIndex], h_tminus1, layerWeights);
      updateUnitsGateValues(unitsData, gatesActivations);
      ({ c_t, h_t } = computeNextStates(c_tminus1, gatesActivations));

      // Re-render states panel on the right
      renderStatesPanel('#states-panel', statesHistory, container, tooltip, h_t, c_t);

      // Rerun update so the input connections use new sample
      updateVisualization(+unitsCountInput.value);

      return
    }
    else if (phase == 2) {
      phase = 3;
      unitsData = []
      // Rerun update so the input connections use new sample
      updateVisualization(0);

      mainGroup.select('.layer-title')
      .text("Final Feed Forward Layer")

      renderFinalLayer();
      document.getElementById('play-sample-btn').disabled = true;
    }
  }
});

// Add this new function after updateVisualization
function renderFinalLayer() {
  const final_h_t = h_t; // Final hidden state from phase 2
  const finalWeights = coreData.finalWeights;
  const y = dotProduct(finalWeights.weight, final_h_t) + finalWeights.bias;

  // calculate the denormalized value based on the scaler
  const denormalizedY = y * the_scaler.scale + the_scaler.mean;

  const numUnits = +unitsCountInput.value;
  const unitsLayer2 = coreData.unitsLayer2;
  const topUnits = unitsLayer2
    .slice()
    .sort((a, b) => d3.descending(a.importance, b.importance))
    .slice(0, numUnits);

  const bounds = mainGroup.node().getBBox();
  const inputX = bounds.x + bounds.width + 100;
  const spacing = 50;
  const yStart = 100;

  // Indent odd-numbered inputs
  topUnits.forEach((unit, i) => {
    const offset = i % 2 === 1 ? +20 : 0; // Indent odd indices by 50px
    unit.x = inputX + offset;
    unit.y = yStart + i * spacing;
    unit.h_t = final_h_t[unit.id];
    unit.weight = finalWeights.weight[unit.id];
  });

  const finalNeuronX = inputX + 200;
  const finalNeuronY = yStart + ((numUnits - 1) * spacing) / 2;

  renderFinalLayerInputs(mainGroup, topUnits, container, tooltip);
  renderFinalNeuron(mainGroup, finalNeuronX, finalNeuronY, y, denormalizedY, container, tooltip, finalWeights.weight); // Pass weights
  renderFinalLayerConnections(mainGroup, topUnits, finalNeuronX, finalNeuronY, container, tooltip);

  repositionLayerTitle(mainGroup);
  centerAllUnits(svg, mainGroup, zoom, width, height, repositionLayerTitle);
}
