// main.js
import { loadSequenceData, loadLSTMCoreData, roundArray } from './modules/dataLoader.js';
import {
  renderSequencePanel,
  renderStatesPanel,
  getGridPosition,
  renderUnits,
  renderConnections,
  renderGateDetails,
  zoomToUnit,
  centerAllUnits,
  repositionLayerTitle
} from './modules/visualization.js';
import { generateRandomVector } from './modules/randomUtils.js';
import { computeAllGateOutputs, updateUnitsGateValues, computeNextStates } from './modules/utils.js'; 

// -------------- 1) CONFIG & GLOBALS ---------------
export let NUM_COLUMNS = 3;
const RECT_WIDTH = 160;
const RECT_HEIGHT = 80;
const UNIT_SPACING_X = 360;
const UNIT_SPACING_Y = 200;
const D3TRANSITION_DURATION = 750;
const DECIMAL_PRECISION = 4;

const MAX_SAMPLES = 24;
let currentSampleIndex = 0;
let stepCounter = 0; // each press of "Play Next Sample" increments step

let sequenceData = [];
let sequenceList = [];
let unitsData = [];
let statesHistory = [];
let h_t = null, h_tminus1 = null;
let c_t = null, c_tminus1 = null;
let layer1Weights = null; // The object {kernel, recurrent, bias}

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
  loadSequenceData(MAX_SAMPLES),
  loadLSTMCoreData()
]).then(([seq, core]) => {
  sequenceData = seq;
  sequenceList = seq.map(item => Object.values(item));
  unitsData = core.unitsLayer1;
  layer1Weights = core.layer1Weights;
  const lenUnits = layer1Weights.kernel.length/4;
  h_tminus1 = new Array(lenUnits).fill(0);
  h_t = new Array(lenUnits).fill(0);
  c_tminus1 = new Array(lenUnits).fill(0);
  c_t = new Array(lenUnits).fill(0);

  const gatesActivations = computeAllGateOutputs(sequenceList[currentSampleIndex], h_tminus1, layer1Weights);
  updateUnitsGateValues(unitsData, gatesActivations);
  ({ c_t, h_t } = computeNextStates(c_tminus1, gatesActivations));

  // Render initial left panel
  renderSequencePanel('#sequence-panel', sequenceData, currentSampleIndex);

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
  renderGateDetails(mainGroup, d, container, tooltip, layer1Weights, sequenceList[currentSampleIndex], h_tminus1, h_t, c_tminus1, c_t);
}

// -------------- 5) UI ---------------
// #Units slider
const unitsCountInput = document.getElementById('units-count');
const unitsLabel = document.getElementById('units-label');
unitsCountInput.addEventListener('input', function() {
  const val = +this.value;
  unitsLabel.textContent = val;
  updateVisualization(val);
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
  renderSequencePanel('#sequence-panel', sequenceData, currentSampleIndex);

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

  const gatesActivations = computeAllGateOutputs(sequenceList[currentSampleIndex], h_tminus1, layer1Weights);
  updateUnitsGateValues(unitsData, gatesActivations);
  ({ c_t, h_t } = computeNextStates(c_tminus1, gatesActivations));

  // Re-render states panel on the right
  renderStatesPanel('#states-panel', statesHistory, container, tooltip, h_t, c_t);

  // Rerun update so the input connections use new sample
  updateVisualization(+unitsCountInput.value);

  if (currentSampleIndex >= MAX_SAMPLES - 1) {
    // Optionally, process the final sample update here if needed.
    // Then disable the play button to prevent further clicks.
    document.getElementById('play-sample-btn').disabled = true;
    return;
  }
});
