// main.js
import { loadSequenceData, loadUnitsData } from './modules/dataLoader.js';
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

// -------------- 1) CONFIG & GLOBALS ---------------
export let NUM_COLUMNS = 3;
const RECT_WIDTH = 160;
const RECT_HEIGHT = 80;
const UNIT_SPACING_X = 360;
const UNIT_SPACING_Y = 200;

const MAX_SAMPLES = 24;
let currentSampleIndex = 0;
let stepCounter = 0; // each press of "Play Next Sample" increments step

let sequenceData = [];
let unitsData = [];
let statesHistory = [];

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

// -------------- 2) LOAD DATA ---------------
Promise.all([
  loadSequenceData(MAX_SAMPLES),
  loadUnitsData()
]).then(([seqData, uData]) => {
  sequenceData = seqData;
  unitsData = uData;

  // Initial render of sequence panel
  renderSequencePanel('#sequence-panel', sequenceData, currentSampleIndex);

  // Render initial top-n units
  updateVisualization(10);
}).catch(err => {
  console.error('Error loading CSV data', err);
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
  renderUnits(mainGroup, topUnits, 750, RECT_WIDTH, RECT_HEIGHT, handleUnitClick);
  renderConnections(
    mainGroup, topUnits, sequenceData, currentSampleIndex,
    container, tooltip,
    RECT_WIDTH, RECT_HEIGHT,
    750
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
  renderGateDetails(mainGroup, d, container, tooltip);
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
  currentSampleIndex = (currentSampleIndex + 1) % MAX_SAMPLES;

  // Re-render sequence panel to highlight new current
  renderSequencePanel('#sequence-panel', sequenceData, currentSampleIndex);

  // Generate random hidden & cell states for demonstration
  const randomHidden = generateRandomVector(50, 3);
  const randomCell   = generateRandomVector(50, 3);

  stepCounter++;
  statesHistory.push({
    stepIndex: stepCounter,
    hiddenState: randomHidden,
    cellState: randomCell
  });

  // Re-render states panel on the right
  renderStatesPanel('#states-panel', statesHistory, container, tooltip);

  // Rerun update so the input connections use new sample
  updateVisualization(+unitsCountInput.value);
});
