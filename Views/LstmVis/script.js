/*****************************************************************************
 * 1) CONFIG & GLOBALS
 *****************************************************************************/
// Default number of columns (user can override via input)
let NUM_COLUMNS = 3; 

let RECT_WIDTH = 160;
let RECT_HEIGHT = 80;
const UNIT_SPACING_X = 360;
const UNIT_SPACING_Y = 200;

const container = d3.select('#lstm-visualization');
const rect = container.node().getBoundingClientRect();
const width = rect.width;
const height = rect.height;

const svg = container
  .append('svg')
  .attr('width', width)
  .attr('height', height);

const mainGroup = svg.append('g').attr('class', 'main-group');

/** 
 * A text element for the "LSTM Layer 1" title. 
 * We'll reposition it dynamically after each render. 
 */
const layerTitle = mainGroup.append('text')
  .attr('class', 'layer-title')
  .attr('text-anchor', 'middle')
  .attr('font-size', 24)     // bigger
  .attr('font-weight', 'bold')  // bold
  .text('LSTM Layer 1');

/**
 * We'll reposition layerTitle in repositionLayerTitle()
 */

const zoom = d3.zoom()
  .scaleExtent([0.5, 8])
  .on('zoom', (event) => {
    mainGroup.attr('transform', event.transform);
  });

svg.call(zoom);

// If user clicks background, reset
svg.on('click', (event) => {
  if (event.target === svg.node()) {
    centerAllUnits();
  }
});

const tooltip = d3.select('body')
  .append('div')
  .attr('class', 'tooltip');

const defs = svg.append('defs');

// Markers
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

// Data arrays
let unitsData = [];       // from internal.csv
let sequenceData = [];    // from InputSequence.csv (Open, High, Low, Close, Volume)
const MAX_SAMPLES = 24;   // We'll display 24 steps in input panel
let currentSampleIndex = 0;

// For the hidden/cell states history
let statesHistory = [];
let stepCounter = 0; // each time we press play, we treat it as a "step"


/*****************************************************************************
 * 2) LOAD DATA
 *****************************************************************************/

// 2.1) Load input sequence
d3.csv('Assets/InputSequence.csv').then(seq => {
  // Expect columns: Time, Open, High, Low, Close, Volume, VolumeForecast
  // We only take the first 24 rows and parse them
  sequenceData = seq.slice(0, MAX_SAMPLES).map(d => {
    return {
      Open: +d.Open,
      High: +d.High,
      Low: +d.Low,
      Close: +d.Close,
      Volume: +d.Volume
    };
  });

  renderSequencePanel();

  // 2.2) Load units data (internal.csv)
  d3.csv('Assets/internal.csv').then(units => {
    const layer1Data = units.filter(d => +d.Layer === 1);
    layer1Data.forEach(d => { d.MSE_Difference = +d.MSE_Difference; });

    // Build unitsData
    unitsData = layer1Data.map(row => {
      return {
        id: +row.Unit,
        importance: row.MSE_Difference,

        // We'll keep hidden and cell vectors as zero, for example
        hiddenVector: Array(50).fill(0),
        cellVector: Array(50).fill(0),

        // Gating and outputs randomly
        gates: {
          forget: Math.random(),
          input: Math.random(),
          candidate: Math.random(),
          output: Math.random()
        },
        hiddenOutputValue: (Math.random()*2 - 1).toFixed(3),
        cellOutputValue: (Math.random()*2 - 1).toFixed(3)
      };
    });

    // Now we can do an initial update
    updateVisualization(10);
  }).catch(err => {
    console.error('Error loading internal.csv', err);
  });

}).catch(err => {
  console.error('Error loading InputSequence.csv', err);
});

/*****************************************************************************
 * 3) SEQUENCE PANEL (LEFT)
 *****************************************************************************/
function renderSequencePanel() {
  const panel = d3.select('#sequence-panel');
  panel.html('');

  panel.append('h4').text('Input Sequence (24 steps)');

  const ul = panel.append('ul');

  sequenceData.forEach((d, i) => {
    const li = ul.append('li');
    const text = `Open=${d.Open}, High=${d.High}, Low=${d.Low}, Close=${d.Close}, Vol=${d.Volume}`;
    if (i === currentSampleIndex) {
      li.style('font-weight', 'bold')
        .style('font-size', '14px')
        .text(text + ' (current)');
    } else {
      li.text(text);
    }
  });
}

/*****************************************************************************
 * 4) STATES PANEL (RIGHT)
 *****************************************************************************/
function renderStatesPanel() {
  const panel = d3.select('#states-panel');
  panel.html('');

  panel.append('h4').text('Hidden/Cell States History');

  const ul = panel.append('ul');

  statesHistory.forEach((stateObj, i) => {
    const li = ul.append('li');
    if (i === statesHistory.length - 1) {
      li.style('font-weight', 'bold')
        .style('font-size', '14px');
    }

    function showVectorTooltip(event, arr, label) {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip
        .style('visibility', 'visible')
        .html(`<b>${label}:</b><br>${JSON.stringify(arr)}`)
        .style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    }

    li.append('span')
      .text(`Step ${stateObj.stepIndex}: `);

    // HiddenState
    li.append('span')
      .text(`HiddenState${stateObj.stepIndex}`)
      .style('color', 'blue')
      .style('cursor', 'pointer')
      .on('mouseover', (event) => {
        showVectorTooltip(event, stateObj.hiddenState, `HiddenState${stateObj.stepIndex}`);
      })
      .on('mousemove', (event) => {
        const [mx, my] = d3.pointer(event, container.node());
        tooltip
          .style('left', (mx + 15) + 'px')
          .style('top', (my + 15) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });

    li.append('span').text('  '); // small gap

    // CellState
    li.append('span')
      .text(`CellState${stateObj.stepIndex}`)
      .style('color', 'green')
      .style('cursor', 'pointer')
      .on('mouseover', (event) => {
        showVectorTooltip(event, stateObj.cellState, `CellState${stateObj.stepIndex}`);
      })
      .on('mousemove', (event) => {
        const [mx, my] = d3.pointer(event, container.node());
        tooltip
          .style('left', (mx + 15) + 'px')
          .style('top', (my + 15) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });
  });
}

/*****************************************************************************
 * 5) LAYOUT & RENDERING
 *****************************************************************************/
function getGridPosition(i) {
  const col = i % NUM_COLUMNS;
  const row = Math.floor(i / NUM_COLUMNS);
  return {
    x: col * UNIT_SPACING_X + 200,
    y: row * UNIT_SPACING_Y + 100
  };
}

function renderUnits(data) {
  const unitGroups = mainGroup.selectAll('.unit-group')
    .data(data, d => d.id);

  // EXIT
  unitGroups.exit().remove();

  // ENTER
  const unitGroupsEnter = unitGroups
    .enter()
    .append('g')
    .attr('class', 'unit-group')
    .on('click', handleUnitClick);

  // Rectangle
  unitGroupsEnter
    .append('rect')
    .attr('class', 'lstm-unit')
    .attr('rx', 10)
    .attr('ry', 10)
    .attr('width', RECT_WIDTH)
    .attr('height', RECT_HEIGHT)
    .attr('opacity', 0);

  // Label above & left
  unitGroupsEnter
    .append('text')
    .attr('class', 'unit-label')
    .attr('text-anchor', 'start');

  // MERGE
  const merged = unitGroupsEnter.merge(unitGroups);

  merged.select('rect')
    .transition().duration(750)
    .attr('x', d => d.x - RECT_WIDTH/2)
    .attr('y', d => d.y - RECT_HEIGHT/2)
    .attr('opacity', 1);

  merged.select('text')
    .transition().duration(750)
    .attr('x', d => (d.x - RECT_WIDTH/2) - 5)
    .attr('y', d => (d.y - RECT_HEIGHT/2) - 5)
    .tween('text', function(d) {
      const self = d3.select(this);
      const newText = `Unit ${d.id} (Imp: ${d.importance.toFixed(4)})`;
      return () => {
        self.text(newText);
      };
    });
}

function renderConnections(data) {
  const currentInput = sequenceData[currentSampleIndex];
  const currentInputArray = [
    currentInput.Open,
    currentInput.High,
    currentInput.Low,
    currentInput.Close,
    currentInput.Volume
  ];

  const connections = [];

  data.forEach(unit => {
    const joinPoint = {
      x: unit.x - RECT_WIDTH / 2,
      y: unit.y + RECT_HEIGHT / 2 - 10
    };

    // Cell line
    connections.push({
      type: 'cell',
      vector: unit.cellVector,
      source: {
        x: unit.x - RECT_WIDTH / 2 - 60,
        y: unit.y - RECT_HEIGHT / 2 + 20
      },
      target: {
        x: unit.x - RECT_WIDTH / 2,
        y: unit.y - RECT_HEIGHT / 2 + 20
      },
      markerStart: 'url(#square)',
      markerEnd: 'url(#arrow)',
      isCurve: false
    });

    // Hidden line
    connections.push({
      type: 'hidden',
      vector: unit.hiddenVector,
      source: {
        x: unit.x - RECT_WIDTH / 2 - 60,
        y: unit.y + RECT_HEIGHT / 2 - 10
      },
      target: {
        x: unit.x - RECT_WIDTH / 2,
        y: unit.y + RECT_HEIGHT / 2 - 10
      },
      markerStart: 'url(#square)',
      markerEnd: null,
      isCurve: false
    });

    // Input line
    connections.push({
      type: 'input',
      vector: currentInputArray,
      source: {
        x: unit.x - RECT_WIDTH / 2 - 60,
        y: unit.y + 70
      },
      target: joinPoint,
      markerStart: 'url(#square)',
      markerEnd: 'url(#arrow)',
      isCurve: true
    });

    // Hidden Output
    connections.push({
      type: 'hiddenOutput',
      value: unit.hiddenOutputValue,
      source: {
        x: unit.x + RECT_WIDTH/2,
        y: unit.y + RECT_HEIGHT/2 - 10
      },
      target: {
        x: unit.x + RECT_WIDTH/2 + 70,
        y: unit.y + RECT_HEIGHT/2 - 10
      },
      markerStart: null,
      markerEnd: 'url(#arrow)',
      isCurve: false
    });

    // Cell Output
    connections.push({
      type: 'cellOutput',
      value: unit.cellOutputValue,
      source: {
        x: unit.x + RECT_WIDTH/2,
        y: unit.y - RECT_HEIGHT/2 + 20
      },
      target: {
        x: unit.x + RECT_WIDTH/2 + 70,
        y: unit.y - RECT_HEIGHT/2 + 20
      },
      markerStart: null,
      markerEnd: 'url(#arrow)',
      isCurve: false
    });
  });

  const pathSel = mainGroup.selectAll('.connection-group')
    .data(connections, (d, i) => i);

  // EXIT
  pathSel.exit().remove();

  // ENTER
  const pathEnter = pathSel.enter()
    .append('g')
    .attr('class', 'connection-group');

  // Add path
  pathEnter.append('path')
    .attr('class', 'connection');

  // Start node circles for input/cell/hidden
  pathEnter.filter(d => d.type === 'input' || d.type === 'hidden' || d.type === 'cell')
    .append('circle')
    .attr('class', 'start-node')
    .attr('r', 5);

  // Output labels
  pathEnter.filter(d => d.type === 'hiddenOutput' || d.type === 'cellOutput')
    .append('text')
    .attr('class', 'output-label')
    .attr('text-anchor', 'start')
    .text(d => d.value);

  // MERGE
  const allGroups = pathEnter.merge(pathSel);

  // Update path
  allGroups.select('path')
    .transition().duration(750)
    .attr('d', d => d.isCurve ? generateCurve(d.source, d.target) : generateLine(d.source, d.target))
    .attr('marker-start', d => d.markerStart)
    .attr('marker-end', d => d.markerEnd);

  // Start-node circles
  allGroups.select('.start-node')
    .transition().duration(750)
    .attr('cx', d => d.source.x)
    .attr('cy', d => d.source.y);

  // Output labels near end
  allGroups.select('.output-label')
    .transition().duration(750)
    .attr('x', d => d.target.x + 5)
    .attr('y', d => d.target.y + 4);

  // Tooltips
  allGroups
    .on('mouseover', (event, d) => {
      let htmlContent = '';
      if (d.type === 'input' || d.type === 'hidden' || d.type === 'cell') {
        htmlContent = `<b>${d.type} vector:</b><br>${JSON.stringify(d.vector)}`;
      } else if (d.type === 'hiddenOutput' || d.type === 'cellOutput') {
        htmlContent = `<b>${d.type}:</b> ${d.value}`;
      }
      const [mx, my] = d3.pointer(event, container.node());
      tooltip
        .style('visibility', 'visible')
        .html(htmlContent)
        .style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip
        .style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    })
    .on('mouseout', () => {
      tooltip.style('visibility', 'hidden');
    });
}

/*****************************************************************************
 * 6) LINE HELPERS
 *****************************************************************************/
function generateLine(source, target) {
  return `M ${source.x},${source.y} L ${target.x},${target.y}`;
}

function generateCurve(source, target) {
  const midX = (source.x + target.x) / 2;
  const midY = (source.y + target.y) / 2;
  return `
    M ${source.x},${source.y}
    C ${source.x},${source.y},
      ${source.x},${target.y},
      ${target.x},${target.y}
  `;
}

/*****************************************************************************
 * 7) INTERNAL LSTM UNIT DETAILS
 *****************************************************************************/
/**
 * Renders the internal LSTM unit diagram in a style similar to your reference image.
 * Replaces the old code that drew large rectangles/gates. 
 */
function renderGateDetails(unitData) {
  // Remove old internal-lstm shapes
  mainGroup.selectAll('.internal-lstm-group').remove();

  // Create a group for the internal diagram
  const internalGroup = mainGroup.append('g')
    .attr('class', 'internal-lstm-group');

  // Center of this LSTM unit
  const cx = unitData.x;
  const cy = unitData.y;

  // ====== 1) Four Gates (small orange rectangles) ======
  const gateWidth = 20, gateHeight = 12;
  const gates = [
    { name: 'Forget Gate',    x: cx - 60, y: cy + 18 },
    { name: 'Input Gate',     x: cx - 25, y: cy + 18 },
    { name: 'Cell Gate',      x: cx     , y: cy + 18 },
    { name: 'Output Gate',    x: cx + 30, y: cy + 30 }
  ];

  internalGroup.selectAll('.lstm-gate')
    .data(gates)
    .enter()
    .append('rect')
    .attr('class', 'lstm-gate')
    .attr('x', d => d.x - gateWidth / 2)
    .attr('y', d => d.y - gateHeight / 2)
    .attr('width', gateWidth)
    .attr('height', gateHeight)
    .attr('fill', '#ff944d')
    .attr('stroke', '#333')
    .attr('stroke-width', 0.5)
    .on('mouseover', (event, d) => {
      const [mx, my] = d3.pointer(event, container.node());
      const randWeight = (Math.random()*2 - 1).toFixed(3);
      tooltip.style('visibility', 'visible')
        .html(`<b>${d.name}</b><br>Weight: ${randWeight}`)
        .style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip.style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    })
    .on('mouseout', () => { tooltip.style('visibility', 'hidden'); });

  // *** NEW: Add symbol labels on gates ***
  internalGroup.selectAll('.lstm-gate-label')
    .data(gates)
    .enter()
    .append('text')
    .attr('class', 'lstm-gate-label')
    .attr('x', d => d.x)
    .attr('y', d => d.y + 3)
    .attr('text-anchor', 'middle')
    .attr('font-size', 8)
    .text(d => d.name === 'Cell Gate' ? 'tanh' : 'σ');

  // ====== 2) Five Elementwise Ops (small green circles) ======
  const opRadius = 7;
  const ops = [
    { name: 'Forget Operator',   x: cx - 60, y: cy - 20 },
    { name: 'Input Operator',    x: cx     , y: cy - 2  },
    { name: 'Add => c_t',         x: cx     , y: cy - 20 },
    { name: 'Tanh(c_t)',          x: cx + 60, y: cy - 2  },
    { name: 'Output Operator',   x: cx + 60, y: cy + 30 }
  ];

  internalGroup.selectAll('.lstm-op')
    .data(ops)
    .enter()
    .append('circle')
    .attr('class', 'lstm-op')
    .attr('r', opRadius)
    .attr('cx', d => d.x)
    .attr('cy', d => d.y)
    .attr('fill', '#ccff66')
    .attr('stroke', '#333')
    .attr('stroke-width', 0.5)
    .on('mouseover', (event, d) => {
      const [mx, my] = d3.pointer(event, container.node());
      const randomVec = Array.from({length: 5}, () => (Math.random()*2 -1).toFixed(2));
      tooltip.style('visibility', 'visible')
        .html(`<b>${d.name}</b><br>Vec: ${JSON.stringify(randomVec)}`)
        .style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip.style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    })
    .on('mouseout', () => { tooltip.style('visibility', 'hidden'); });

  // *** NEW: Add symbol labels on operators ***
  internalGroup.selectAll('.lstm-op-label')
    .data(ops)
    .enter()
    .append('text')
    .attr('class', 'lstm-op-label')
    .attr('x', d => d.x)
    .attr('y', d => d.y + 3)
    .attr('text-anchor', 'middle')
    .attr('font-size', (d) => {
      if(d.name === 'Tanh(c_t)') return 6;
      else return 8;
    })
    .text((d) => {
      if(d.name === 'Add => c_t') return '+';
      else if(d.name === 'Tanh(c_t)') return 'tanh';
      else return '×';
    });

  // ====== 3) Connection Lines (approx Colah-style) ======
  // --- Modified: Change the input gate => multiply(input) connection to be curved.
  const lines = [
    { x1: cx - 80,  y1: cy - 20, x2: ops[0].x - opRadius, y2: ops[0].y, label: 'c_{t-1}' },
    { x1: gates[0].x, y1: gates[0].y - gateHeight/2, x2: ops[0].x, y2: ops[0].y + opRadius, label: 'f_t' },
    { x1: ops[0].x + opRadius, y1: ops[0].y, x2: ops[2].x - opRadius, y2: ops[2].y, label: 'Remembered Cell State' },
    // --- Modified: Mark this connection as curved:
    { x1: gates[1].x, y1: gates[1].y - gateHeight/2, x2: ops[1].x - opRadius, y2: ops[1].y, label: 'i_t', curve: true },
    { x1: gates[2].x, y1: gates[2].y - gateHeight/2, x2: ops[1].x, y2: ops[1].y + opRadius, label: 'c~_t' },
    { x1: ops[1].x, y1: ops[1].y - opRadius, x2: ops[2].x, y2: ops[2].y + opRadius, label: 'i_t*c~_t' },
    { x1: ops[2].x + opRadius, y1: ops[2].y, x2: cx + 80, y2: cy - 20, label: 'c_t' },
    { x1: cx + 60, y1: cy - 20, x2: ops[3].x, y2: ops[3].y - opRadius, label: 'c_t' },
    { x1: gates[3].x + gateWidth/2, y1: gates[3].y, x2: ops[4].x - opRadius, y2: ops[4].y, label: 'o_t' },
    { x1: ops[3].x, y1: ops[3].y + opRadius, x2: ops[4].x, y2: ops[4].y - opRadius, label: 'Long Term Multiplier' },
    { x1: ops[4].x + opRadius, y1: ops[4].y, x2: cx + 80, y2: ops[4].y, label: 'h_t' },
    { x1: cx - 80, y1: gates[3].y, x2: gates[3].x - gateWidth/2, y2: gates[3].y, label: '[h_{t-1}, x_t]' },
    { x1: gates[0].x, y1: gates[3].y, x2: gates[0].x, y2: gates[0].y + gateHeight/2, label: '[h_{t-1}, x_t]' },
    { x1: gates[1].x, y1: gates[3].y, x2: gates[1].x, y2: gates[1].y + gateHeight/2, label: '[h_{t-1}, x_t]' },
    { x1: gates[2].x, y1: gates[3].y, x2: gates[2].x, y2: gates[2].y + gateHeight/2, label: '[h_{t-1}, x_t]' },
  ];

  // --- Split the connection rendering into curved and straight lines:
  const connEnter = internalGroup.selectAll('.internal-conn')
    .data(lines)
    .enter();

  // For curved connections:
  connEnter.filter(d => d.curve)
    .append('path')
    .attr('class', 'internal-conn')
    .attr('d', d => generateCurve({x: d.x1, y: d.y1}, {x: d.x2, y: d.y2}))
    .attr('stroke', '#333')
    .attr('stroke-width', 2)
    .attr('fill', 'none')
    .on('mouseover', (event, d) => {
      const [mx, my] = d3.pointer(event, container.node());
      const randW = (Math.random()*2 -1).toFixed(2);
      let html = d.label ? `<b>${d.label}</b><br>` : '';
      html += `Connection Weight: ${randW}`;
      tooltip.style('visibility', 'visible')
        .html(html)
        .style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip.style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
    })
    .on('mouseout', () => { tooltip.style('visibility', 'hidden'); });

  // For straight connections:
  connEnter.filter(d => !d.curve)
    .append('line')
    .attr('class', 'internal-conn')
    .attr('x1', d => d.x1)
    .attr('y1', d => d.y1)
    .attr('x2', d => d.x2)
    .attr('y2', d => d.y2)
    .attr('stroke', '#333')
    .attr('stroke-width', 2)
    .on('mouseover', (event, d) => {
      const [mx, my] = d3.pointer(event, container.node());
      const randW = (Math.random()*2 -1).toFixed(2);
      let html = d.label ? `<b>${d.label}</b><br>` : '';
      html += `Connection Weight: ${randW}`;
      tooltip.style('visibility', 'visible')
        .html(html)
        .style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip.style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
    })
    .on('mouseout', () => { tooltip.style('visibility', 'hidden'); });
}


/*****************************************************************************
 * 8) ZOOM & SELECT
 *****************************************************************************/
function handleUnitClick(event, d) {
  event.stopPropagation();
  zoomToUnit(d);
  d3.select(this).select('rect').classed('zoomed-in', true);
  renderGateDetails(d);
}

/**
 * Zoom in on the clicked LSTM unit.
 */
function zoomToUnit(unitData) {
  const scale = 3;
  svg.transition()
    .duration(750)
    .call(
      zoom.transform,
      d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(scale)
        .translate(-unitData.x, -unitData.y)
    );
}

/*****************************************************************************
 * 9) CENTER & REPOSITION TITLE
 *****************************************************************************/
function centerAllUnits() {
  const bounds = mainGroup.node().getBBox();
  if (!bounds.width || !bounds.height) return;

  const marginFactor = 0.9;
  let scale = Math.min(
    (width * marginFactor) / bounds.width,
    (height * marginFactor) / bounds.height
  );
  if (scale > 1) scale = 1;

  const tx = (width - bounds.width * scale) / 2 - bounds.x * scale;
  const ty = (height - bounds.height * scale) / 2 - bounds.y * scale;

  mainGroup.selectAll('.lstm-unit').classed('zoomed-in', false);
  // Remove old internal-lstm shapes
  mainGroup.selectAll('.internal-lstm-group').remove();

  svg.transition()
    .duration(750)
    .call(
      zoom.transform,
      d3.zoomIdentity.translate(tx, ty).scale(scale)
    )
    .on('end', () => {
      repositionLayerTitle();
    });
}

/**
 * Reposition "LSTM Layer 1" text above the bounding box of all units
 */
function repositionLayerTitle() {
  const bounds = mainGroup.node().getBBox();
  const titleX = bounds.x + bounds.width / 2;
  layerTitle
    .transition()             // Start a transition
    .duration(500)            // Transition duration in milliseconds
    .attr('x', titleX);       // Update the x attribute over the duration
}

function resetZoom() {
  centerAllUnits();
}

/*****************************************************************************
 * 10) UPDATE VISUALIZATION
 *****************************************************************************/
function updateVisualization(numUnits) {
  // Sort by importance
  const topUnits = unitsData
    .slice()
    .sort((a,b) => d3.descending(a.importance, b.importance))
    .slice(0, numUnits);

  // Compute layout
  topUnits.forEach((u, i) => {
    const {x,y} = getGridPosition(i);
    u.x = x;
    u.y = y;
  });

  mainGroup.selectAll('.lstm-unit').classed('zoomed-in', false);
  // Remove old internal-lstm shapes
  mainGroup.selectAll('.internal-lstm-group').remove();

  renderUnits(topUnits);
  renderConnections(topUnits);

  // After we render, we center + reposition title
  setTimeout(() => {
    centerAllUnits();
  }, 800);
}

/*****************************************************************************
 * 11) UI
 *****************************************************************************/
// #Units slider
const unitsCountInput = document.getElementById('units-count');
const unitsLabel = document.getElementById('units-label');
unitsCountInput.addEventListener('input', function() {
  const val = +this.value;
  unitsLabel.textContent = val;
  updateVisualization(val);
});
unitsLabel.textContent = unitsCountInput.value;

// #Columns number input
const columnsInput = document.getElementById('columns-count');
columnsInput.addEventListener('change', function() {
  const colVal = +this.value;
  if (colVal >= 1 && colVal <= 6) {
    NUM_COLUMNS = colVal;
    updateVisualization(+unitsCountInput.value);
  }
});

// Reset Zoom
document.getElementById('reset-zoom-btn')
  .addEventListener('click', resetZoom);

// "Play Next Sample"
document.getElementById('play-sample-btn')
  .addEventListener('click', () => {
    currentSampleIndex = (currentSampleIndex + 1) % MAX_SAMPLES;
    renderSequencePanel();

    // Generate random hidden & cell states for demonstration
    const randomHidden = Array.from({length: 50}, () => +(Math.random().toFixed(3)));
    const randomCell = Array.from({length: 50}, () => +(Math.random().toFixed(3)));

    stepCounter++;
    statesHistory.push({
      stepIndex: stepCounter,
      hiddenState: randomHidden,
      cellState: randomCell
    });

    // Re-render states panel
    renderStatesPanel();

    // Re-run update so the input connection uses the new sample
    updateVisualization(+unitsCountInput.value);
  });