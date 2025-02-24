// visualization.js
import {
  generateRandomWeight,
  generateRandomVector
} from './randomUtils.js';

import { roundArray } from './dataLoader.js';

import { sigmoid, tanh, computeAllGateOutputs } from './utils.js'; 

const DECIMAL_POINTS = 4;

/**
 * Renders the left sequence panel.
 */
export function renderSequencePanel(panelSelector, sequenceData, currentSampleIndex) {
  const panel = d3.select(panelSelector);
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

/**
 * Renders the right states panel (history of hidden/cell states).
 */
export function renderStatesPanel(panelSelector, statesHistory, container, tooltip, h_t, c_t) {
  const panel = d3.select(panelSelector);
  panel.html('');

  const DICIMAL_PRECISION = 4;
  h_t = roundArray(h_t, DICIMAL_PRECISION);
  c_t = roundArray(c_t, DICIMAL_PRECISION);

  // Helper function to convert array to dictionary format
  function arrayToDict(arr) {
    return arr.reduce((dict, value, index) => {
      dict[index] = value;
      return dict;
    }, {});
  }

  // For showing vector tooltip with dynamic column layout
  function showVectorTooltip(event, arr, label) {
    const [mx, my] = d3.pointer(event, container.node());
    const dictFormat = arrayToDict(arr);
    const entries = Object.entries(dictFormat);

    // Define tooltip constraints
    const maxWidth = 400;  // Maximum tooltip width in pixels (adjustable)
    const maxHeight = 300; // Maximum tooltip height in pixels (adjustable)
    const itemWidth = 100; // Minimum width per item (adjustable)
    const itemHeight = 20; // Approximate height per item (adjustable)

    // Calculate number of columns to fit within maxWidth
    const columns = Math.min(Math.floor(maxWidth / itemWidth), entries.length);
    const rows = Math.ceil(entries.length / columns);

    // Ensure total height fits within maxHeight
    const adjustedColumns = Math.ceil(entries.length / Math.floor(maxHeight / itemHeight));
    const finalColumns = Math.max(1, Math.min(columns, adjustedColumns));
    const itemsPerColumn = Math.ceil(entries.length / finalColumns);

    // Build HTML content
    let htmlContent = `<b>${label}:</b><br><div style="display: flex; flex-direction: row; width: ${finalColumns * itemWidth} px;">`;
    
    for (let col = 0; col < finalColumns; col++) {
      const startIdx = col * itemsPerColumn;
      const endIdx = Math.min(startIdx + itemsPerColumn, entries.length);
      htmlContent += '<div style="display: flex; flex-direction: column; width: 100px;">';
      
      for (let i = startIdx; i < endIdx; i++) {
        const [index, value] = entries[i];
        htmlContent += `<span>${index}: ${value}</span>`;
      }
      
      htmlContent += '</div>';
    }
    htmlContent += '</div>';

    tooltip
      .style('visibility', 'visible')
      .html(htmlContent)
      .style('left', (mx - 60) + 'px')
      .style('top', (my) + 'px')
      .style('max-width', `${finalColumns * itemWidth}px`); // Set tooltip width dynamically
  }

  function appendSpan(d3Object, label, color, arr) {
    // Hidden State link
    d3Object.append('span')
      .text(label)
      .style('color', color)
      .style('cursor', 'pointer')
      .on('mouseover', (event) => {
        showVectorTooltip(event, arr, label);
      })
      .on('mousemove', (event) => {
        const [mx, my] = d3.pointer(event, container.node());
        tooltip
          .style('left', (mx - 60) + 'px')
          .style('top', (my) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });
  }

  panel.append('h4').text('Current Hidden/Cell State');

  const currentStates = panel.append('h4');
  appendSpan(currentStates, `HiddenState`, 'blue', h_t);
  currentStates.append('span').text('     '); // small gap
  appendSpan(currentStates, `CellState`, 'green', c_t);

  panel.append('h4').text('Hidden/Cell States History');

  const ul = panel.append('ul');

  statesHistory.forEach((stateObj, i) => {
    const li = ul.append('li');
    if (i === statesHistory.length - 1) {
      li.style('font-weight', 'bold').style('font-size', '14px');
    }

    li.append('span').text(`Step ${stateObj.stepIndex}: `);

    // Hidden State link
    appendSpan(li, `HiddenState${stateObj.stepIndex}`, 'blue', stateObj.hiddenState);

    li.append('span').text('  '); // small gap

    // Cell State link
    appendSpan(li, `CellState${stateObj.stepIndex}`, 'green', stateObj.cellState);
  });
}

/**
 * Given a unit index and total columns, returns the (x,y) position in the grid.
 */
export function getGridPosition(i, NUM_COLUMNS, UNIT_SPACING_X, UNIT_SPACING_Y) {
  const col = i % NUM_COLUMNS;
  const row = Math.floor(i / NUM_COLUMNS);
  return {
    x: col * UNIT_SPACING_X + 200,
    y: row * UNIT_SPACING_Y + 100
  };
}

// Helpers for lines
function generateLine(source, target) {
  return `M ${source.x},${source.y} L ${target.x},${target.y}`;
}

function generateCurve(source, target) {
  // A simple curve from source to target
  return `
    M ${source.x},${source.y}
    C ${source.x},${source.y},
      ${source.x},${target.y},
      ${target.x},${target.y}
  `;
}

/**
 * Renders the LSTM units (rectangles + labels).
 */
export function renderUnits(mainGroup, data, d3TransitionDuration, RECT_WIDTH, RECT_HEIGHT, handleUnitClick) {
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
    .transition().duration(d3TransitionDuration)
    .attr('x', d => d.x - RECT_WIDTH/2)
    .attr('y', d => d.y - RECT_HEIGHT/2)
    .attr('opacity', 1);

  merged.select('text')
    .transition().duration(d3TransitionDuration)
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

/**
 * Renders the lines (“connections”) around each unit (input, cell, hidden, outputs).
 */
export function renderConnections(
  mainGroup, data, x_t,
  container, tooltip,
  RECT_WIDTH, RECT_HEIGHT,
  d3TransitionDuration,
  c_tminus1, c_t, h_tminus1, h_t
) {

  const connections = [];
  data.forEach(unit => {
    const joinPoint = {
      x: unit.x - RECT_WIDTH / 2,
      y: unit.y + RECT_HEIGHT / 2 - 10
    };
    // Cell line
    connections.push({
      type: 'cell',
      value: c_tminus1[unit.id].toFixed(DECIMAL_POINTS),
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
      vector: roundArray(h_tminus1, DECIMAL_POINTS),
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
      vector: x_t,
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
      type: 'hidden output',
      value: h_t[unit.id].toFixed(DECIMAL_POINTS),
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
      type: 'cell output',
      value: c_t[unit.id].toFixed(DECIMAL_POINTS),
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
  pathEnter
    .filter(d => d.type === 'input' || d.type === 'hidden' || d.type === 'cell')
    .append('circle')
    .attr('class', 'start-node')
    .attr('r', 5);

  // Output labels
  pathEnter
    .filter(d => d.type === 'hidden output' || d.type === 'cell output')
    .append('text')
    .attr('class', 'output-label')
    .attr('text-anchor', 'start')
    .text(d => d.value);

  // MERGE
  const allGroups = pathEnter.merge(pathSel);

  // Update path
  allGroups.select('path')
    .transition().duration(d3TransitionDuration)
    .attr('d', d => d.isCurve
      ? generateCurve(d.source, d.target)
      : generateLine(d.source, d.target)
    )
    .attr('marker-start', d => d.markerStart)
    .attr('marker-end', d => d.markerEnd);

  // Start-node circles
  allGroups.select('.start-node')
    .transition().duration(d3TransitionDuration)
    .attr('cx', d => d.source.x)
    .attr('cy', d => d.source.y);

  // Output labels near end
  allGroups.select('.output-label')
    .text(d => d.value) // Update the text content
    .transition().duration(d3TransitionDuration)
    .attr('x', d => d.target.x + 5)
    .attr('y', d => d.target.y + 5);

  // Tooltips
  allGroups
    .on('mouseover', (event, d) => {
      let htmlContent = '';
      if (d.type === 'input' || d.type === 'hidden') {
        htmlContent = `<b>${d.type} vector:</b><br>${JSON.stringify(d.vector)}`;
      } else if (d.type === 'hidden output' || d.type === 'cell output' || d.type === "cell") {
        htmlContent = `<b>${d.type}:</b> ${d.value}`;
      }
      const [mx, my] = d3.pointer(event, container.node());
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      tooltip
        .style('visibility', 'visible')
        .html(htmlContent)
        .style('left', (mx + svgRect.left + 15) + 'px')
        .style('top', (my + svgRect.top + 15) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      tooltip
        .style('left', (mx + svgRect.left  + 15) + 'px')
        .style('top', (my + svgRect.top  + 15) + 'px');
    })
    .on('mouseout', () => {
      tooltip.style('visibility', 'hidden');
    });
}

/**
 * Renders internal LSTM gate details for a single unit, but with real weight usage.
 * @param {Object} mainGroup    The D3 selection of your main <g>
 * @param {Object} unitData     The single unit object (has .x, .y, .id, etc.)
 * @param {Object} container    The D3 container for pointer calculations
 * @param {Object} tooltip      The D3 tooltip selection
 * @param {Object} lstmWeights  { kernel: [...], recurrent: [...], bias: [...] }
 */
export function renderGateDetails(mainGroup, unitData, container, tooltip, lstmWeights, x_t, h_tminus1, h_t, c_tminus1, c_t) {
  // Remove any older diagram
  mainGroup.selectAll('.internal-lstm-group').remove();

  // Create a group for internal diagram
  const internalGroup = mainGroup.append('g')
    .attr('class', 'internal-lstm-group');

  const cx = unitData.x;
  const cy = unitData.y;
  const id = unitData.id;
  const n_units = lstmWeights.kernel.length / 4;

  console.log(lstmWeights);

  // (1) Gates
  const gateWidth = 20, gateHeight = 12;
  const gates = [
    { name: 'Forget Gate',    x: cx - 60, y: cy + 18, kernel:lstmWeights.kernel[id], recurrent:lstmWeights.recurrent[id], bias:lstmWeights.bias[id]},
    { name: 'Input Gate',     x: cx - 25, y: cy + 18, kernel:lstmWeights.kernel[id + n_units], recurrent:lstmWeights.recurrent[id + n_units], bias:lstmWeights.bias[id + n_units] },
    { name: 'Cell Gate',      x: cx     , y: cy + 18, kernel:lstmWeights.kernel[id + 2*n_units], recurrent:lstmWeights.recurrent[id + 2*n_units], bias:lstmWeights.bias[id + 2*n_units] },
    { name: 'Output Gate',    x: cx + 30, y: cy + 30, kernel:lstmWeights.kernel[id + 3*n_units], recurrent:lstmWeights.recurrent[id + 3*n_units], bias:lstmWeights.bias[id + 3*n_units] }
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
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      // Suppose we do a dummy input & hidden state:
      // (In a real app, you might use actual current input or previous hidden)
      tooltip
        .style('visibility', 'visible')
        .html(`
          <b>${d.name}</b><br>
          Unit index = ${unitData.id}<br>
          Weights = {<br>
               Kernel: ${d.kernel}<br>
               Recurrenct: ${d.recurrent}<br>
               Bias: ${d.bias}<br>
        }
        `)
        .style('left', (mx + svgRect.left + 40) + 'px')
        .style('top', (my + svgRect.top + 40) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      tooltip
        .style('left', (mx + svgRect.left + 40) + 'px')
        .style('top', (my + svgRect.top + 50) + 'px');
    })
    .on('mouseout', () => {
      tooltip.style('visibility', 'hidden');
    });

  // Labels on gates
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

  // (2) Elementwise Ops
  const opRadius = 7;
  const ops = [
    { name: 'Forget Operator',   x: cx - 60, y: cy - 20 },
    { name: 'Input Operator',    x: cx     , y: cy - 2  },
    { name: 'Add => c_t',        x: cx     , y: cy - 20 },
    { name: 'Tanh(c_t)',         x: cx + 60, y: cy - 2  },
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

  // Labels on ops
  internalGroup.selectAll('.lstm-op-label')
    .data(ops)
    .enter()
    .append('text')
    .attr('class', 'lstm-op-label')
    .attr('x', d => d.x)
    .attr('y', d => d.y + 3)
    .attr('text-anchor', 'middle')
    .attr('font-size', d => (d.name === 'Tanh(c_t)' ? 6 : 8))
    .text(d => {
      if (d.name === 'Add => c_t') return '+';
      if (d.name === 'Tanh(c_t)') return 'tanh';
      return '×';
    });

  // (3) Connection lines
  // Mark some as curve, some as straight
  const lines = [
    { x1: cx - 80,  y1: cy - 20, x2: ops[0].x - opRadius, y2: ops[0].y, label: 'c_{t-1}', value: c_tminus1[id]},
    { x1: gates[0].x, y1: gates[0].y - gateHeight/2, x2: ops[0].x, y2: ops[0].y + opRadius, label: 'f_t', value: unitData.gates.forget},
    { x1: ops[0].x + opRadius, y1: ops[0].y, x2: ops[2].x - opRadius, y2: ops[2].y, label: 'c_{t-1}*f_t', value:  c_tminus1[id]*unitData.gates.forget},
    // --- Modified: Mark this connection as curved:
    { x1: gates[1].x, y1: gates[1].y - gateHeight/2, x2: ops[1].x - opRadius, y2: ops[1].y, label: 'i_t', value: unitData.gates.input, curve: true },
    { x1: gates[2].x, y1: gates[2].y - gateHeight/2, x2: ops[1].x, y2: ops[1].y + opRadius, label: 'c~_t', value: unitData.gates.cell },
    { x1: ops[1].x, y1: ops[1].y - opRadius, x2: ops[2].x, y2: ops[2].y + opRadius, label: 'i_t*c~_t', value: unitData.gates.input*unitData.gates.cell },
    { x1: ops[2].x + opRadius, y1: ops[2].y, x2: cx + 80, y2: cy - 20, label: 'c_t', value: c_t[id] },
    { x1: cx + 60, y1: cy - 20, x2: ops[3].x, y2: ops[3].y - opRadius, label: 'c_t', value: c_t[id] },
    { x1: gates[3].x + gateWidth/2, y1: gates[3].y, x2: ops[4].x - opRadius, y2: ops[4].y, label: 'o_t', value: unitData.gates.output },
    { x1: ops[3].x, y1: ops[3].y + opRadius, x2: ops[4].x, y2: ops[4].y - opRadius, label: 'tanh(c_t)', value: tanh(c_t[id]) },
    { x1: ops[4].x + opRadius, y1: ops[4].y, x2: cx + 80, y2: ops[4].y, label: 'h_t', value: h_t[id] },
    { x1: cx - 80, y1: gates[3].y, x2: gates[3].x - gateWidth/2, y2: gates[3].y, label: '[x_t, h_{t-1}]', value: roundArray(x_t.concat(h_tminus1), DECIMAL_POINTS) },
    { x1: gates[0].x, y1: gates[3].y, x2: gates[0].x, y2: gates[0].y + gateHeight/2, label: '[x_t, h_{t-1}]', value: roundArray(x_t.concat(h_tminus1), DECIMAL_POINTS) },
    { x1: gates[1].x, y1: gates[3].y, x2: gates[1].x, y2: gates[1].y + gateHeight/2, label: '[x_t, h_{t-1}]', value: roundArray(x_t.concat(h_tminus1), DECIMAL_POINTS) },
    { x1: gates[2].x, y1: gates[3].y, x2: gates[2].x, y2: gates[2].y + gateHeight/2, label: '[x_t, h_{t-1}]', value: roundArray(x_t.concat(h_tminus1), DECIMAL_POINTS) },
  ];


  const connEnter = internalGroup.selectAll('.internal-conn')
    .data(lines)
    .enter();

  connEnter.filter(d => d.curve)
    .append('path')
    .attr('class', 'internal-conn')
    .attr('d', d => {
      return generateCurve({ x: d.x1, y: d.y1 }, { x: d.x2, y: d.y2 });
    })
    .attr('stroke', '#333')
    .attr('stroke-width', 2)
    .attr('fill', 'none')
    .on('mouseover', (event, d) => {
      const [mx, my] = d3.pointer(event, container.node());
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      let html = d.label ? `<b>${d.label}</b><br>` : '';
      html += `Value: ${d.value}`;
      tooltip.style('visibility', 'visible')
        .html(html)
        .style('left', (mx + svgRect.left + 10) + 'px')
        .style('top', (my + svgRect.top + 10) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      tooltip
        .style('left', (mx + svgRect.left + 10) + 'px')
        .style('top', (my + svgRect.top + 10) + 'px');
    })
    .on('mouseout', () => { tooltip.style('visibility', 'hidden'); });

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
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      const randW = generateRandomWeight(2);
      let html = d.label ? `<b>${d.label}</b><br>` : '';
      html += `Value: ${d.value}`;
      tooltip.style('visibility', 'visible')
        .html(html)
        .style('left', (mx + svgRect.left + 10) + 'px')
        .style('top', (my + svgRect.top + 10) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      const svgRect = container.node().getBoundingClientRect(); // Get SVG position
      tooltip
        .style('left', (mx + svgRect.left + 10) + 'px')
        .style('top', (my + svgRect.top + 10) + 'px');
    })
    .on('mouseout', () => { tooltip.style('visibility', 'hidden'); });
}

/**
 * Zoom to a specific unit.
 */
export function zoomToUnit(svg, zoom, width, height, unitData) {
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

/**
 * Center (zoom out to fit) all units in view.
 */
export function centerAllUnits(svg, mainGroup, zoom, width, height, repositionLayerTitle) {
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
  mainGroup.selectAll('.internal-lstm-group').remove();

  svg.transition()
    .duration(750)
    .call(
      zoom.transform,
      d3.zoomIdentity.translate(tx, ty).scale(scale)
    )
    .on('end', () => {
      repositionLayerTitle(mainGroup);
    });
}

/**
 * Reposition "LSTM Layer 1" text above the bounding box of all units
 */
export function repositionLayerTitle(mainGroup) {
  const bounds = mainGroup.node().getBBox();
  const titleX = bounds.x + bounds.width / 2;
  // Animate x position
  mainGroup.select('.layer-title')
    .transition()
    .duration(500)
    .attr('x', titleX);
}
