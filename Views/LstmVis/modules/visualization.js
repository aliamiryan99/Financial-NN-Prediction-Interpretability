// visualization.js
import {
  generateRandomWeight,
  generateRandomVector
} from './randomUtils.js';

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
export function renderStatesPanel(panelSelector, statesHistory, container, tooltip) {
  const panel = d3.select(panelSelector);
  panel.html('');

  panel.append('h4').text('Hidden/Cell States History');

  const ul = panel.append('ul');

  statesHistory.forEach((stateObj, i) => {
    const li = ul.append('li');
    if (i === statesHistory.length - 1) {
      li.style('font-weight', 'bold').style('font-size', '14px');
    }

    // For showing vector tooltip
    function showVectorTooltip(event, arr, label) {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip
        .style('visibility', 'visible')
        .html(`<b>${label}:</b><br>${JSON.stringify(arr)}`)
        .style('left', (mx + 15) + 'px')
        .style('top', (my + 15) + 'px');
    }

    li.append('span').text(`Step ${stateObj.stepIndex}: `);

    // Hidden State link
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

    // Cell State link
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
  mainGroup, data, sequenceData, currentSampleIndex,
  container, tooltip,
  RECT_WIDTH, RECT_HEIGHT,
  d3TransitionDuration
) {
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
  pathEnter
    .filter(d => d.type === 'input' || d.type === 'hidden' || d.type === 'cell')
    .append('circle')
    .attr('class', 'start-node')
    .attr('r', 5);

  // Output labels
  pathEnter
    .filter(d => d.type === 'hiddenOutput' || d.type === 'cellOutput')
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
    .transition().duration(d3TransitionDuration)
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

/**
 * Renders internal LSTM gate details for a single unit.
 * In the original code, each time you click a unit, we create these shapes.
 */
export function renderGateDetails(mainGroup, unitData, container, tooltip) {
  // Remove old shapes
  mainGroup.selectAll('.internal-lstm-group').remove();

  // Create a group for internal diagram
  const internalGroup = mainGroup.append('g')
    .attr('class', 'internal-lstm-group');

  const cx = unitData.x;
  const cy = unitData.y;

  // (1) Gates
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
      // Pull a random weight from randomUtils
      const randWeight = generateRandomWeight(3);
      tooltip.style('visibility', 'visible')
        .html(`<b>${d.name}</b><br>Weight: ${randWeight}`)
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
    .on('mouseover', (event, d) => {
      const [mx, my] = d3.pointer(event, container.node());
      // Example: random vector of length 5
      const randomVec = generateRandomVector(5, 2);
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
    .on('mouseout', () => {
      tooltip.style('visibility', 'hidden');
    });

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
      const randW = generateRandomWeight(2);
      let html = d.label ? `<b>${d.label}</b><br>` : '';
      html += `Connection Weight: ${randW}`;
      tooltip.style('visibility', 'visible')
        .html(html)
        .style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip
        .style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
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
      const randW = generateRandomWeight(2);
      let html = d.label ? `<b>${d.label}</b><br>` : '';
      html += `Connection Weight: ${randW}`;
      tooltip.style('visibility', 'visible')
        .html(html)
        .style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
    })
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, container.node());
      tooltip
        .style('left', (mx + 10) + 'px')
        .style('top', (my + 10) + 'px');
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
