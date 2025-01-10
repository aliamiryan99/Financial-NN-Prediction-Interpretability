/*****************************************************************************
 * 1) Setup & Mock Data
 *****************************************************************************/

const NUM_UNITS = 50;

// Generate random arrays
function randomArray(len) {
  return Array.from({ length: len }, () => Math.random().toFixed(3));
}

// Mock data
let unitsData = Array.from({ length: NUM_UNITS }, (_, i) => {
  return {
    id: i,
    importance: Math.random(),
    inputVector: randomArray(5),
    hiddenVector: randomArray(50),
    cellVector: randomArray(50),
    hiddenOutputValue: Math.random().toFixed(3),
    cellOutputValue: Math.random().toFixed(3),
    gates: {
      forget: Math.random(),
      input: Math.random(),
      candidate: Math.random(),
      output: Math.random()
    }
  };
});

/*****************************************************************************
 * 2) Create SVG + Zoom + Markers
 *****************************************************************************/

const container = d3.select('#lstm-visualization');
const rect = container.node().getBoundingClientRect();

const width = rect.width;
const height = rect.height;

const svg = container
  .append('svg')
  .attr('width', width)
  .attr('height', height);

const mainGroup = svg.append('g').attr('class', 'main-group');

// Define a global zoom
const zoom = d3
  .zoom()
  .scaleExtent([0.5, 8])
  .on('zoom', (event) => {
    mainGroup.attr('transform', event.transform);
  });

// Enable zoom on the svg
svg.call(zoom);

// If user clicks on the SVG background (not on a rectangle), reset zoom
svg.on('click', (event) => {
  // If the target is the <svg> itself (and not a rect)
  if (event.target === svg.node()) {
    centerAllUnits();
  }
});

// Tooltip
const tooltip = container
  .append('div')
  .attr('class', 'tooltip');

// Markers (arrow & smaller square)
const defs = svg.append('defs');

// Arrow marker
defs
  .append('marker')
  .attr('id', 'arrow')
  .attr('markerWidth', 10)
  .attr('markerHeight', 10)
  .attr('refX', 6)
  .attr('refY', 3)
  .attr('orient', 'auto')
  .attr('markerUnits', 'strokeWidth')
  .append('path')
  .attr('d', 'M0,0 L0,6 L6,3 z')
  .attr('fill', '#000');

// Smaller square marker
defs
  .append('marker')
  .attr('id', 'square')
  .attr('markerWidth', 5)
  .attr('markerHeight', 5)
  .attr('refX', 2.5)
  .attr('refY', 2.5)
  .attr('markerUnits', 'strokeWidth')
  .append('rect')
  .attr('x', 0)
  .attr('y', 0)
  .attr('width', 5)
  .attr('height', 5)
  .attr('fill', '#000');

/*****************************************************************************
 * 3) Layout: Grid (2 columns)
 *****************************************************************************/

const NUM_COLUMNS = 2;
const UNIT_SPACING_X = 360; 
const UNIT_SPACING_Y = 200; 
const RECT_WIDTH = 160;
const RECT_HEIGHT = 80;

/**
 * Compute grid (col, row) => (x, y) center
 */
function getGridPosition(i) {
  const col = i % NUM_COLUMNS;
  const row = Math.floor(i / NUM_COLUMNS);
  // We’ll place them with some offsets; final centering is done after bounding box is known.
  const x = col * UNIT_SPACING_X + 200;
  const y = row * UNIT_SPACING_Y + 100;
  return { x, y };
}

/*****************************************************************************
 * 4) Rendering Units
 *****************************************************************************/

function renderUnits(data) {
  const rects = mainGroup.selectAll('.lstm-unit').data(data, (d) => d.id);

  // EXIT
  rects
    .exit()
    .transition()
    .duration(750)
    .attr('opacity', 0)
    .remove();

  // UPDATE
  rects
    .transition()
    .duration(750)
    .attr('x', (d) => d.x - RECT_WIDTH / 2)
    .attr('y', (d) => d.y - RECT_HEIGHT / 2);

  // ENTER
  const rectsEnter = rects
    .enter()
    .append('rect')
    .attr('class', 'lstm-unit')
    .attr('rx', 10)
    .attr('ry', 10)
    .attr('width', RECT_WIDTH)
    .attr('height', RECT_HEIGHT)
    .attr('opacity', 0)
    .attr('x', (d) => d.x - RECT_WIDTH / 2)
    .attr('y', (d) => d.y - RECT_HEIGHT / 2)
    .on('click', handleUnitClick);

  rectsEnter
    .transition()
    .duration(750)
    .attr('opacity', 1);

  rectsEnter.merge(rects).each(function (d) {
    // d.x, d.y are set
  });
}

/*****************************************************************************
 * 5) Rendering Connections
 *    - Align lines so they start at the left edge of the rectangle (with small offset)
 *      and end exactly at the rectangle edges.
 *****************************************************************************/

function renderConnections(data) {
  const connections = [];

  data.forEach((unit) => {
    // We'll define a single "join" point near bottom-left
    const joinPoint = {
      x: unit.x - RECT_WIDTH / 2 + 10, 
      y: unit.y + RECT_HEIGHT / 2 - 10
    };

    // --- Cell line (straight) from left side
    connections.push({
      type: 'cell',
      vector: unit.cellVector,
      source: {
        x: unit.x - RECT_WIDTH / 2 - 60,
        y: unit.y - 0 // aligned with center Y
      },
      target: {
        x: unit.x - RECT_WIDTH / 2,
        y: unit.y - RECT_HEIGHT / 2 + 10
      },
      markerStart: 'url(#square)',
      markerEnd: 'url(#arrow)',
      isCurve: false
    });

    // --- Hidden line (straight) from bottom-left
    connections.push({
      type: 'hidden',
      vector: unit.hiddenVector,
      source: {
        x: unit.x - RECT_WIDTH / 2 - 60,
        y: unit.y + 20 // slightly below center
      },
      target: {
        x: unit.x - RECT_WIDTH / 2,
        y: unit.y + RECT_HEIGHT / 2 - 10
      },
      markerStart: 'url(#square)',
      markerEnd: null,
      isCurve: false
    });

    // --- Input line (curved) from further below -> join
    connections.push({
      type: 'input',
      vector: unit.inputVector,
      source: {
        x: unit.x - RECT_WIDTH / 2 - 60,
        y: unit.y + 70
      },
      target: joinPoint,
      markerStart: 'url(#square)',
      markerEnd: null,
      isCurve: true
    });

    // --- Join -> rectangle
    connections.push({
      type: 'mergeLine',
      vector: null,
      source: joinPoint,
      target: {
        x: unit.x - RECT_WIDTH / 2,
        y: unit.y + RECT_HEIGHT / 2 - 10
      },
      markerStart: null,
      markerEnd: 'url(#arrow)',
      isCurve: false
    });

    // --- Hidden Output (straight)
    connections.push({
      type: 'hiddenOutput',
      value: unit.hiddenOutputValue,
      source: {
        x: unit.x + RECT_WIDTH / 2,
        y: unit.y + 15
      },
      target: {
        x: unit.x + RECT_WIDTH / 2 + 70,
        y: unit.y + 15
      },
      markerStart: null,
      markerEnd: 'url(#arrow)',
      isCurve: false
    });

    // --- Cell Output (straight)
    connections.push({
      type: 'cellOutput',
      value: unit.cellOutputValue,
      source: {
        x: unit.x + RECT_WIDTH / 2,
        y: unit.y - 15
      },
      target: {
        x: unit.x + RECT_WIDTH / 2 + 70,
        y: unit.y - 15
      },
      markerStart: null,
      markerEnd: 'url(#arrow)',
      isCurve: false
    });
  });

  const paths = mainGroup.selectAll('.connection').data(connections, (d, i) => i);

  // EXIT
  paths.exit().remove();

  // ENTER + UPDATE
  paths
    .enter()
    .append('path')
    .attr('class', 'connection')
    .merge(paths)
    .transition()
    .duration(750)
    .attr('d', (d) => d.isCurve
      ? generateCurve(d.source, d.target)
      : generateLine(d.source, d.target)
    )
    .attr('marker-start', (d) => d.markerStart)
    .attr('marker-end', (d) => d.markerEnd);

  // Tooltips
  mainGroup.selectAll('.connection')
    .on('mouseover', (event, d) => {
      let htmlContent = '';
      if (d.type === 'input' || d.type === 'hidden' || d.type === 'cell') {
        htmlContent = `<b>${d.type} vector:</b><br>${JSON.stringify(d.vector)}`;
      } else if (d.type === 'hiddenOutput' || d.type === 'cellOutput') {
        htmlContent = `<b>${d.type}:</b> ${d.value}`;
      }
      if (htmlContent) {
        tooltip
          .style('visibility', 'visible')
          .html(htmlContent)
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY + 10 + 'px');
      }
    })
    .on('mousemove', (event) => {
      tooltip
        .style('left', event.pageX + 10 + 'px')
        .style('top', event.pageY + 10 + 'px');
    })
    .on('mouseout', () => {
      tooltip.style('visibility', 'hidden');
    });
}

/** Straight line generator */
function generateLine(source, target) {
  return `M ${source.x},${source.y} L ${target.x},${target.y}`;
}

/** Cubic bezier curve generator */
function generateCurve(source, target) {
  const midX = (source.x + target.x) / 2;
  return `
    M ${source.x},${source.y}
    C ${midX},${source.y},
      ${midX},${target.y},
      ${target.x},${target.y}
  `;
}

/*****************************************************************************
 * 6) Zoom & Gate Details
 *****************************************************************************/

function handleUnitClick(event, d) {
  // Prevent background from also triggering
  event.stopPropagation();
  zoomToUnit(d);
  d3.select(this).classed('zoomed-in', true);
  renderGateDetails(d);
}

function zoomToUnit(unitData) {
  const scale = 3;
  svg
    .transition()
    .duration(750)
    .call(
      zoom.transform,
      d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(scale)
        .translate(-unitData.x, -unitData.y)
    );
}

function renderGateDetails(unitData) {
  // Remove old gate circles
  mainGroup.selectAll('.gate-circle').remove();

  const circlesData = [0, 1, 2, 3];
  const circleRadius = 10;

  mainGroup
    .selectAll('.gate-circle')
    .data(circlesData)
    .enter()
    .append('circle')
    .attr('class', 'gate-circle')
    .attr('cx', (g, i) => {
      const col = i % 2;
      return unitData.x - 20 + col * 40;
    })
    .attr('cy', (g, i) => {
      const row = Math.floor(i / 2);
      return unitData.y - 10 + row * 20;
    })
    .attr('r', 0)
    .transition()
    .duration(750)
    .attr('r', circleRadius)
    .attr('opacity', 1);
}

/*****************************************************************************
 * 7) Centering & Reset Zoom
 *    - We'll compute bounding box, then use scale so all units fit inside,
 *      and center them horizontally & vertically.
 *****************************************************************************/

function resetZoom() {
  centerAllUnits();
}

/** Center the entire mainGroup so all units are visible using scale + translate. */
function centerAllUnits() {
  // bounding box of mainGroup
  const bounds = mainGroup.node().getBBox();
  const fullWidth = width;
  const fullHeight = height;

  if (bounds.width === 0 || bounds.height === 0) return;

  // We'll compute a scale to fit everything within fullWidth x fullHeight
  // with a little margin factor (say 0.9)
  const marginFactor = 0.9;
  let scale = Math.min(
    (fullWidth * marginFactor) / bounds.width,
    (fullHeight * marginFactor) / bounds.height
  );

  // We don't want to exceed 1 if we have fewer units
  if (scale > 1) {
    scale = 1;
  }

  // The translate needed to center the group
  const tx = (fullWidth - bounds.width * scale) / 2 - bounds.x * scale;
  const ty = (fullHeight - bounds.height * scale) / 2 - bounds.y * scale;

  svg
    .transition()
    .duration(750)
    .call(
      zoom.transform,
      d3.zoomIdentity.translate(tx, ty).scale(scale)
    );
}

/*****************************************************************************
 * 8) Update & Initialization
 *****************************************************************************/

function updateVisualization(numUnits) {
  // Sort by importance
  const topUnits = unitsData
    .slice()
    .sort((a, b) => d3.descending(a.importance, b.importance))
    .slice(0, numUnits);

  // Compute grid positions
  topUnits.forEach((unit, i) => {
    const { x, y } = getGridPosition(i);
    unit.x = x;
    unit.y = y;
  });

  // Clear "zoomed-in" class
  mainGroup.selectAll('.lstm-unit').classed('zoomed-in', false);

  renderUnits(topUnits);
  renderConnections(topUnits);

  // After rendering, fit to screen
  setTimeout(() => {
    centerAllUnits();
  }, 800);
}

// Initial load
updateVisualization(10);

// Spinner
const unitsCountInput = document.getElementById('units-count');
const unitsLabel = document.getElementById('units-label');

unitsCountInput.addEventListener('input', function () {
  const n = +this.value;
  unitsLabel.textContent = n;
  updateVisualization(n);
});
unitsLabel.textContent = unitsCountInput.value;

// Reset Zoom button
document.getElementById('reset-zoom-btn').addEventListener('click', resetZoom);
