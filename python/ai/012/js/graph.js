// 画图函数

// 坐标系设置
const DEFAULT_RANGE = 3;
const GRID_INTERVAL = 0.5;
const SAMPLE_POINTS = 200;

// 颜色定义
const COLORS = {
  grid: '#f0f0f0',
  axis: '#374151',
  curve: '#3b82f6',
  outerCurve: '#10b981',
  point: '#f59e0b',
  tangent: '#ef4444',
  connection: '#8b5cf6'
};

function createGraph(canvasId, width, height) {
  const canvas = document.getElementById(canvasId);
  canvas.width = width;
  canvas.height = height;
  return {
    canvas,
    ctx: canvas.getContext('2d'),
    width,
    height,
    centerX: width / 2,
    centerY: height / 2,
    scaleX: width / (DEFAULT_RANGE * 2),
    scaleY: height / (DEFAULT_RANGE * 2)
  };
}

function clearGraph(graph) {
  graph.ctx.fillStyle = '#ffffff';
  graph.ctx.fillRect(0, 0, graph.width, graph.height);
}

function drawGrid(graph, xRange, yRange) {
  const ctx = graph.ctx;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;

  // 垂直网格线
  for (let x = Math.floor(xRange[0] / GRID_INTERVAL) * GRID_INTERVAL; x <= xRange[1]; x += GRID_INTERVAL) {
    const screenX = graph.centerX + x * graph.scaleX;
    ctx.beginPath();
    ctx.moveTo(screenX, 0);
    ctx.lineTo(screenX, graph.height);
    ctx.stroke();
  }

  // 水平网格线
  for (let y = Math.floor(yRange[0] / GRID_INTERVAL) * GRID_INTERVAL; y <= yRange[1]; y += GRID_INTERVAL) {
    const screenY = graph.centerY - y * graph.scaleY;
    ctx.beginPath();
    ctx.moveTo(0, screenY);
    ctx.lineTo(graph.width, screenY);
    ctx.stroke();
  }
}

function drawAxes(graph, xRange, yRange) {
  const ctx = graph.ctx;
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.5;

  // X轴
  if (yRange[0] <= 0 && yRange[1] >= 0) {
    const y0 = graph.centerY;
    ctx.beginPath();
    ctx.moveTo(0, y0);
    ctx.lineTo(graph.width, y0);
    ctx.stroke();

    // X轴标签
    ctx.fillStyle = '#374151';
    ctx.font = '12px PingFang SC';
    ctx.textAlign = 'left';
    for (let x = Math.ceil(xRange[0]); x <= xRange[1]; x++) {
      if (x !== 0) {
        const screenX = graph.centerX + x * graph.scaleX;
        ctx.fillText(x.toString(), screenX + 3, y0 + 15);
      }
    }
  }

  // Y轴
  if (xRange[0] <= 0 && xRange[1] >= 0) {
    const x0 = graph.centerX;
    ctx.beginPath();
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, graph.height);
    ctx.stroke();

    // Y轴标签
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'right';
    for (let y = Math.ceil(yRange[0]); y <= yRange[1]; y++) {
      if (y !== 0) {
        const screenY = graph.centerY - y * graph.scaleY;
        ctx.fillText(y.toString(), x0 - 5, screenY + 4);
      }
    }
  }

  // 原点
  ctx.fillStyle = '#374151';
  ctx.textAlign = 'right';
  ctx.fillText('0', graph.centerX - 5, graph.centerY + 15);
}

function toScreen(graph, x, y) {
  return {
    x: graph.centerX + x * graph.scaleX,
    y: graph.centerY - y * graph.scaleY
  };
}

function drawCurve(graph, fn, xRange, yRange, color, lineWidth = 2) {
  const ctx = graph.ctx;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();

  const step = (xRange[1] - xRange[0]) / SAMPLE_POINTS;
  let started = false;

  for (let i = 0; i <= SAMPLE_POINTS; i++) {
    const x = xRange[0] + i * step;
    const y = fn(x);

    if (isFinite(y) && y >= yRange[0] - 1 && y <= yRange[1] + 1) {
      const screen = toScreen(graph, x, y);
      if (!started) {
        ctx.moveTo(screen.x, screen.y);
        started = true;
      } else {
        ctx.lineTo(screen.x, screen.y);
      }
    } else {
      started = false;
    }
  }

  ctx.stroke();
}

function drawTangentLine(graph, x, y, dy, color) {
  const ctx = graph.ctx;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;

  // 切线长度（以数学单位计）
  const tangentLength = 1.2;

  // 切线上的两个点
  const x1 = x - tangentLength;
  const y1 = y - dy * tangentLength;
  const x2 = x + tangentLength;
  const y2 = y + dy * tangentLength;

  const screen1 = toScreen(graph, x1, y1);
  const screen2 = toScreen(graph, x2, y2);

  ctx.beginPath();
  ctx.moveTo(screen1.x, screen1.y);
  ctx.lineTo(screen2.x, screen2.y);
  ctx.stroke();

  // 画箭头
  const angle = Math.atan2(screen2.y - screen1.y, screen2.x - screen1.x);
  const arrowSize = 8;

  ctx.beginPath();
  ctx.moveTo(screen2.x, screen2.y);
  ctx.lineTo(
    screen2.x - arrowSize * Math.cos(angle - Math.PI / 6),
    screen2.y - arrowSize * Math.sin(angle - Math.PI / 6)
  );
  ctx.moveTo(screen2.x, screen2.y);
  ctx.lineTo(
    screen2.x - arrowSize * Math.cos(angle + Math.PI / 6),
    screen2.y - arrowSize * Math.sin(angle + Math.PI / 6)
  );
  ctx.stroke();
}

function drawPoint(graph, x, y, radius = 6) {
  const ctx = graph.ctx;
  const screen = toScreen(graph, x, y);

  ctx.beginPath();
  ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.point;
  ctx.fill();
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawLabel(graph, text, x, y, color, offsetX = 10, offsetY = -10) {
  const ctx = graph.ctx;
  const screen = toScreen(graph, x, y);

  ctx.fillStyle = color;
  ctx.font = 'bold 13px PingFang SC';
  ctx.textAlign = 'left';
  ctx.fillText(text, screen.x + offsetX, screen.y + offsetY);
}
