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

// 自适应范围画布：xRange/yRange 自动决定缩放和原点位置
function createDerivativeGraph(canvasId, xRange, yRange, width, height) {
  const canvas = document.getElementById(canvasId);
  canvas.width = width;
  canvas.height = height;
  const xSpan = xRange[1] - xRange[0];
  const ySpan = yRange[1] - yRange[0];
  return {
    canvas,
    ctx: canvas.getContext('2d'),
    width,
    height,
    xRange,
    yRange,
    // centerX/centerY 是数学原点 (x=0) 在屏幕上的位置
    centerX: (-xRange[0] / xSpan) * width,
    centerY: (yRange[1] / ySpan) * height,
    scaleX: width / xSpan,
    scaleY: height / ySpan
  };
}

// 计算"漂亮"的刻度间隔（1, 2, 5 × 10^n 序列）
function niceTickInterval(span) {
  if (span <= 0 || !isFinite(span)) return 1;
  const target = span / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(target)));
  const norm = target / mag;
  if (norm < 1.5) return mag;
  if (norm < 3) return 2 * mag;
  if (norm < 7) return 5 * mag;
  return 10 * mag;
}

function formatTick(v) {
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  if (abs >= 0.1) return v.toFixed(3);
  return v.toFixed(4);
}

// 在自适应范围画布上画坐标轴 + 网格 + 刻度
function drawDerivativeAxes(graph, xRange, yRange) {
  const ctx = graph.ctx;

  // 背景网格
  const xTick = niceTickInterval(xRange[1] - xRange[0]);
  const yTick = niceTickInterval(yRange[1] - yRange[0]);

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let x = Math.ceil(xRange[0] / xTick) * xTick; x <= xRange[1]; x += xTick) {
    if (Math.abs(x) < xTick * 0.001) continue;
    const screenX = graph.centerX + x * graph.scaleX;
    ctx.beginPath();
    ctx.moveTo(screenX, 0);
    ctx.lineTo(screenX, graph.height);
    ctx.stroke();
  }
  for (let y = Math.ceil(yRange[0] / yTick) * yTick; y <= yRange[1]; y += yTick) {
    if (Math.abs(y) < yTick * 0.001) continue;
    const screenY = graph.centerY - y * graph.scaleY;
    ctx.beginPath();
    ctx.moveTo(0, screenY);
    ctx.lineTo(graph.width, screenY);
    ctx.stroke();
  }

  // 坐标轴（仅当原点落在画布内时画）
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.5;
  if (yRange[0] <= 0 && yRange[1] >= 0) {
    const y0 = graph.centerY;
    ctx.beginPath();
    ctx.moveTo(0, y0);
    ctx.lineTo(graph.width, y0);
    ctx.stroke();
  }
  if (xRange[0] <= 0 && xRange[1] >= 0) {
    const x0 = graph.centerX;
    ctx.beginPath();
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, graph.height);
    ctx.stroke();
  }

  // X 轴刻度标签
  ctx.fillStyle = '#6b7280';
  ctx.font = '11px PingFang SC';
  ctx.textAlign = 'center';
  for (let x = Math.ceil(xRange[0] / xTick) * xTick; x <= xRange[1]; x += xTick) {
    if (Math.abs(x) < xTick * 0.001) continue;
    const screenX = graph.centerX + x * graph.scaleX;
    ctx.fillText(formatTick(x), screenX, graph.height - 4);
  }

  // Y 轴刻度标签
  ctx.textAlign = 'right';
  for (let y = Math.ceil(yRange[0] / yTick) * yTick; y <= yRange[1]; y += yTick) {
    if (Math.abs(y) < yTick * 0.001) continue;
    const screenY = graph.centerY - y * graph.scaleY;
    ctx.fillText(formatTick(y), graph.width - 4, screenY + 4);
  }

  // 原点标签
  ctx.textAlign = 'right';
  ctx.fillText('0', graph.centerX - 4, graph.height - 4);
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

// 绘制导数曲线：自动识别纵向跳跃并断开（适用于 |u| 等间断点）
function drawDerivativeCurve(graph, fn, xRange, yRange, color, lineWidth = 2) {
  const ctx = graph.ctx;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();

  const step = (xRange[1] - xRange[0]) / SAMPLE_POINTS;
  const ySpan = yRange[1] - yRange[0];
  const jumpThreshold = ySpan * 0.5;
  let started = false;
  let prevY = null;

  for (let i = 0; i <= SAMPLE_POINTS; i++) {
    const x = xRange[0] + i * step;
    const y = fn(x);

    if (!isFinite(y) || y < yRange[0] - 1 || y > yRange[1] + 1) {
      started = false;
      prevY = null;
      continue;
    }

    if (prevY !== null && Math.abs(y - prevY) > jumpThreshold) {
      started = false;
    }

    const screen = toScreen(graph, x, y);
    if (!started) {
      ctx.moveTo(screen.x, screen.y);
      started = true;
    } else {
      ctx.lineTo(screen.x, screen.y);
    }
    prevY = y;
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
