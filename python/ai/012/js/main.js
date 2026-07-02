// 主逻辑

// 内层函数定义
const innerFunctions = {
  'x²': {
    fn: x => x * x,
    derivative: x => 2 * x,
    xRange: [-2, 2],
    label: 'u = x²',
    latex: 'u = x^{2}'
  },
  'sin(x)': {
    fn: x => Math.sin(x),
    derivative: x => Math.cos(x),
    xRange: [-2 * Math.PI, 2 * Math.PI],
    label: 'u = sin(x)',
    latex: 'u = \\sin x'
  },
  'eˣ': {
    fn: x => Math.exp(x),
    derivative: x => Math.exp(x),
    xRange: [-2, 2],
    label: 'u = eˣ',
    latex: 'u = e^{x}'
  },
  'x³': {
    fn: x => x * x * x,
    derivative: x => 3 * x * x,
    xRange: [-1.5, 1.5],
    label: 'u = x³',
    latex: 'u = x^{3}'
  }
};

// 外层函数定义
const outerFunctions = {
  'u²': {
    fn: u => u * u,
    derivative: u => 2 * u,
    validate: () => true,
    label: 'y = u²',
    latex: 'y = u^{2}'
  },
  'sin(u)': {
    fn: u => Math.sin(u),
    derivative: u => Math.cos(u),
    validate: () => true,
    label: 'y = sin(u)',
    latex: 'y = \\sin u'
  },
  'eᵘ': {
    fn: u => Math.exp(u),
    derivative: u => Math.exp(u),
    validate: () => true,
    label: 'y = eᵘ',
    latex: 'y = e^{u}'
  },
  'ln(u)': {
    fn: u => Math.log(u),
    derivative: u => 1 / u,
    validate: (u) => u > 0,
    label: 'y = ln(u)',
    latex: 'y = \\ln u'
  },
  '|u|': {
    fn: u => Math.abs(u),
    derivative: u => u === 0 ? NaN : (u > 0 ? 1 : -1),
    validate: () => true,
    label: 'y = |u|',
    latex: 'y = |u|'
  }
};

// 全局状态
let currentInnerKey = 'x²';
let currentOuterKey = 'u²';
let currentX = 1;
let animation = null;
let innerGraph = null;
let outerGraph = null;
let dudxGraph = null;
let dyduGraph = null;
let dydxGraph = null;

// DOM 元素
const innerSelect = document.getElementById('innerFunction');
const outerSelect = document.getElementById('outerFunction');
const xSlider = document.getElementById('xSlider');
const xValueDisplay = document.getElementById('xValue');
const playBtn = document.getElementById('playBtn');
const speedSelect = document.getElementById('speed');

// 初始化
function init() {
  innerGraph = createGraph('innerCanvas', 320, 280);
  outerGraph = createGraph('outerCanvas', 320, 280);
  animation = new AnimationController();

  // 渲染标题 LaTeX
  const titleEl = document.getElementById('pageTitle');
  titleEl.innerHTML = '链式法则可视化 ' + katex.renderToString('\\frac{dy}{dx} = \\frac{dy}{du} \\times \\frac{du}{dx}', { throwOnError: false });

  // 渲染三幅导数图标题 LaTeX
  const renderDerivativeLabel = (id, frac) => {
    document.getElementById(id).innerHTML = katex.renderToString(
      `${frac} \\text{ 随 } x \\text{ 变化}`,
      { throwOnError: false }
    );
  };
  renderDerivativeLabel('labelDudxGraph', '\\frac{du}{dx}');
  renderDerivativeLabel('labelDyduGraph', '\\frac{dy}{du}');
  renderDerivativeLabel('labelDydxGraph', '\\frac{dy}{dx}');

  // 事件绑定
  innerSelect.addEventListener('change', onInnerFunctionChange);
  outerSelect.addEventListener('change', onOuterFunctionChange);
  xSlider.addEventListener('input', onSliderChange);
  playBtn.addEventListener('click', onPlayClick);
  speedSelect.addEventListener('change', onSpeedChange);

  // 初始化
  updateRange();

  // 动画更新回调
  animation.onUpdate = (x) => {
    currentX = x;
    xValueDisplay.textContent = currentX.toFixed(2);
    xSlider.value = currentX;
    update();
  };

  update();
}

// 更新 x 范围
function updateRange() {
  const inner = innerFunctions[currentInnerKey];
  animation.setRange(inner.xRange[0], inner.xRange[1]);
  xSlider.min = inner.xRange[0];
  xSlider.max = inner.xRange[1];
  xSlider.step = 0.01;

  // 如果当前 x 超出范围，调整它
  if (currentX < inner.xRange[0] || currentX > inner.xRange[1]) {
    currentX = Math.max(inner.xRange[0], Math.min(inner.xRange[1], currentX));
  }

  xSlider.value = currentX;
  xValueDisplay.textContent = currentX.toFixed(2);
}

// 内层函数变化
function onInnerFunctionChange(e) {
  currentInnerKey = e.target.value;
  updateRange();
  update();
}

// 外层函数变化
function onOuterFunctionChange(e) {
  currentOuterKey = e.target.value;
  update();
}

// 滑块变化
function onSliderChange(e) {
  animation.pause();
  playBtn.textContent = '▶ 播放';
  playBtn.classList.remove('playing');

  currentX = parseFloat(e.target.value);
  xValueDisplay.textContent = currentX.toFixed(2);
  xSlider.value = currentX;
  animation.setProgress(currentX);
  update();
}

// 播放按钮
function onPlayClick() {
  animation.toggle();

  if (animation.isPlaying) {
    playBtn.textContent = '⏸ 暂停';
    playBtn.classList.add('playing');
  } else {
    playBtn.textContent = '▶ 播放';
    playBtn.classList.remove('playing');
  }
}

// 速度变化
function onSpeedChange(e) {
  const speedMap = { '0.5x': 0.5, '1x': 1, '2x': 2 };
  animation.setSpeed(speedMap[e.target.value]);
}

// 计算所有值
function computeValues() {
  const inner = innerFunctions[currentInnerKey];
  const outer = outerFunctions[currentOuterKey];

  const u = inner.fn(currentX);
  const dudx = inner.derivative(currentX);

  // 计算 dy/du（使用 u 作为输入）
  const dydu = outer.derivative(u);

  // 计算复合导数 dy/dx = dy/du * du/dx
  const dydx = dydu * dudx;

  return { u, dudx, dydu, dydx };
}

// 主更新函数
function update() {
  const inner = innerFunctions[currentInnerKey];
  const outer = outerFunctions[currentOuterKey];
  const values = computeValues();

  // 计算自适应范围
  const xRange = inner.xRange;
  const u = values.u;

  // 计算 u 的范围
  let uMin = Infinity, uMax = -Infinity;
  for (let x = xRange[0]; x <= xRange[1]; x += (xRange[1] - xRange[0]) / 100) {
    const uVal = inner.fn(x);
    if (isFinite(uVal)) {
      uMin = Math.min(uMin, uVal);
      uMax = Math.max(uMax, uVal);
    }
  }

  // y 的范围基于 u 和 outer 函数
  let yMin = Infinity, yMax = -Infinity;
  for (let uVal = uMin; uVal <= uMax; uVal += (uMax - uMin) / 100) {
    if (outer.validate(uVal)) {
      const yVal = outer.fn(uVal);
      if (isFinite(yVal)) {
        yMin = Math.min(yMin, yVal);
        yMax = Math.max(yMax, yVal);
      }
    }
  }

  // 添加边距
  const uMargin = (uMax - uMin) * 0.1 || 1;
  const yMargin = (yMax - yMin) * 0.1 || 1;

  // 绘制内层函数图
  clearGraph(innerGraph);
  drawGrid(innerGraph, xRange, [uMin - uMargin, uMax + uMargin]);
  drawAxes(innerGraph, xRange, [uMin - uMargin, uMax + uMargin]);
  drawCurve(innerGraph, inner.fn, xRange, [uMin - uMargin, uMax + uMargin], '#3b82f6');

  // 绘制切线
  if (isFinite(values.dudx)) {
    drawTangentLine(innerGraph, currentX, values.u, values.dudx, '#ef4444');
  }

  // 绘制当前点
  drawPoint(innerGraph, currentX, values.u);
  drawLabel(innerGraph, `(${currentX.toFixed(2)}, ${values.u.toFixed(2)})`, currentX, values.u, '#f59e0b', 8, -12);

  // 绘制外层函数图
  clearGraph(outerGraph);
  drawGrid(outerGraph, [uMin - uMargin, uMax + uMargin], [yMin - yMargin, yMax + yMargin]);
  drawAxes(outerGraph, [uMin - uMargin, uMax + uMargin], [yMin - yMargin, yMax + yMargin]);

  // 对于 ln(u) 需要限制 u > 0，其他函数不需要
  const uRangeMin = currentOuterKey === 'ln(u)' ? Math.max(0.01, uMin - uMargin) : uMin - uMargin;
  drawCurve(outerGraph, outer.fn, [uRangeMin, uMax + uMargin], [yMin - yMargin, yMax + yMargin], '#10b981');

  // 绘制切线（在外层图上）
  if (isFinite(values.dydu)) {
    drawTangentLine(outerGraph, values.u, outer.fn(values.u), values.dydu, '#ef4444');
  }

  // 绘制当前点
  drawPoint(outerGraph, values.u, outer.fn(values.u));
  drawLabel(outerGraph, `(${values.u.toFixed(2)}, ${outer.fn(values.u).toFixed(2)})`, values.u, outer.fn(values.u), '#f59e0b', 8, -12);

  // 更新公式板
  updateFormulaBoard(values);

  // 更新数值显示
  updateValueDisplay(values);

  // 更新导数随 x 变化曲线
  updateDerivativeGraphs(values);
}

// 计算某个导数函数在 xRange 上的 y 范围（含 10% 边距）
function computeDerivativeRange(fn, xRange) {
  let yMin = Infinity, yMax = -Infinity;
  const step = (xRange[1] - xRange[0]) / 200;
  for (let i = 0; i <= 200; i++) {
    const x = xRange[0] + i * step;
    const y = fn(x);
    if (isFinite(y)) {
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
  }
  if (!isFinite(yMin)) { yMin = -1; yMax = 1; }
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const margin = (yMax - yMin) * 0.15 || 1;
  return [yMin - margin, yMax + margin];
}

// 更新三个导数曲线图
function updateDerivativeGraphs(values) {
  const inner = innerFunctions[currentInnerKey];
  const outer = outerFunctions[currentOuterKey];
  const xRange = inner.xRange;

  const dudxFn = x => inner.derivative(x);
  const dyduFn = x => outer.derivative(inner.fn(x));
  const dydxFn = x => inner.derivative(x) * outer.derivative(inner.fn(x));

  const dudxRange = computeDerivativeRange(dudxFn, xRange);
  const dyduRange = computeDerivativeRange(dyduFn, xRange);
  const dydxRange = computeDerivativeRange(dydxFn, xRange);

  // 重新创建画布以应用新的 scaleX/scaleY
  dudxGraph = createDerivativeGraph('dudxCanvas', xRange, dudxRange, 320, 180);
  dyduGraph = createDerivativeGraph('dyduCanvas', xRange, dyduRange, 320, 180);
  dydxGraph = createDerivativeGraph('dydxCanvas', xRange, dydxRange, 320, 180);

  // du/dx 图
  clearGraph(dudxGraph);
  drawDerivativeAxes(dudxGraph, xRange, dudxRange);
  drawDerivativeCurve(dudxGraph, dudxFn, xRange, dudxRange, '#3b82f6');
  const dudxVal = dudxFn(currentX);
  if (isFinite(dudxVal) && dudxVal >= dudxRange[0] && dudxVal <= dudxRange[1]) {
    drawPoint(dudxGraph, currentX, dudxVal);
  }
  document.getElementById('valDudxGraph').innerHTML =
    katex.renderToString(`\\frac{du}{dx} = ${dudxVal.toFixed(4)}`, { throwOnError: false, displayMode: false, output: 'html' });

  // dy/du 图
  clearGraph(dyduGraph);
  drawDerivativeAxes(dyduGraph, xRange, dyduRange);
  drawDerivativeCurve(dyduGraph, dyduFn, xRange, dyduRange, '#10b981');
  const dyduVal = dyduFn(currentX);
  if (isFinite(dyduVal) && dyduVal >= dyduRange[0] && dyduVal <= dyduRange[1]) {
    drawPoint(dyduGraph, currentX, dyduVal);
  }
  document.getElementById('valDyduGraph').innerHTML =
    katex.renderToString(`\\frac{dy}{du} = ${dyduVal.toFixed(4)}`, { throwOnError: false, displayMode: false, output: 'html' });

  // dy/dx 图
  clearGraph(dydxGraph);
  drawDerivativeAxes(dydxGraph, xRange, dydxRange);
  drawDerivativeCurve(dydxGraph, dydxFn, xRange, dydxRange, '#ef4444');
  const dydxVal = dydxFn(currentX);
  if (isFinite(dydxVal) && dydxVal >= dydxRange[0] && dydxVal <= dydxRange[1]) {
    drawPoint(dydxGraph, currentX, dydxVal);
  }
  document.getElementById('valDydxGraph').innerHTML =
    katex.renderToString(`\\frac{dy}{dx} = ${dydxVal.toFixed(4)}`, { throwOnError: false, displayMode: false, output: 'html' });
}

// 更新公式板
function updateFormulaBoard(values) {
  const inner = innerFunctions[currentInnerKey];
  const outer = outerFunctions[currentOuterKey];

  const formulaMain = document.getElementById('formulaMain');
  const formulaResult = document.getElementById('formulaResult');

  // 使用 KaTeX 渲染公式
  formulaMain.innerHTML = katex.renderToString(
    `\\frac{dy}{dx} = \\frac{dy}{du} \\times \\frac{du}{dx}`,
    { throwOnError: false }
  );

  formulaResult.innerHTML = katex.renderToString(
    `= ${values.dydx.toFixed(4)}`,
    { throwOnError: false }
  );

  document.getElementById('innerLabel').innerHTML = katex.renderToString(inner.latex, { throwOnError: false });
  document.getElementById('outerLabel').innerHTML = katex.renderToString(outer.latex, { throwOnError: false });
}

// 更新数值显示
function updateValueDisplay(values) {
  document.getElementById('valX').innerHTML = katex.renderToString(`x = ${currentX.toFixed(4)}`, { throwOnError: false, displayMode: false });
  document.getElementById('valU').innerHTML = katex.renderToString(`u = ${values.u.toFixed(4)}`, { throwOnError: false, displayMode: false });
  document.getElementById('valDudx').innerHTML = katex.renderToString(`\\frac{du}{dx} = ${values.dudx.toFixed(4)}`, { throwOnError: false, displayMode: false, output: 'html' });
  document.getElementById('valDydu').innerHTML = katex.renderToString(`\\frac{dy}{du} = ${values.dydu.toFixed(4)}`, { throwOnError: false, displayMode: false, output: 'html' });
  document.getElementById('valDydx').innerHTML = katex.renderToString(`\\frac{dy}{dx} = ${values.dydx.toFixed(4)}`, { throwOnError: false, displayMode: false, output: 'html' });
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);
