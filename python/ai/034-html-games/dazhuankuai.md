我们可以把这份打砖块代码当成一个小型游戏引擎来理解。它虽然只有一个 HTML 文件，但已经包含了一个游戏最核心的组成部分：画面渲染、游戏循环、输入控制、物理运动、碰撞检测和游戏状态管理。

我按照代码实际执行的顺序，从整体架构讲到每个关键函数。理解这些内容后，你就可以自己修改规则，甚至从零开发其他 Canvas 小游戏。

# 一、先理解整体架构

整个文件分成三个部分：

### HTML：游戏结构

Canvas 画布、得分、生命值、按钮

JavaScript 操作页面元素

### JavaScript：游戏逻辑

更新坐标、处理输入、判断碰撞、管理分数

计算游戏状态并绘制画面

### Canvas 2D：画面渲染

绘制砖块、挡板、小球和提示文字

最重要的一个概念是：Canvas 本身只是画布，不会自动产生游戏行为。 你需要用 JavaScript 计算每个物体的位置，然后把它们画出来。

## 1. HTML 和 CSS：搭建游戏界面

例如：

HTML

```
<canvas id="game" width="800" height="600"></canvas>
```

这里创建了一块画布：

* `id="game"`：方便 JavaScript 找到它。

* `width="800"`：画布内部坐标宽度为 800。

* `height="600"`：画布内部坐标高度为 600。

注意，Canvas 有两套尺寸概念：内部绘图尺寸和 CSS 显示尺寸。当前代码通过 CSS 把画布缩放到页面宽度，但游戏内部仍然按照 `800 × 600` 的坐标计算。

然后：

JavaScript

```
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
```

第一行获取画布元素，第二行获取 2D 绘图上下文 `ctx`。

你可以把 `ctx` 理解为画笔，后续的矩形、圆形、文字，都是通过它绘制的。

例如：

JavaScript

```
ctx.fillStyle = "#ffffff";
ctx.fillRect(10, 10, 100, 30);
```

它会在坐标 `(10, 10)` 处画出一个宽 100、高 30 的白色矩形。

# 二、游戏中的物体是怎样表示的？

当前代码没有使用游戏引擎，而是直接用 JavaScript 对象保存物体的数据。

## 2. 小球

JavaScript

```
const ball = {
  x: 400,
  y: 540,
  r: 8,
  vx: 4,
  vy: -5
};
```

各字段的含义：

| 字段   | 含义       |
| ---- | -------- |
| `x`  | 小球圆心的横坐标 |
| `y`  | 小球圆心的纵坐标 |
| `r`  | 小球半径     |
| `vx` | 水平方向速度   |
| `vy` | 垂直方向速度   |

Canvas 的坐标原点位于左上角：

![XML Interchange Language for System Dynamics (XMILE) Version 1.0](https://images.openai.com/static-rsc-4/fykEYZ8_i5dOcWdunHdr64PUyHlPLZ53hCWqQr-e76eiDqHPwRCQ6IPzFd6yICS5gTDZLxq1UoZfTLjPAXXsup5FQtieNx_Sm6BmgwgV_yXmkssis1gpbaHj-rOROIUlRWRQy8Xw79uAbWs2a9bwia-F78z9E4K8i0fFWvVcshg?purpose=inline)

加入收藏

x 越大，物体越靠右；y 越大，物体越靠下。

假设小球当前的位置是：

```
x = 400
y = 540
vx = 4
vy = -5
```

经过一个更新步长后，代码执行：

JavaScript

```
ball.x += ball.vx;
ball.y += ball.vy;
```

结果就是：

```
x = 404
y = 535
```

小球向右移动了 4 个坐标单位，向上移动了 5 个坐标单位。

为什么 `vy` 为负数代表向上？因为 Canvas 的 y 轴方向和数学课中常见的坐标系相反。

这里要注意：`vx` 和 `vy` 是速度分量，并不是坐标本身。

## 3. 挡板

JavaScript

```
const paddle = {
  x: 340,
  y: 560,
  w: 120,
  h: 12,
  speed: 8
};
```

挡板使用矩形表示：

* `x`、`y`：矩形左上角。

* `w`、`h`：宽度和高度。

* `speed`：键盘操作时每个基准更新步长的移动距离。

例如 `paddle.x = 340`、`paddle.w = 120`，说明挡板从横坐标 340 延伸到 460。

## 4. 砖块

砖块不是一个整体，而是很多个对象组成的数组：

JavaScript

```
let bricks = [];
```

每个砖块都有自己的位置、尺寸、颜色和存活状态：

JavaScript

```
{
  x: 100,
  y: 70,
  w: 68,
  h: 22,
  color: "#fb7185",
  alive: true
}
```

其中 `alive` 非常关键：

* `true`：砖块还存在，需要绘制，也能发生碰撞。

* `false`：砖块已经被击中，不再绘制，也不再参与碰撞。

这就是游戏中最基础的对象状态管理。

# 三、游戏初始化：`reset()` 和 `createBricks()`

## 5. `createBricks()`：批量生成砖块

代码：

JavaScript

```
function createBricks() {
  bricks = [];

  const rows = 5, cols = 10;
  const gap = 8, bw = 68, bh = 22;
  const startX = (W - cols * (bw + gap) + gap) / 2;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      bricks.push({
        x: startX + c * (bw + gap),
        y: 70 + r * (bh + gap),
        w: bw, h: bh,
        color: ["#fb7185", "#fb923c", "#facc15",
                "#4ade80", "#38bdf8"][r],
        alive: true
      });
    }
  }
}
```

这里用两个循环生成 50 块砖。

行数 `rows`

# 5

列数 `cols`

# 10

砖块总数

# 50

每块分值

# 10

外层循环决定行，内层循环决定列：

JavaScript

```
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    // 创建一块砖
  }
}
```

例如：

* `r = 0, c = 0`：第一行第一列。

* `r = 0, c = 1`：第一行第二列。

* `r = 1, c = 0`：第二行第一列。

砖块横坐标由下面的表达式计算：

JavaScript

```
x = startX + c * (bw + gap);
```

相邻砖块之间的距离就是砖块宽度加间距，即 `68 + 8 = 76`。

纵坐标类似：

JavaScript

```
y = 70 + r * (bh + gap);
```

这样就不必手动写出 50 个砖块的位置。

## 6. `reset()`：重置整个游戏

JavaScript

```
function reset() {
  score = 0;
  lives = 3;
  running = false;
  over = false;

  paddle.x = (W - paddle.w) / 2;
  resetBall();
  createBricks();
  updateUI();
  statusEl.textContent = "按空格发球";
  draw();
}
```

这个函数负责让游戏回到初始状态。

执行顺序是：

1. 分数清零，生命恢复为 3。

2. 游戏变为未开始状态，结束标志清除。

3. 把挡板移动到画布中央。

4. 重置小球位置和速度。

5. 重新生成所有砖块。

6. 更新页面上的分数和生命值。

7. 绘制初始画面。

其中：

JavaScript

```
paddle.x = (W - paddle.w) / 2;
```

表示挡板水平居中。

而：

JavaScript

```
function resetBall() {
  ball.x = paddle.x + paddle.w / 2;
  ball.y = paddle.y - ball.r - 2;
  ball.vx = 4 * (Math.random() > 0.5 ? 1 : -1);
  ball.vy = -5;
}
```

会让小球回到挡板上方，并随机决定初始向左或向右运动的方向。

这里 `Math.random()` 产生一个 0 到 1 之间的随机数，因此小球每次重新发射时，初始水平方向可能不同。

# 四、游戏循环：整个游戏的核心

## 7. `requestAnimationFrame()` 是什么？

游戏不会只运行一次 `draw()` 就结束，而是需要持续更新和绘制。

当前代码的循环是：

JavaScript

```
function loop(time) {
  const dt = lastTime ? time - lastTime : 16.67;
  lastTime = time;

  update(dt);
  draw();

  requestAnimationFrame(loop);
}

reset();
requestAnimationFrame(loop);
```

可以把它理解为：

1. 获取当前帧的时间

2. `update(dt)` 更新游戏状态

移动小球、挡板，检测碰撞

3. `draw()` 绘制当前状态

清屏，重新画出所有物体

4. 请求浏览器执行下一帧

持续重复，直到页面停止运行

`requestAnimationFrame(loop)` 会请求浏览器在下一次合适的绘制时机调用 `loop`，通常接近屏幕刷新率，例如每秒 60 次。

### 为什么需要 `dt`？

JavaScript

```
const dt = lastTime ? time - lastTime : 16.67;
```

`time` 是浏览器传入的时间戳，单位是毫秒。`dt` 表示距离上一帧过去了多少毫秒。

随后：

JavaScript

```
const step = Math.min(dt / 16.67, 2);
```

代码把时间换算为相对于约 60 FPS 的更新步长。

例如：

| 两帧间隔 `dt` | `step` |
| --------- | ------ |
| 16.67 ms  | 约 1    |
| 33.33 ms  | 约 2    |
| 50 ms     | 被限制为 2 |

这个设计试图让游戏在帧率降低时仍然保持相近的运动速度，并限制单次更新过大的移动。

不过，当前代码使用的是简化方案：`step` 最大为 2。因此，如果浏览器卡顿特别严重，游戏仍可能变慢。这并不是严格的固定时间步长模拟。

# 五、`update()`：游戏逻辑是怎样执行的？

这是整份代码最值得理解的函数。它可以拆成四个阶段。

## 8. 第一阶段：移动挡板和小球

首先检查游戏是否正在进行：

JavaScript

```
if (!running || over) return;
```

如果游戏还没开始，或者已经结束，就不再执行物理更新。

接着处理键盘：

JavaScript

```
if (keys.ArrowLeft) paddle.x -= paddle.speed * step;
if (keys.ArrowRight) paddle.x += paddle.speed * step;
```

`keys` 是一个记录按键状态的对象。按下左箭头后，`keys.ArrowLeft` 为 `true`，挡板就向左移动。

为了避免挡板移出画布：

JavaScript

```
paddle.x = Math.max(
  0,
  Math.min(W - paddle.w, paddle.x)
);
```

这是一个常见的边界限制技巧。

* `Math.min(W - paddle.w, paddle.x)`：挡板不能超过右边界。

* `Math.max(0, ...)`：挡板不能小于左边界。

随后移动小球：

JavaScript

```
ball.x += ball.vx * step;
ball.y += ball.vy * step;
```

每一帧根据速度更新位置，这就是最简单的运动模拟。

## 9. 第二阶段：检测墙壁碰撞

左侧墙壁的代码：

JavaScript

```
if (ball.x - ball.r < 0) {
  ball.x = ball.r;
  ball.vx = Math.abs(ball.vx);
}
```

这里有两个动作。

第一步：修正位置。

JavaScript

```
ball.x = ball.r;
```

避免小球有一部分进入墙壁。

第二步：反转速度方向。

JavaScript

```
ball.vx = Math.abs(ball.vx);
```

`Math.abs()` 会把速度转换成非负数，因此小球碰到左墙后会向右移动。

右墙的逻辑相反：

JavaScript

```
ball.vx = -Math.abs(ball.vx);
```

顶部墙壁则修改 `vy`，让小球向下反弹。

注意：代码没有在底部设置反弹墙，因为小球落出画布意味着失去一次生命。

## 10. 第三阶段：检测挡板碰撞

这是关键的一段：

JavaScript

```
if (
  ball.vy > 0 &&
  ball.y + ball.r >= paddle.y &&
  ball.y - ball.r <= paddle.y + paddle.h &&
  ball.x >= paddle.x &&
  ball.x <= paddle.x + paddle.w
) {
  // 反弹
}
```

它判断小球是否正在向下运动，以及小球圆形的范围是否与挡板矩形重叠。

具体来说：

* `ball.vy > 0`：小球正在向下。

* `ball.y + ball.r >= paddle.y`：小球底部已经到达挡板顶部。

* `ball.y - ball.r <= paddle.y + paddle.h`：小球顶部没有完全越过挡板底部。

* 最后两个条件：小球圆心的横坐标位于挡板左右边界内。

这是一个简化的圆形与矩形碰撞判断，并不完全精确：它没有把圆形和矩形的最近点距离作为判据。

### 为什么击中挡板不同位置，反弹角度也不同？

JavaScript

```
const hit =
  (ball.x - (paddle.x + paddle.w / 2))
  / (paddle.w / 2);

const angle = hit * Math.PI / 3;
```

`hit` 是小球击中位置相对于挡板中心的归一化偏移。

击中左边

击中中心

击中右边

`hit = -1`

`hit = 0`

`hit = 1`

左侧击中

小球向左上方反弹。

中心击中

小球近乎垂直向上反弹。

右侧击中

小球向右上方反弹。

接下来：

JavaScript

```
const speed = Math.min(
  9,
  Math.hypot(ball.vx, ball.vy) + 0.1
);

ball.vx = speed * Math.sin(angle);
ball.vy = -speed * Math.cos(angle);
```

这里使用三角函数把速度大小转换为水平和垂直两个分量。

* `Math.hypot(ball.vx, ball.vy)`：计算当前速度的大小。

* `Math.sin(angle)`：计算水平方向的分量。

* `Math.cos(angle)`：计算垂直方向的分量。

* `-` 号：确保反弹后小球朝上运动。

* `Math.min(9, ...)`：限制速度最大为 9。

这种设计比单纯执行 `ball.vy *= -1` 更有趣，因为玩家可以通过控制击球位置来改变球的方向。

## 11. 第四阶段：砖块碰撞检测

代码遍历所有砖块：

JavaScript

```
for (const b of bricks) {
  if (!b.alive) continue;

  if (
    ball.x + ball.r > b.x &&
    ball.x - ball.r < b.x + b.w &&
    ball.y + ball.r > b.y &&
    ball.y - ball.r < b.y + b.h
  ) {
    // 击中砖块
  }
}
```

这段代码使用的是圆形小球的外接正方形与砖块矩形重叠检测。

可以把小球看成一个边长为 `2 * ball.r` 的正方形，然后检查这个正方形和砖块矩形是否相交。

四个条件分别判断：

1. 小球右边超过砖块左边。

2. 小球左边没有超过砖块右边。

3. 小球底部超过砖块顶部。

4. 小球顶部没有超过砖块底部。

四个条件同时成立，就认为发生了碰撞。

这种算法实现简单、计算量小，但因为忽略了小球的真实圆形边界，所以存在少量误判可能。

### 击中后发生什么？

JavaScript

```
b.alive = false;
score += 10;
```

首先让砖块消失，然后增加 10 分。

接着，代码计算小球与砖块在水平和垂直方向上的重叠深度：

JavaScript

```
const overlapX = Math.min(
  ball.x + ball.r - b.x,
  b.x + b.w - (ball.x - ball.r)
);

const overlapY = Math.min(
  ball.y + ball.r - b.y,
  b.y + b.h - (ball.y - ball.r)
);
```

然后：

JavaScript

```
if (overlapX < overlapY) ball.vx *= -1;
else ball.vy *= -1;
```

意思是：

* 水平方向重叠较少：认为小球撞到了砖块侧面，反转 `vx`。

* 垂直方向重叠较少：认为小球撞到了砖块顶部或底部，反转 `vy`。

最后：

JavaScript

```
updateUI();
break;
```

更新得分显示，并跳出循环。

这里的 `break` 很重要：一次更新最多处理一块砖，避免同一帧连续消除多个砖块。但当前实现没有修正碰撞后的球体位置，在高速运动或特殊角度下，仍可能发生重复碰撞或穿透现象。

## 12. 游戏胜利与失败

胜利条件：

JavaScript

```
if (bricks.every(b => !b.alive)) {
  over = true;
  running = false;
  statusEl.textContent = "🎉 你赢了！";
}
```

`every()` 会检查数组中的每一项是否满足条件。

`!b.alive` 表示砖块已经被消除。

所以，当所有砖块都被消除时，游戏结束并判定胜利。

失败条件：

JavaScript

```
if (ball.y - ball.r > H) {
  lives--;
  updateUI();

  if (lives <= 0) {
    over = true;
    running = false;
    statusEl.textContent = "游戏结束";
  } else {
    running = false;
    resetBall();
    statusEl.textContent = "按空格继续";
  }
}
```

当小球整个越过画布底部时：

1. 生命值减一。

2. 更新页面显示。

3. 如果还有生命，就重置小球并等待再次发球。

4. 如果生命耗尽，就结束游戏。

这里要注意一个细节：小球落下后，`running` 变为 `false`，因此挡板仍可移动，但小球要等玩家再次按空格或者点击画布才会重新运动。

# 六、`draw()`：如何把游戏状态变成画面？

`update()` 改变游戏数据，`draw()` 把这些数据显示出来。

## 13. 为什么每一帧都要清屏？

JavaScript

```
ctx.clearRect(0, 0, W, H);
```

这行代码清除画布上的旧画面。

假如上一帧小球在 `(400, 300)`，下一帧移动到了 `(404, 295)`。如果不清除旧画面，画布上就会留下两个小球，最终形成一条运动轨迹。

清屏后再根据最新坐标绘制，屏幕上看起来就只有一个小球在移动。

## 14. 绘制砖块、挡板和小球

砖块：

JavaScript

```
ctx.fillStyle = b.color;
ctx.beginPath();
ctx.roundRect(b.x, b.y, b.w, b.h, 5);
ctx.fill();
```

这里使用 `roundRect()` 绘制圆角矩形，最后用 `fill()` 填充颜色。

挡板使用同样的方式绘制。

小球则使用：

JavaScript

```
ctx.beginPath();
ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
ctx.fill();
```

`arc()` 的参数分别表示圆心坐标、半径、起始角度和结束角度。结束角度 `Math.PI * 2` 就是一整圈，因此绘制出完整的圆形。

发光效果来自：

JavaScript

```
ctx.shadowColor = "#38bdf8";
ctx.shadowBlur = 14;
```

它会让小球产生蓝色光晕。绘制完小球后，代码又执行：

JavaScript

```
ctx.shadowBlur = 0;
```

防止后续绘制的物体也受到同样的阴影设置影响。

## 15. 游戏结束遮罩

当 `over` 为 `true` 时，代码绘制一个半透明矩形：

JavaScript

```
ctx.fillStyle = "rgba(2, 6, 23, 0.72)";
ctx.fillRect(0, 0, W, H);
```

`rgba()` 中最后一个参数 `0.72` 表示透明度。

随后在画布中央绘制“恭喜通关”或者“游戏结束”的文字。

这里的文字是 Canvas 绘制的，不是 HTML 页面中的普通文字，因此它会跟随画布缩放。

# 七、键盘、鼠标和触屏是怎么控制游戏的？

## 16. 键盘事件

JavaScript

```
document.addEventListener("keydown", e => {
  keys[e.key] = true;

  if (e.code === "Space") start();
});

document.addEventListener("keyup", e => {
  keys[e.key] = false;
});
```

`keydown` 是按键按下事件，`keyup` 是按键释放事件。

为什么不直接在 `keydown` 中移动挡板？因为按住方向键时，代码只需将状态记录为 `true`，再由每一帧的 `update()` 持续移动挡板。这样操作更加平滑，也不依赖浏览器重复触发键盘事件的频率。

另外，代码通过：

JavaScript

```
if (["ArrowLeft", "ArrowRight", " "].includes(e.key)) {
  e.preventDefault();
}
```

阻止方向键和空格键触发浏览器默认行为，例如滚动页面。

## 17. 鼠标与触屏控制

JavaScript

```
canvas.addEventListener("pointermove", movePaddle);
canvas.addEventListener("pointerdown", e => {
  movePaddle(e);
  if (!over) start();
});
```

`pointer` 事件可以统一处理鼠标、触控笔和触屏输入。

`movePaddle()` 负责把鼠标位置换算成游戏内部坐标：

JavaScript

```
const rect = canvas.getBoundingClientRect();
const scale = W / rect.width;

paddle.x = (e.clientX - rect.left) * scale
            - paddle.w / 2;
```

为什么要做坐标转换？

因为画布在网页上可能显示为 600 像素宽，但内部绘图坐标宽度仍是 800。鼠标位置必须转换回内部坐标，否则挡板会跟随得不准确。

这里的 `scale` 就是内部宽度与显示宽度的比例。

# 八、这份代码还有哪些值得改进的地方？

理解现有代码后，还需要知道它目前有哪些简化处理。

| 问题    | 当前实现                     | 可以怎样改进                                |
| ----- | ------------------------ | ------------------------------------- |
| 碰撞精度  | 用外接矩形近似圆形小球              | 使用圆形与矩形的最近点距离判断                       |
| 高速穿透  | 直接更新坐标后检测碰撞              | 使用连续碰撞检测或更小的固定时间步长                    |
| 多块砖碰撞 | 每帧最多处理一块砖                | 计算最早碰撞时间，处理多次碰撞                       |
| 游戏状态  | 使用两个布尔值 `running`、`over` | 使用 `ready`、`playing`、`won`、`lost` 等状态 |
| 物理更新  | `step` 最大为 2             | 使用固定时间步长累积器                           |
| 触屏操作  | 指针事件控制挡板                 | 加入移动端按钮、暂停和更明确的发球操作                   |

还有一个细节：当前 `pointerdown` 在游戏未结束时会调用 `start()`，所以点击画布不仅可以发球，也可以在游戏进行中重复触发开始逻辑。实际游戏中最好明确区分“准备发球”和“游戏进行中”两种状态。

# 九、建议怎样学习这份代码？

如果你的目标是掌握 Canvas 游戏开发，我建议按照下面的顺序动手实践。

### 实践学习清单

0/5

第一步：只画出小球

理解 ctx.arc()、坐标和半径。

第二步：让小球移动

理解 vx、vy 和 requestAnimationFrame()。

第三步：让小球反弹

先实现墙壁碰撞，再实现挡板碰撞。

第四步：添加砖块

理解数组、循环和碰撞检测。

第五步：完善游戏规则

实现计分、生命值、胜负状态和重新开始。

最后记住三个最重要的函数：

* `update(dt)`：决定游戏世界下一刻会发生什么。

* `draw()`：把当前游戏世界画到画布上。

* `loop(time)`：让更新与绘制持续运行。

掌握这三个函数，再理解游戏对象和碰撞检测，你就已经掌握了大部分简单 2D Canvas 游戏的基础原理。

