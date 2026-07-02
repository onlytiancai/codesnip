// 动画控制

class AnimationController {
  constructor() {
    this.isPlaying = false;
    this.speed = 1; // 1x, 2x, 0.5x
    this.duration = 4000; // 完整周期 4 秒
    this.currentTime = 0;
    this.lastTimestamp = 0;
    this.animationId = null;
    this.onUpdate = null;
    this.xMin = -2;
    this.xMax = 2;
  }

  setRange(min, max) {
    this.xMin = min;
    this.xMax = max;
  }

  setSpeed(speed) {
    this.speed = speed;
  }

  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.lastTimestamp = performance.now();
    this.animate();
  }

  pause() {
    this.isPlaying = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  toggle() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  animate() {
    if (!this.isPlaying) return;

    const now = performance.now();
    const delta = now - this.lastTimestamp;
    this.lastTimestamp = now;

    // 根据速度调整增量
    const cycleTime = this.duration / this.speed;
    this.currentTime = (this.currentTime + delta) % cycleTime;

    // 计算当前 x 值（使用 smoothstep 实现平滑往复）
    const t = this.currentTime / cycleTime;
    const smoothT = this.smoothstep(t);

    // 往复运动
    const x = this.xMin + (this.xMax - this.xMin) * (t < 0.5 ? t * 2 : 2 - t * 2);

    if (this.onUpdate) {
      this.onUpdate(x);
    }

    this.animationId = requestAnimationFrame(() => this.animate());
  }

  // Smoothstep 函数用于平滑过渡
  smoothstep(t) {
    return t * t * (3 - 2 * t);
  }

  setProgress(x) {
    // 直接设置当前 x 值（从滑块）
    if (this.onUpdate) {
      this.onUpdate(x);
    }
  }
}
