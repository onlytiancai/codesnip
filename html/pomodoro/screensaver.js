// 屏保功能
class Screensaver {
    constructor() {
        this.createScreensaverElement();
        this.active = false;
        this.bubbles = [];
        this.maxBubbles = 15;
        this.animationFrame = null;
    }

    createScreensaverElement() {
        // 创建屏保容器
        this.element = document.createElement('div');
        this.element.className = 'screensaver';
        
        // 创建计时器显示
        this.timerElement = document.createElement('div');
        this.timerElement.className = 'timer-overlay';
        this.element.appendChild(this.timerElement);
        
        // 创建退出按钮
        this.exitButton = document.createElement('button');
        this.exitButton.className = 'exit-button';
        this.exitButton.textContent = '退出屏保';
        this.exitButton.addEventListener('click', () => this.hide());
        this.element.appendChild(this.exitButton);
        
        // 添加到body
        document.body.appendChild(this.element);
    }

    show(timeLeft, onExit) {
        this.active = true;
        this.element.classList.add('active');
        this.onExit = onExit;
        this.updateTimer(timeLeft);
        this.createBubbles();
        this.animate();
        
        // 禁止页面滚动
        document.body.style.overflow = 'hidden';
    }

    hide() {
        this.active = false;
        this.element.classList.remove('active');
        
        // 清除所有气泡
        this.bubbles.forEach(bubble => {
            if (bubble.parentNode) {
                bubble.parentNode.removeChild(bubble);
            }
        });
        this.bubbles = [];
        
        // 取消动画
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        
        // 恢复页面滚动
        document.body.style.overflow = '';
        
        // 调用退出回调
        if (this.onExit) {
            this.onExit();
        }
    }

    updateTimer(timeLeft) {
        const minutes = Math.floor(timeLeft / 60);
        const seconds = timeLeft % 60;
        this.timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    createBubbles() {
        // 清除现有气泡
        this.bubbles.forEach(bubble => {
            if (bubble.parentNode) {
                bubble.parentNode.removeChild(bubble);
            }
        });
        this.bubbles = [];
        
        // 创建新气泡
        for (let i = 0; i < this.maxBubbles; i++) {
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            
            // 随机大小和位置
            const size = Math.random() * 100 + 50;
            bubble.style.width = `${size}px`;
            bubble.style.height = `${size}px`;
            bubble.style.left = `${Math.random() * 100}%`;
            bubble.style.top = `${Math.random() * 100}%`;
            
            // 随机颜色
            const hue = Math.random() * 360;
            bubble.style.background = `radial-gradient(circle at 30% 30%, hsla(${hue}, 100%, 70%, 0.8), hsla(${hue}, 100%, 50%, 0.1))`;
            
            // 随机动画延迟
            bubble.style.animationDelay = `${Math.random() * 5}s`;
            
            this.element.appendChild(bubble);
            this.bubbles.push(bubble);
        }
    }

    animate() {
        if (!this.active) return;
        
        // 移动气泡
        this.bubbles.forEach(bubble => {
            const currentTransform = getComputedStyle(bubble).transform;
            if (currentTransform === 'none') return;
            
            // 添加微小的随机移动
            const x = (Math.random() - 0.5) * 2;
            const y = (Math.random() - 0.5) * 2;
            bubble.style.transform = `${currentTransform} translate(${x}px, ${y}px)`;
        });
        
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
}

// 导出屏保类
window.Screensaver = Screensaver;