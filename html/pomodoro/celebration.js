// 庆祝特效功能
class Celebration {
    constructor() {
        this.createCelebrationElement();
    }

    createCelebrationElement() {
        // 创建庆祝容器
        this.element = document.createElement('div');
        this.element.className = 'celebration-container';
        document.body.appendChild(this.element);
    }

    // 显示庆祝特效
    show(message) {
        this.element.classList.add('active');
        this.createConfetti();
        this.showCongratulationMessage(message);
        
        // 3秒后自动隐藏
        setTimeout(() => {
            this.hide();
        }, 3000);
    }

    // 隐藏庆祝特效
    hide() {
        this.element.classList.remove('active');
        this.element.innerHTML = '';
    }

    // 创建彩色纸屑
    createConfetti() {
        const colors = ['#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5', 
                       '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50', 
                       '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722'];
        
        // 创建100个纸屑
        for (let i = 0; i < 100; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            
            // 随机颜色
            const color = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.backgroundColor = color;
            
            // 随机位置
            confetti.style.left = `${Math.random() * 100}%`;
            
            // 随机大小
            const size = Math.random() * 10 + 5;
            confetti.style.width = `${size}px`;
            confetti.style.height = `${size}px`;
            
            // 随机形状
            if (Math.random() > 0.5) {
                confetti.style.borderRadius = '50%';
            } else {
                confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
            }
            
            // 随机动画延迟
            confetti.style.animationDelay = `${Math.random() * 2}s`;
            
            this.element.appendChild(confetti);
        }
    }

    // 显示祝贺消息
    showCongratulationMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'congratulation-message';
        messageElement.textContent = message || '恭喜完成！';
        this.element.appendChild(messageElement);
    }
}

// 导出庆祝类
window.Celebration = Celebration;