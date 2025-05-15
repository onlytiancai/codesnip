// 由于无法下载外部提示音，我们使用Web Audio API创建一个简单的提示音
document.addEventListener('DOMContentLoaded', function() {
    // 替换原有的音频元素引用
    const originalAudioElement = document.getElementById('alarm-sound');
    if (originalAudioElement) {
        originalAudioElement.remove();
    }
    
    // 创建一个音频上下文
    window.AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContext();
    
    // 创建提示音函数
    window.playAlarm = function() {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // 设置音调和音量
        oscillator.type = 'sine';
        oscillator.frequency.value = 800;
        gainNode.gain.value = 0.5;
        
        // 设置音量变化
        const now = audioContext.currentTime;
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.5, now + 0.1);
        gainNode.gain.linearRampToValueAtTime(0, now + 1.5);
        
        // 播放提示音
        oscillator.start(now);
        oscillator.stop(now + 1.5);
        
        // 创建第二个提示音（双音提示）
        setTimeout(() => {
            const oscillator2 = audioContext.createOscillator();
            const gainNode2 = audioContext.createGain();
            
            oscillator2.connect(gainNode2);
            gainNode2.connect(audioContext.destination);
            
            oscillator2.type = 'sine';
            oscillator2.frequency.value = 600;
            gainNode2.gain.value = 0.5;
            
            const now2 = audioContext.currentTime;
            gainNode2.gain.setValueAtTime(0, now2);
            gainNode2.gain.linearRampToValueAtTime(0.5, now2 + 0.1);
            gainNode2.gain.linearRampToValueAtTime(0, now2 + 1.5);
            
            oscillator2.start(now2);
            oscillator2.stop(now2 + 1.5);
        }, 700);
    };
});