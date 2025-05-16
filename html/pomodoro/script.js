// 番茄钟应用的主要JavaScript代码

// 定义常量
const WORK_TIME = 25 * 60; // 25分钟（以秒为单位）
const SHORT_BREAK_TIME = 5 * 60; // 5分钟
const LONG_BREAK_TIME = 15 * 60; // 15分钟
const POMODORO_CYCLE = 4; // 4个番茄钟后长休息

// DOM元素
const timeLeftDisplay = document.getElementById('time-left');
const modeLabel = document.getElementById('mode-label');
const startBtn = document.getElementById('start-btn');
const resetBtn = document.getElementById('reset-btn');
const workBtn = document.getElementById('work-btn');
const shortBreakBtn = document.getElementById('short-break-btn');
const longBreakBtn = document.getElementById('long-break-btn');
const completedCountDisplay = document.getElementById('completed-count');
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const settingsToggle = document.querySelector('.settings-toggle');
const settingsContent = document.querySelector('.settings-content');
const settingsToggleIcon = document.getElementById('settings-toggle-icon');
const notificationHelp = document.getElementById('notification-help');

// 应用状态
let timer = null;
let timeLeft = WORK_TIME;
let isRunning = false;
let currentMode = 'work';
let completedPomodoros = 0;
let pomodoroSequence = 0;
let screensaver = null;
let celebration = null;

// 初始化应用
function initApp() {
    updateTimeDisplay();
    loadCompletedPomodoros();
    loadTheme();
    setupUI();
    
    // 初始化屏保和庆祝特效
    screensaver = new Screensaver();
    celebration = new Celebration();
    
    // 设置主题切换按钮图标
    themeToggleBtn.innerHTML = ICONS.theme;
    
    // 设置设置切换图标
    settingsToggleIcon.innerHTML = '▼';
}

// 设置UI
function setupUI() {
    // 设置切换
    if (settingsToggle) {
        settingsToggle.addEventListener('click', () => {
            settingsContent.classList.toggle('open');
            settingsToggle.classList.toggle('open');
            settingsToggleIcon.textContent = settingsContent.classList.contains('open') ? '▲' : '▼';
        });
    }
    
    // 通知帮助
    if (notificationHelp) {
        notificationHelp.addEventListener('click', () => {
            if (window.openNotificationHelp) {
                window.openNotificationHelp();
            }
        });
    }
}

// 更新时间显示
function updateTimeDisplay() {
    const minutes = Math.floor(timeLeft / 60);
    const seconds = timeLeft % 60;
    timeLeftDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    document.title = `${timeLeftDisplay.textContent} - 蛙蛙番茄钟`;
}

// 开始/暂停计时器
function toggleTimer() {
    if (isRunning) {
        clearInterval(timer);
        startBtn.textContent = '开始';
        
        // 如果在休息模式且屏保正在显示，则隐藏屏保
        // 不在这里调用hide，避免递归调用
    } else {
        timer = setInterval(() => {
            timeLeft--;
            updateTimeDisplay();
            
            // 如果屏保正在显示，更新屏保上的时间
            if (screensaver && (currentMode === 'shortBreak' || currentMode === 'longBreak')) {
                screensaver.updateTimer(timeLeft);
            }
            
            if (timeLeft <= 0) {
                clearInterval(timer);
                playAlarm(); // 使用alarm.js中定义的函数
                completePomodoro();
            }
        }, 1000);
        startBtn.textContent = '暂停';
        
        // 如果是休息模式且全屏设置开启，显示屏保
        if ((currentMode === 'shortBreak' || currentMode === 'longBreak') && screensaver && window.isFullscreenEnabled()) {
            screensaver.show(timeLeft, () => {
                // 避免递归调用
                if (isRunning) {
                    isRunning = false;
                    startBtn.textContent = '开始';
                    clearInterval(timer);
                }
            });
        }
    }
    isRunning = !isRunning;
}

// 重置计时器
function resetTimer() {
    clearInterval(timer);
    
    // 如果屏保正在显示，隐藏它
    if (screensaver) {
        screensaver.hide();
    }
    
    setMode(currentMode);
    isRunning = false;
    startBtn.textContent = '开始';
}

// 设置模式
function setMode(mode) {
    // 移除所有模式按钮的active类
    workBtn.classList.remove('active');
    shortBreakBtn.classList.remove('active');
    longBreakBtn.classList.remove('active');
    
    // 如果从休息模式切换到工作模式，隐藏屏保
    if ((currentMode === 'shortBreak' || currentMode === 'longBreak') && mode === 'work') {
        if (screensaver) {
            screensaver.hide();
        }
    }
    
    currentMode = mode;
    
    switch (mode) {
        case 'work':
            timeLeft = WORK_TIME;
            modeLabel.textContent = '工作模式';
            workBtn.classList.add('active');
            break;
        case 'shortBreak':
            timeLeft = SHORT_BREAK_TIME;
            modeLabel.textContent = '短休息';
            shortBreakBtn.classList.add('active');
            break;
        case 'longBreak':
            timeLeft = LONG_BREAK_TIME;
            modeLabel.textContent = '长休息';
            longBreakBtn.classList.add('active');
            break;
    }
    
    updateTimeDisplay();
    
    // 如果是休息模式且计时器正在运行且全屏设置开启，显示屏保
    if ((mode === 'shortBreak' || mode === 'longBreak') && isRunning && screensaver && window.isFullscreenEnabled()) {
        screensaver.show(timeLeft, () => {
            // 当用户退出屏保时，如果计时器仍在运行，暂停计时器
            if (isRunning) {
                isRunning = false;
                startBtn.textContent = '开始';
                clearInterval(timer);
            }
        });
    }
}

// 完成一个番茄钟
function completePomodoro() {
    if (currentMode === 'work') {
        completedPomodoros++;
        pomodoroSequence++;
        saveCompletedPomodoros();
        updateCompletedDisplay();
        
        // 显示庆祝特效
        if (celebration) {
            celebration.show('恭喜完成一个番茄钟！');
        }
        
        // 显示浏览器通知
        if (window.showInAppNotification) {
            window.showInAppNotification('蛙蛙番茄钟', '恭喜完成一个番茄钟！休息一下吧~');
        }
        
        if (window.showNotification) {
            window.showNotification('蛙蛙番茄钟', '恭喜完成一个番茄钟！休息一下吧~');
        }
        
        // 重置计时器但不自动切换到休息模式
        clearInterval(timer);
        timeLeft = WORK_TIME;
        updateTimeDisplay();
        isRunning = false;
        startBtn.textContent = '开始';
        
        // 更新模式计数
        if (pomodoroSequence >= POMODORO_CYCLE) {
            pomodoroSequence = 0;
        }
    } else {
        // 休息结束后回到工作模式
        setMode('work');
        
        // 自动开始下一个计时
        toggleTimer();
    }
}

// 更新已完成番茄钟显示
function updateCompletedDisplay() {
    completedCountDisplay.textContent = completedPomodoros;
}

// 保存已完成番茄钟数量
function saveCompletedPomodoros() {
    localStorage.setItem('completedPomodoros', completedPomodoros);
}

// 加载已完成番茄钟数量
function loadCompletedPomodoros() {
    const saved = localStorage.getItem('completedPomodoros');
    if (saved) {
        completedPomodoros = parseInt(saved);
        updateCompletedDisplay();
    }
}

// 主题切换功能
function toggleTheme() {
    const body = document.body;
    if (body.classList.contains('light-theme')) {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
    } else {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        localStorage.setItem('theme', 'light');
    }
}

// 加载保存的主题
function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    const body = document.body;
    
    // 确保body有一个默认主题类
    if (!body.classList.contains('light-theme') && !body.classList.contains('dark-theme')) {
        body.classList.add('light-theme');
    }
    
    if (savedTheme === 'dark') {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
    } else {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
    }
}

// 事件监听器
startBtn.addEventListener('click', toggleTimer);
resetBtn.addEventListener('click', resetTimer);
workBtn.addEventListener('click', () => setMode('work'));
shortBreakBtn.addEventListener('click', () => setMode('shortBreak'));
longBreakBtn.addEventListener('click', () => setMode('longBreak'));
themeToggleBtn.addEventListener('click', toggleTheme);

// 初始化应用
document.addEventListener('DOMContentLoaded', initApp);