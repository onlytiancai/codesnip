// 通知功能
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const notificationSetting = document.getElementById('notification-setting');
    const fullscreenSetting = document.getElementById('fullscreen-setting');
    
    // 初始化
    let notificationPermission = 'default';
    
    // 加载设置
    loadSettings();
    
    // 检查通知权限
    checkNotificationPermission();
    
    // 加载设置
    function loadSettings() {
        // 加载全屏设置
        const fullscreenEnabled = localStorage.getItem('fullscreenEnabled');
        if (fullscreenEnabled !== null) {
            fullscreenSetting.checked = fullscreenEnabled === 'true';
        }
        
        // 加载通知设置
        const notificationEnabled = localStorage.getItem('notificationEnabled');
        if (notificationEnabled !== null) {
            notificationSetting.checked = notificationEnabled === 'true';
        }
        
        // 添加设置变更事件监听
        fullscreenSetting.addEventListener('change', () => {
            localStorage.setItem('fullscreenEnabled', fullscreenSetting.checked);
        });
        
        notificationSetting.addEventListener('change', () => {
            localStorage.setItem('notificationEnabled', notificationSetting.checked);
            if (notificationSetting.checked) {
                requestNotificationPermission();
            }
        });
    }
    
    // 检查通知权限
    function checkNotificationPermission() {
        if (!('Notification' in window)) {
            console.log('浏览器不支持通知');
            notificationSetting.disabled = true;
            return;
        }
        
        notificationPermission = Notification.permission;
        
        if (notificationPermission === 'granted') {
            console.log('通知权限已获取');
        } else if (notificationPermission === 'denied') {
            console.log('通知权限被拒绝');
            notificationSetting.disabled = true;
        }
    }
    
    // 请求通知权限
    function requestNotificationPermission() {
        if (!('Notification' in window)) {
            return;
        }
        
        if (notificationPermission !== 'granted' && notificationPermission !== 'denied') {
            Notification.requestPermission().then(permission => {
                notificationPermission = permission;
                if (permission !== 'granted') {
                    notificationSetting.checked = false;
                    localStorage.setItem('notificationEnabled', 'false');
                }
            });
        }
    }
    
    // 显示浏览器通知
    window.showNotification = function(title, message) {
        if (!('Notification' in window) || notificationPermission !== 'granted' || !notificationSetting.checked) {
            return;
        }
        
        const notification = new Notification(title, {
            body: message,
            icon: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="50" height="50"><circle cx="50" cy="50" r="45" fill="%23e74c3c"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="40" fill="white">🍅</text></svg>'
        });
        
        // 5秒后自动关闭
        setTimeout(() => {
            notification.close();
        }, 5000);
    };
    
    // 导出全屏设置状态
    window.isFullscreenEnabled = function() {
        return fullscreenSetting.checked;
    };
});