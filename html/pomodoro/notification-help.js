// 通知帮助功能
document.addEventListener('DOMContentLoaded', function() {
    // 创建帮助模态框
    const helpModal = document.createElement('div');
    helpModal.className = 'modal';
    helpModal.id = 'notification-help-modal';
    
    helpModal.innerHTML = `
        <div class="modal-content">
            <button class="modal-close">${ICONS.close}</button>
            <h3 class="modal-title">如何启用浏览器通知</h3>
            <div class="modal-body">
                <p>如果您没有收到通知，可能是由于操作系统或浏览器的设置限制。请按照以下步骤检查：</p>
                
                <h4>Windows 系统设置：</h4>
                <ul>
                    <li>打开 设置 > 系统 > 通知</li>
                    <li>确保"从应用和其他发送者获取通知"已开启</li>
                    <li>确保"专注助手"已关闭</li>
                    <li>检查 Google Chrome（或您使用的浏览器）是否有权限发送通知</li>
                </ul>
                
                <h4>macOS 系统设置：</h4>
                <ul>
                    <li>打开 系统设置 > 通知与专注模式</li>
                    <li>滚动找到 Google Chrome（或您使用的浏览器）</li>
                    <li>确保"允许通知"已勾选</li>
                    <li>您还可以调整其他通知设置，如提醒样式、声音和角标</li>
                </ul>
                
                <h4>浏览器设置：</h4>
                <ul>
                    <li>Chrome：打开 设置 > 隐私和安全 > 网站设置 > 通知</li>
                    <li>Firefox：打开 设置 > 隐私与安全 > 权限 > 通知</li>
                    <li>Edge：打开 设置 > Cookie和网站权限 > 通知</li>
                </ul>
                
                <div style="margin-top: 20px; text-align: center;">
                    <button id="test-notification-in-modal" class="btn">测试通知</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(helpModal);
    
    // 添加事件监听器
    const closeBtn = helpModal.querySelector('.modal-close');
    closeBtn.addEventListener('click', () => {
        helpModal.classList.remove('open');
    });
    
    // 点击模态框外部关闭
    helpModal.addEventListener('click', (e) => {
        if (e.target === helpModal) {
            helpModal.classList.remove('open');
        }
    });
    
    // 测试通知按钮
    const testBtn = helpModal.querySelector('#test-notification-in-modal');
    testBtn.addEventListener('click', () => {
        if (window.showInAppNotification) {
            window.showInAppNotification('测试通知', '这是一条测试通知，如果您看到这条消息，说明应用内通知功能正常工作！');
        }
        
        if (window.showNotification) {
            window.showNotification('测试通知', '这是一条测试通知，如果您看到这条消息，说明浏览器通知功能正常工作！');
        }
    });
    
    // 导出打开帮助模态框的函数
    window.openNotificationHelp = function() {
        helpModal.classList.add('open');
    };
});