// 显示消息提示
export const showToast = (content: string, type: string = 'info', duration: number = 3000): void => {
  // 创建消息元素
  const messageElement = document.createElement('div');
  messageElement.className = `fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-lg transition-all duration-300 ease-in-out transform translate-y-0 opacity-100`;
  
  // 根据类型设置样式
  switch (type) {
    case 'info':
      messageElement.classList.add('bg-blue-500', 'text-white');
      break;
    case 'success':
      messageElement.classList.add('bg-green-500', 'text-white');
      break;
    case 'error':
      messageElement.classList.add('bg-red-500', 'text-white');
      break;
    case 'warning':
      messageElement.classList.add('bg-yellow-500', 'text-white');
      break;
    default:
      messageElement.classList.add('bg-blue-500', 'text-white');
  }
  
  messageElement.textContent = content;
  document.body.appendChild(messageElement);
  
  // 3秒后自动关闭
  setTimeout(() => {
    messageElement.classList.add('opacity-0', 'translate-y-[-10px]');
    setTimeout(() => {
      if (messageElement.parentNode) {
        messageElement.parentNode.removeChild(messageElement);
      }
    }, 300);
  }, duration);
};

// 下载视频到本地
export const downloadVideo = (videoUrl: string, filename: string = 'english-reading-video.mp4'): void => {
  if (!videoUrl) return;
  
  const link = document.createElement('a');
  link.href = videoUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
