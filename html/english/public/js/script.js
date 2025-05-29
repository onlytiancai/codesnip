// 首页脚本
document.addEventListener('DOMContentLoaded', function() {
  // 搜索功能
  const searchInput = document.getElementById('search');
  if (searchInput) {
    searchInput.addEventListener('input', function() {
      const searchTerm = this.value.toLowerCase();
      const items = document.querySelectorAll('.dialogue-item');
      
      items.forEach(item => {
        const title = item.querySelector('.dialogue-title').textContent.toLowerCase();
        const scenarios = item.querySelector('.dialogue-scenarios').textContent.toLowerCase();
        
        if (title.includes(searchTerm) || scenarios.includes(searchTerm)) {
          item.style.display = '';
        } else {
          item.style.display = 'none';
        }
      });
    });
  }
  
  // 详情页脚本 - 点击英文显示中文
  bindTranslationToggle();
});

// 绑定翻译切换事件的函数
function bindTranslationToggle() {
  const englishTexts = document.querySelectorAll('.english');
  englishTexts.forEach(text => {
    text.addEventListener('click', function() {
      const chineseText = this.nextElementSibling;
      if (chineseText.style.display === 'block') {
        chineseText.style.display = 'none';
      } else {
        chineseText.style.display = 'block';
      }
    });
  });
}