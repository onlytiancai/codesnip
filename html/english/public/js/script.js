// 详情页脚本 - 点击英文显示中文
document.addEventListener('DOMContentLoaded', function() {
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