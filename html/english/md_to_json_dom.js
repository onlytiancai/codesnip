const fs = require('fs');
const path = require('path');
const marked = require('marked');
const { JSDOM } = require('jsdom');

function convertMarkdownToJson(filename) {
  // 读取md文件
  const filePath = path.join(__dirname, 'dialogues', filename);
  const content = fs.readFileSync(filePath, 'utf8');

  // 使用marked将markdown转换为HTML
  const html = marked.parse(content);

  // 使用jsdom解析HTML
  const dom = new JSDOM(html);
  const document = dom.window.document;

  // 提取标题和类别信息
  const titleElement = document.querySelector('h3');
  const titleText = titleElement ? titleElement.textContent : '';
  const titleMatch = titleText.match(/(\d+)\.\s+([^(]+)\s*\(([^)]+)\)/);

  const result = {
    number: titleMatch ? parseInt(titleMatch[1]) : 2,
    category: {
      zh: titleMatch ? titleMatch[2].trim() : '',
      en: titleMatch ? titleMatch[3].trim() : ''
    },
    scenarios: []
  };

  // 提取场景信息
  const paragraphs = document.querySelectorAll('p');
  let scenarioElement = null;

  for (const p of paragraphs) {
    if (p.textContent.includes('场景') && p.textContent.includes('Scenario')) {
      scenarioElement = p;
      break;
    }
  }

  const scenarioText = scenarioElement ? scenarioElement.textContent : '';
  const scenarioMatch = scenarioText.match(/场景\s*\([^)]+\):\s*([^(]+)\s*\(([^)]+)\)/);

  if (scenarioMatch) {
    const scenario = {
      title: {
        zh: scenarioMatch[1].trim(),
        en: scenarioMatch[2].trim()
      },
      dialogues: []
    };
    
    // 查找所有对话
    const dialogueTitles = Array.from(document.querySelectorAll('p')).filter(p => 
      p.textContent.includes('对话') && p.textContent.includes('Dialogue')
    );
    
    dialogueTitles.forEach((titleElem, index) => {
      const dialogueTitle = titleElem.textContent.match(/对话[^(]*\s*\(([^)]+)\):/);
      if (!dialogueTitle) return;
      
      const dialogue = {
        title: dialogueTitle[1].trim(),
        exchanges: []
      };
      
      // 找到对话内容（在下一个ul元素中）
      const ulElement = titleElem.nextElementSibling;
      if (ulElement && ulElement.tagName === 'UL') {
        // 获取所有对话行
        const liElements = Array.from(ulElement.querySelectorAll('li'));
        
        for (let i = 0; i < liElements.length; i++) {
          const li = liElements[i];
          
          // 检查是否是英文对话行（包含strong标签）
          if (li.querySelector('strong')) {
            const strongText = li.querySelector('strong').textContent;
            const speakerMatch = strongText.match(/([^:]+):(.*)/);
            
            if (speakerMatch) {
              const speaker = speakerMatch[1].trim();
              const text = speakerMatch[2].trim();
              
              const translationUl = li.querySelector('ul');
              const translation = translationUl ? translationUl.querySelector('li').textContent : '';
              const translationMatch = translation.match(/([^：]+)：(.*)/);
              const speaker_cn = translationMatch ? translationMatch[1].trim() : '';
              const translationText = translationMatch ? translationMatch[2].trim() : translation;
              dialogue.exchanges.push({
                speaker,
                speaker_cn, 
                text,
                translation: translationText
              });
            }
          }
        }
      }
      
      scenario.dialogues.push(dialogue);
    });
    
    result.scenarios.push(scenario);
  }

  return result;
}

// 导出函数
module.exports = {
  convertMarkdownToJson
};
