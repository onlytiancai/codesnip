const express = require('express');
const fs = require('fs');
const path = require('path');
const { convertMarkdownToJson } = require('./md_to_json_dom');
const app = express();
const port = 3000;

// 提供静态文件
app.use(express.static('public'));

// 读取index.json并提供API
app.get('/api/index', (req, res) => {
  try {
    const filePath = path.join(__dirname, 'index.json');
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      res.json(JSON.parse(content));
    } else {
      res.status(404).send('Index file not found');
    }
  } catch (error) {
    console.error('Error reading index file:', error);
    res.status(500).send('Error reading index file');
  }
});

// 获取特定对话的详细信息
app.get('/api/dialogue/:number', (req, res) => {
  try {
    const number = req.params.number;
    const filename = `${number}.md`;
    const jsonData = convertMarkdownToJson(filename);
    res.json(jsonData);
  } catch (error) {
    console.error('Error reading dialogue file:', error);
    res.status(500).send('Error reading dialogue file');
  }
});

// 首页路由
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 详情页路由
app.get('/detail/:number', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'detail.html'));
});

// 保留原有API路由
app.get('/api/dialogues', (req, res) => {
  try {
    const filePath = path.join(__dirname, 'english_dialogue.md');
    const content = fs.readFileSync(filePath, 'utf8');
    
    // 解析markdown内容
    const sections = [];
    let currentSection = null;
    let currentScenario = null;
    
    // 按行分割内容
    const lines = content.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // 匹配章节标题 (### **1. 工作 (Work)**)
      if (line.startsWith('### **')) {
        const titleMatch = line.match(/### \*\*\d+\.\s+([^(]+)\s*\(([^)]+)\)\*\*/);
        if (titleMatch) {
          currentSection = {
            title: titleMatch[1].trim(),
            titleEn: titleMatch[2].trim(),
            scenarios: []
          };
          sections.push(currentSection);
        }
      }
      // 匹配场景 (**场景 (Scenario): 安排会议 (Scheduling a Meeting)**)
      else if (line.startsWith('**场景') || line.startsWith('**Scenario')) {
        const scenarioMatch = line.match(/\*\*场景\s*\([^)]+\):\s*([^(]+)\s*\(([^)]+)\)\*\*/);
        if (scenarioMatch && currentSection) {
          currentScenario = {
            title: scenarioMatch[1].trim(),
            titleEn: scenarioMatch[2].trim(),
            dialogues: []
          };
          currentSection.scenarios.push(currentScenario);
        }
      }
      // 匹配对话 (**对话一 (Dialogue 1):**)
      else if (line.startsWith('**对话') || line.startsWith('**Dialogue')) {
        const dialogueMatch = line.match(/\*\*对话[^(]*\s*\(([^)]+)\):\*\*/);
        if (dialogueMatch && currentScenario) {
          const dialogue = {
            title: dialogueMatch[1].trim(),
            exchanges: []
          };
          
          // 收集对话内容
          let j = i + 1;
          while (j < lines.length && !lines[j].startsWith('**对话') && !lines[j].startsWith('**Dialogue') && !lines[j].startsWith('**场景') && !lines[j].startsWith('**Scenario') && !lines[j].startsWith('###') && !lines[j].startsWith('---')) {
            const line = lines[j].trim();
            if (line.startsWith('* **')) {
              const speakerMatch = line.match(/\* \*\*([^:]+):(.*)\*\*/);
              if (speakerMatch) {
                const speaker = speakerMatch[1].trim();
                const text = speakerMatch[2].trim();
                
                const exchange = {
                  speaker,
                  text,
                  translation: null
                };
                
                // 检查下一行是否为翻译
                if (j + 1 < lines.length && lines[j + 1].trim().startsWith('* ')) {
                  const translationMatch = lines[j + 1].trim().match(/\* ([^:]+):(.*)/);
                  if (translationMatch) {
                    exchange.translation = translationMatch[2].trim();
                    j++;
                  }
                }
                
                dialogue.exchanges.push(exchange);
              }
            }
            j++;
          }
          
          currentScenario.dialogues.push(dialogue);
        }
      }
    }
    
    res.json(sections);
  } catch (error) {
    console.error('Error reading markdown file:', error);
    res.status(500).send('Error reading dialogue file');
  }
});

// 保留原有API路由
app.get('/api/md-to-json/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    // 调用md_to_json_dom模块的函数
    const jsonData = convertMarkdownToJson(filename);
    res.json(jsonData);
  } catch (error) {
    console.error('Error converting markdown to JSON:', error);
    res.status(500).send('Error converting markdown to JSON');
  }
});

// 保留原有API路由
app.get('/api/generate-html/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    // 调用md_to_json_dom模块的函数
    const jsonData = convertMarkdownToJson(filename);
    
    // 根据JSON生成HTML内容
    let html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${jsonData.category.en} - ${jsonData.category.zh}</title>
  <link rel="stylesheet" href="/css/style.css">
</head>
<body>
  <div class="container">
    <h1>${jsonData.number}. ${jsonData.category.zh} (${jsonData.category.en})</h1>`;
    
    // 遍历所有场景
    jsonData.scenarios.forEach(scenario => {
      html += `
    <section class="scenario">
      <h2>场景 (Scenario): ${scenario.title.zh} (${scenario.title.en})</h2>`;
      
      // 遍历所有对话
      scenario.dialogues.forEach(dialogue => {
        html += `
      <div class="dialogue">
        <h3>${dialogue.title}</h3>
        <div class="exchanges">`;
        
        // 遍历所有对话内容
        dialogue.exchanges.forEach(exchange => {
          html += `
          <div class="exchange">
            <div class="english">
              <strong>${exchange.speaker}:</strong> ${exchange.text}
            </div>
            <div class="chinese">
              <strong>${exchange.speaker_cn}:</strong> ${exchange.translation}
            </div>
          </div>`;
        });
        
        html += `
        </div>
      </div>`;
      });
      
      html += `
    </section>`;
    });
    
    html += `
  </div>
  <script src="/js/script.js"></script>
</body>
</html>`;
    
    res.send(html);
  } catch (error) {
    console.error('Error generating HTML from markdown:', error);
    res.status(500).send('Error generating HTML content');
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});