const fs = require('fs');
const path = require('path');
const { convertMarkdownToJson } = require('./md_to_json_dom');

// 读取所有markdown文件并生成index.json
function generateIndex() {
  const dialoguesDir = path.join(__dirname, 'dialogues');
  const outputPath = path.join(__dirname, 'index.json');
  
  // 获取所有markdown文件
  const files = fs.readdirSync(dialoguesDir)
    .filter(file => file.endsWith('.md'))
    .sort((a, b) => {
      // 按数字排序
      const numA = parseInt(a.replace('.md', ''));
      const numB = parseInt(b.replace('.md', ''));
      return numA - numB;
    });
  
  // 处理每个文件并提取信息
  const indexData = [];
  
  files.forEach(file => {
    try {
      const jsonData = convertMarkdownToJson(file);
      
      // 提取标题和场景信息
      const entry = {
        number: jsonData.number,
        title: jsonData.category,
        scenarios: jsonData.scenarios.map(scenario => ({
          title: scenario.title
        }))
      };
      
      indexData.push(entry);
    } catch (error) {
      console.error(`处理文件 ${file} 时出错:`, error);
    }
  });
  
  // 写入index.json
  fs.writeFileSync(outputPath, JSON.stringify(indexData, null, 2), 'utf8');
  console.log(`成功生成 index.json，包含 ${indexData.length} 个条目`);
}

// 执行生成
generateIndex();