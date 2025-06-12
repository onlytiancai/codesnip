const fs = require('fs');
const path = require('path');
const { convertMarkdownToJson } = require('./md_to_json_dom');

// 创建数据目录
const dataDir = path.join(__dirname, 'public', 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

// 复制index.json到public/data目录
function copyIndexJson() {
  const sourcePath = path.join(__dirname, 'index.json');
  const destPath = path.join(dataDir, 'index.json');
  
  if (fs.existsSync(sourcePath)) {
    fs.copyFileSync(sourcePath, destPath);
    console.log('已复制 index.json 到 public/data 目录');
  } else {
    console.error('index.json 文件不存在，请先运行 generate_index.js');
    process.exit(1);
  }
}

// 为每个markdown文件生成对应的JSON文件
function generateDialogueJsonFiles() {
  const dialoguesDir = path.join(__dirname, 'dialogues');
  const files = fs.readdirSync(dialoguesDir)
    .filter(file => file.endsWith('.md'))
    .sort((a, b) => {
      const numA = parseInt(a.replace('.md', ''));
      const numB = parseInt(b.replace('.md', ''));
      return numA - numB;
    });
  
  console.log(`找到 ${files.length} 个markdown文件，开始生成JSON...`);
  
  files.forEach(file => {
    try {
      const number = file.replace('.md', '');
      const jsonData = convertMarkdownToJson(file);
      const outputPath = path.join(dataDir, `dialogue-${number}.json`);
      
      fs.writeFileSync(outputPath, JSON.stringify(jsonData, null, 2), 'utf8');
      console.log(`已生成 dialogue-${number}.json`);
    } catch (error) {
      console.error(`处理文件 ${file} 时出错:`, error);
    }
  });
}

// 执行生成
copyIndexJson();
generateDialogueJsonFiles();

console.log('所有静态JSON文件生成完成！');