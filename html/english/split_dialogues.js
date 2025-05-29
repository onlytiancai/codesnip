const fs = require('fs');
const path = require('path');

// 创建dialogues目录（如果不存在）
const dialoguesDir = path.join(__dirname, 'dialogues');
if (!fs.existsSync(dialoguesDir)) {
  fs.mkdirSync(dialoguesDir);
  console.log(`Created directory: ${dialoguesDir}`);
}

// 读取markdown文件
const filePath = path.join(__dirname, 'english_dialogue.md');
const content = fs.readFileSync(filePath, 'utf8');

// 按三级标题分割内容
const sections = content.split(/(?=### \*\*\d+\.)/);

// 处理每个部分并保存到单独的文件
let count = 0;
sections.forEach(section => {
  if (!section.trim()) return;
  
  // 提取序号
  const match = section.match(/### \*\*(\d+)\./);
  if (match) {
    const sectionNumber = match[1];
    const outputPath = path.join(dialoguesDir, `${sectionNumber}.md`);
    
    // 写入文件
    fs.writeFileSync(outputPath, section.trim());
    console.log(`Created file: ${outputPath}`);
    count++;
  }
});

console.log(`Splitting complete! Created ${count} files.`);