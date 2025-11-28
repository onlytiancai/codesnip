// build.js
import fs from 'fs';
import path from 'path';
import { marked } from 'marked';

const cwd = '.';

// 找出所有 .md 文件
let mdFiles = fs.readdirSync(cwd)
  .filter(f => path.extname(f).toLowerCase() === '.md')
  .sort((a, b) => a.localeCompare(b, 'en'));   // 字典序排序

// 转换 md → html
mdFiles.forEach(file => {
  const name = file.replace(/\.md$/i, '');
  const mdContent = fs.readFileSync(file, 'utf8');
  const bodyContent = marked(mdContent);

  // 包裹在完整 HTML 中并加入 charset
  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${name}</title>
</head>
<body>
${bodyContent}
</body>
</html>`;

  const htmlFile = `${name}.html`;
  fs.writeFileSync(htmlFile, htmlContent);

  console.log(`Converted: ${file} -> ${htmlFile}`);
});

// 生成 index.html
const links = mdFiles.map(md => {
  const name = md.replace(/\.md$/i, '');
  const html = `${name}.html`;
  return `<li><a href="${html}">${name}</a></li>`;
}).join('\n');

const indexHtml = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Index</title>
  <style>
    body { font-family: sans-serif; padding: 20px; }
    a { text-decoration: none; color: #0366d6; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Markdown Pages</h1>
  <ul>
    ${links}
  </ul>
</body>
</html>
`;

fs.writeFileSync('index.html', indexHtml);
console.log(`Generated index.html`);
