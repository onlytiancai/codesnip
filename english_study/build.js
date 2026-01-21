// build.js
import fs from 'fs';
import path from 'path';
import { marked } from 'marked';

const cwd = '.';

// 获取文件修改时间（不存在则返回 0）
function getMtime(file) {
  try {
    return fs.statSync(file).mtimeMs;
  } catch {
    return 0;
  }
}

// 找出所有 .md 文件
let mdFiles = fs.readdirSync(cwd)
  .filter(f => path.extname(f).toLowerCase() === '.md')
  .sort((a, b) => a.localeCompare(b, 'en'));   // 字典序排序

// 转换 md → html（包含增量检查）
mdFiles.forEach(file => {
  const name = file.replace(/\.md$/i, '');
  const htmlFile = `${name}.html`;

  const mdMtime = getMtime(file);
  const htmlMtime = getMtime(htmlFile);

  // 若 html 存在且未过期 → 跳过
  if (htmlMtime >= mdMtime) {
    console.log(`Skip (up-to-date): ${file}`);
    return;
  }

  // 重新生成 HTML
  const mdContent = fs.readFileSync(file, 'utf8');
  const bodyContent = marked(mdContent);

  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body { font-family: sans-serif; padding: 20px; }
    a { text-decoration: none; color: #0366d6; }
    a:hover { text-decoration: underline; }
    table {
      border-collapse: collapse;
      border: 1px solid #ccc;
    }

    th, td {
      border: 1px solid #ccc;
      padding: 5px;
    }
  </style>

  <title>${name}</title>
</head>
<body>
${bodyContent}
</body>
</html>`;

  fs.writeFileSync(htmlFile, htmlContent);

  // 将 html 的 mtime 设置为 md 的 mtime（同步时间戳）
  fs.utimesSync(htmlFile, mdMtime / 1000, mdMtime / 1000);

  console.log(`Converted: ${file} -> ${htmlFile}`);
});

// 生成 index.html（每次都更新）
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

