# 英语对话集静态网站

这是一个纯静态的英语对话集网站，不需要Node.js服务器，只需将public目录放在任何Web服务器的静态目录下即可运行。

## 准备工作

在部署静态网站之前，需要先生成所有必要的JSON文件：

1. 首先运行 `generate_index.js` 生成 `index.json`：

```
node generate_index.js
```

2. 然后运行 `generate_static_files.js` 生成所有静态JSON文件：

```
node generate_static_files.js
```

这将会：
- 在 `public/data` 目录下创建 `index.json` 的副本
- 为每个markdown文件生成对应的JSON文件，命名为 `dialogue-{number}.json`

## 部署

生成所有静态文件后，只需将 `public` 目录复制到任何Web服务器的静态目录下即可。

例如，使用Apache：
- 复制 `public` 目录到 `/var/www/html/english-dialogues`
- 访问 `http://your-server/english-dialogues`

或者使用Nginx：
- 复制 `public` 目录到 `/usr/share/nginx/html/english-dialogues`
- 访问 `http://your-server/english-dialogues`

## 网站功能

- **首页**：显示所有对话的列表，按主题分类
- **详情页**：显示特定对话的详细内容，点击英文可以显示/隐藏中文翻译

## 文件结构

```
public/
├── css/
│   └── style.css
├── js/
│   └── script.js
├── data/
│   ├── index.json
│   ├── dialogue-1.json
│   ├── dialogue-2.json
│   └── ...
├── index.html
└── detail.html
```