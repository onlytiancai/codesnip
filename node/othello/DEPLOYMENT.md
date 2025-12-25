# 黑白棋游戏部署文档

## 环境准备

### 服务器要求
- Linux 操作系统（推荐 Ubuntu 20.04+ 或 CentOS 7+）
- Node.js 16+ （推荐使用 nvm 管理 Node.js 版本）
- pnpm 包管理器
- Nginx 1.18+ 

### 安装依赖
```bash
# 安装 Node.js 和 npm（以 Ubuntu 为例）
sudo apt update
sudo apt install -y nodejs npm

# 安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
source ~/.bashrc

# 使用 nvm 安装 Node.js 18
nvm install 18
nvm use 18

# 安装 pnpm
npm install -g pnpm

# 安装 Nginx
sudo apt install -y nginx
```

## 项目部署

### 1. 克隆代码
```bash
git clone <仓库地址>
cd othello
```

### 2. 客户端构建

#### 安装依赖
```bash
cd client
pnpm install
```

#### 修改 WebSocket 连接地址
编辑 `client/index.html` 文件，将 WebSocket URL 从开发环境改为生产环境：

```html
<!-- 原配置 -->
<script>
  window.wsUrl = 'ws://localhost:3001';
</script>

<!-- 修改为生产环境配置 -->
<script>
  window.wsUrl = 'ws://example.com/ws';
</script>
```

#### 构建生产版本
```bash
pnpm run build
```

构建完成后，客户端文件将生成在 `client/dist/` 目录下。

### 3. 服务器配置

#### 安装依赖
```bash
cd ../server
pnpm install
```

#### 创建环境变量文件
创建 `.env` 文件来配置服务器端口（可选）：

```bash
echo "PORT=3001" > .env
```

#### 设置进程管理（使用 PM2）
```bash
# 安装 PM2
npm install -g pm2

# 启动服务器并设置开机自启
pm2 start src/index.js
pm2 startup
pm2 save
```

## Nginx 配置

### 创建 Nginx 配置文件
```bash
sudo nano /etc/nginx/sites-available/othello
```

### 配置文件内容
```nginx
server {
    listen 80;
    server_name example.com;

    # 客户端静态文件
    location / {
        root /path/to/othello/client/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # WebSocket 反向代理
    location /ws {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # 服务器 API（如果有）
    location /api {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 启用配置
```bash
sudo ln -s /etc/nginx/sites-available/othello /etc/nginx/sites-enabled/

# 测试配置是否正确
sudo nginx -t

# 重新加载 Nginx
sudo systemctl reload nginx
```

## 启动服务

### 启动服务器（直接方式）
```bash
cd /path/to/othello/server
node src/index.js
```

### 或使用 PM2 管理进程
```bash
# 启动服务器
pm2 start src/index.js

# 设置开机自启
pm2 startup
pm2 save

# 查看进程状态
pm2 status

# 重启服务器
pm2 restart src/index.js
```

## 验证部署

1. 打开浏览器访问：`http://example.com`
2. 检查游戏是否能正常加载
3. 测试游戏对战功能，确保 WebSocket 连接正常

## 常见问题排查

### 1. 客户端无法连接到 WebSocket 服务器
- 检查 `client/index.html` 中的 `wsUrl` 是否正确配置
- 检查 Nginx 配置中的 WebSocket 代理是否正确
- 检查服务器是否正在运行且监听正确的端口

### 2. 客户端静态文件无法加载
- 检查 Nginx 配置中的 `root` 路径是否正确
- 确保客户端已成功构建且文件存在于 `client/dist/` 目录

### 3. Nginx 启动失败
- 使用 `sudo nginx -t` 检查配置文件语法错误
- 检查端口 80 是否已被其他服务占用

### 4. 服务器无法启动
- 检查 Node.js 版本是否符合要求
- 检查依赖是否已正确安装
- 检查端口是否已被占用

## 更新部署

当需要更新代码时，执行以下步骤：

```bash
# 拉取最新代码
git pull

# 更新客户端
cd client
pnpm install
pnpm run build

# 更新服务器
cd ../server
pnpm install

# 重启服务器（直接方式）
node src/index.js

# 或使用 PM2 重启
pm2 restart src/index.js
```
