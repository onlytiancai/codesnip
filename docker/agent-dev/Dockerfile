# Dockerfile
FROM ubuntu:22.04

# 使用 root 权限（Ubuntu 默认就是 root）
USER root

# 安装一些基础工具（可选）
RUN apt-get update && apt-get install -y \
    vim curl wget git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 暴露端口（5000-6000）
EXPOSE 5000-6000

# 默认命令启动 bash
CMD ["/bin/bash"]
