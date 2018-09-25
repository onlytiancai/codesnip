# 基础镜像信息
FROM ubuntu:18.04

# 维护者信息
MAINTAINER onlytiancai onlytiancai@gmail.com

# 更新源
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
ADD aliyun.sources.list /etc/apt/sources.list

# 更新apt缓存、安装ssh服务
RUN apt-get update && apt-get install -y openssh-server vim
RUN mkdir -p /var/run/sshd /root/.ssh

# 配置 sshd 服务
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN echo 'UseDNS no' >> /etc/ssh/sshd_config
RUN echo "root:root" | chpasswd

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

    # 自启动脚本
    ADD run.sh /run.sh
    RUN chmod 755 /run.sh

    # 暴露22端口
    EXPOSE 22

    # 设置脚本自启动
    CMD ["/run.sh"]


