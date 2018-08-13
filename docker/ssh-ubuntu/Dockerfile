# 基础镜像信息
FROM ubuntu:14.04

# 维护者信息
MAINTAINER onlytiancai onlytiancai@gmail.com

# 更新apt缓存、安装ssh服务
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd /root/.ssh
RUN sed -ir 's/^PermitRootLogin\s.*/PermitRootLogin yes/ig' /etc/ssh/sshd_config
RUN echo 'UseDNS no' >> /etc/ssh/sshd_config 
RUN echo "root:123456" | chpasswd


# Set the locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 
RUN echo 'LANG="en_US.UTF-8"' > /etc/default/locale
RUN echo 'LANGUAGE="en_US:en"' >> /etc/default/locale

# 配置免密要和自启动脚本
ADD run.sh /run.sh
RUN chmod 755 /run.sh

# 暴露22端口
EXPOSE 22

# 设置脚本自启动
CMD ["/run.sh"]
