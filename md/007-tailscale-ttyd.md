## 最简单的替代方案：Tailscale + tmux + ttyd

这里需要先说明：如果你当前的 `claude` 会话不是在 `tmux` 里面启动的，不能直接把它无损地转移到 tmux。为了保留现有会话，先别关闭当前终端。

### 第一步：确认 Mac 的 Tailscale IP

在 Mac 的另一个终端窗口运行：

```
tailscale ip -4
```

记下输出的 `100.x.x.x` 地址。

### 第二步：检查是否已安装 tmux 和 ttyd

在 Mac 终端运行：

```
brew --version
tmux -V
ttyd --version
```

如果后两个命令提示找不到，可以安装：

```
brew install tmux ttyd
```

### 第三步：决定如何接入当前会话

- 想保留当前 Claude Code 会话： 优先考虑给现有终端提供远程访问，先不要退出 Claude。
- 可以重新开一个会话： 可以在 tmux 中运行 Claude Code，再用 ttyd 提供网页终端。

新会话的基本示例：

```
tmux new -s claude
claude
```

然后在另一个终端窗口启动 Web 终端：

```
ttyd -i 100.100.100.100 -p 7681 tmux attach -t claude
```

这里的 `100.100.100.100` 是 Mac 本机的 Tailscale 接口地址，不是手机的地址。这个绑定方式限制服务仅监听该接口；访问时使用 Mac 的 Tailscale IP，例如 `http://100.x.x.x:7681`。

安全提醒： ttyd 会提供终端操作能力，可能间接控制 Claude Code 和你的文件。不要直接暴露到公网，也不要在没有认证的情况下开放给不可信的设备。

```
ttyd -i 100.80.12.34 -p 7681 -c 手机用户名:设置一个强密码 -W bash -l
```

参数说明：
- -i：只监听 Mac 的 Tailscale IP。
- -p 7681：网页终端端口。
- -c：启用 HTTP Basic 认证。
- -W：允许网页输入命令。
- bash -l：启动登录式 Bash 终端。

如果想让 Web 终端始终打开同一个 tmux 会话，可以让 ttyd 启动：

```
ttyd -i "$(tailscale ip -4)" -p 7681 -c 手机用户名:你的强密码 -W tmux new-session -A -s claude
```