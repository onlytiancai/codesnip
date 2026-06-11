编译镜像

    docker build -t agent-dev .

启动

    mkdir /Volumes/data/agent-data

    docker run -dit \
      --name agent-dev \
      -v /Volumes/data/agent-data:/app/data \
      -p 5000-6000:5000-6000 \
      agent-dev

进入

    docker exec -it agent-dev bash  

安装 node js

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    \. "$HOME/.nvm/nvm.sh"
    nvm install 24

安装 openclaw

    npm install -g openclaw@latest

提示

    https://bailian.console.aliyun.com/cn-beijing/?spm=5176.44390736.J_HUYtckZb0sVGMJi_5-NTF.2.d5aa33a68uWqhP&tab=doc#/doc/?type=model&url=3023085

    ◇  Security ─────────────────────────────────────────────────────────────────────────────────╮
    │                                                                                            │
    │  Security warning — please read.                                                           │
    │                                                                                            │
    │  OpenClaw is a hobby project and still in beta. Expect sharp edges.                        │
    │  By default, OpenClaw is a personal agent: one trusted operator boundary.                  │
    │  This bot can read files and run actions if tools are enabled.                             │
    │  A bad prompt can trick it into doing unsafe things.                                       │
    │                                                                                            │
    │  OpenClaw is not a hostile multi-tenant boundary by default.                               │
    │  If multiple users can message one tool-enabled agent, they share that delegated tool      │
    │  authority.                                                                                │
    │                                                                                            │
    │  If you’re not comfortable with security hardening and access control, don’t run           │
    │  OpenClaw.                                                                                 │
    │  Ask someone experienced to help before enabling tools or exposing it to the internet.     │
    │                                                                                            │
    │  Recommended baseline:                                                                     │
    │  - Pairing/allowlists + mention gating.                                                    │
    │  - Multi-user/shared inbox: split trust boundaries (separate gateway/credentials, ideally  │
    │    separate OS users/hosts).                                                               │
    │  - Sandbox + least-privilege tools.                                                        │
    │  - Shared inboxes: isolate DM sessions (`session.dmScope: per-channel-peer`) and keep      │
    │    tool access minimal.                                                                    │
    │  - Keep secrets out of the agent’s reachable filesystem.                                   │
    │  - Use the strongest available model for any bot with tools or untrusted inboxes.          │
    │                                                                                            │
    │  Run regularly:                                                                            │
    │  openclaw security audit --deep                                                            │
    │  openclaw security audit --fix                                                             │
    │                                                                                            │
    │  Must read: https://docs.openclaw.ai/gateway/security                                      │
    │                                                                                            │
    ├────────────────────────────────────────────────────────────────────────────────────────────╯
    ◇  QuickStart ─────────────────────────╮
    │                                      │
    │  Gateway port: 18789                 │
    │  Gateway bind: Loopback (127.0.0.1)  │
    │  Gateway auth: Token (default)       │
    │  Tailscale exposure: Off             │
    │  Direct to chat channels.            │
    │                                      │
    ├──────────────────────────────────────╯
