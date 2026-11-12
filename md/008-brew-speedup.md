```bash
export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808

HOMEBREW_NO_AUTO_UPDATE=1 brew install tmux ttyd

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```