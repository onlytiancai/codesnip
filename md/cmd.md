/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/Users/huhao/chrome-profile-for-ai

netstat -anp tcp | grep LISTEN

ttyd -i "$(tailscale ip -4)" -p 7681  -W tmux new-session -A -s claude

node ~/src/codesnip/node/html-to-md/src/index.js -u https://hacker-podcast.agi.li/episode/2026-10-30|python -m rich.markdown -

python /Users/huhao/src/codesnip/python/ai/024-cet4/scripts/cet4.py
