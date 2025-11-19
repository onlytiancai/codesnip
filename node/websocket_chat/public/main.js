const logEl = document.getElementById('log');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const changeNickBtn = document.getElementById('changeNick');
const clearBtn = document.getElementById('clearBtn');

let nick = null;
let ws = null;

function appendLine(html) {
  const d = document.createElement('div');
  d.className = 'msg';
  d.innerHTML = html;
  logEl.appendChild(d);
  logEl.scrollTop = logEl.scrollHeight;
}

function appendMessage(msg, self) {
  const d = document.createElement('div');
  d.className = 'msg ' + (self ? 'self' : 'other');
  const t = new Date(msg.ts).toLocaleTimeString();
  const ipPart = msg.ip ? ` <span style="color:#999;font-size:12px">(${escapeHtml(msg.ip)})</span>` : '';
  d.innerHTML = `<div class="bubble"><div class="meta">[${t}] <strong>${escapeHtml(msg.nick)}</strong>${ipPart}</div><div class="text">${escapeHtml(msg.text)}</div></div>`;
  logEl.appendChild(d);
  logEl.scrollTop = logEl.scrollHeight;
}

function connect() {
  // 使用相同 origin 的 ws
  const loc = window.location;
  const protocol = loc.protocol === 'https:' ? 'wss' : 'ws';
  const url = protocol + '://' + loc.host;
  ws = new WebSocket(url);

  ws.addEventListener('open', () => {
    statusEl.textContent = '已连接';
  });

  ws.addEventListener('message', ev => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }

    if (msg.type === 'assign') {
      nick = msg.nick;
  const ipPart = msg.ip ? ` <span style="color:#999;font-size:12px">(${escapeHtml(msg.ip)})</span>` : '';
  appendLine(`<div class="meta"><strong>${nick}</strong>${ipPart} 已加入（你的昵称）</div>`);
      return;
    }

    if (msg.type === 'join') {
  const ipPart = msg.ip ? ` <span style="color:#999;font-size:12px">(${escapeHtml(msg.ip)})</span>` : '';
  appendLine(`<div class="meta">🔔 <strong>${escapeHtml(msg.nick)}</strong>${ipPart} 加入聊天室</div>`);
      return;
    }

    if (msg.type === 'leave') {
  const ipPart = msg.ip ? ` <span style="color:#999;font-size:12px">(${escapeHtml(msg.ip)})</span>` : '';
  appendLine(`<div class="meta">🔕 <strong>${escapeHtml(msg.nick)}</strong>${ipPart} 离开</div>`);
      return;
    }

    if (msg.type === 'message') {
      appendMessage(msg, msg.nick === nick);
      return;
    }

    if (msg.type === 'nick') {
      // 显示昵称变更事件
  const ipPart = msg.ip ? ` <span style="color:#999;font-size:12px">(${escapeHtml(msg.ip)})</span>` : '';
  appendLine(`<div class="meta">🔁 <strong>${escapeHtml(msg.oldNick)}</strong> 改名为 <strong>${escapeHtml(msg.newNick)}</strong>${ipPart}</div>`);
      return;
    }
  });

  ws.addEventListener('close', () => {
    statusEl.textContent = '已断开，正在重连...';
    // 简单重连策略
    setTimeout(connect, 1500);
  });

  ws.addEventListener('error', () => {
    statusEl.textContent = '出错';
  });
}

function escapeHtml(s){
  return s.replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

sendBtn.addEventListener('click', () => {
  const v = inputEl.value.trim();
  if (!v) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'message', text: v }));
    inputEl.value = '';
  }
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendBtn.click();
});

// 启动连接
connect();

// 修改昵称按钮逻辑
changeNickBtn.addEventListener('click', () => {
  const newNick = prompt('输入新的昵称（1-32 字符）', nick || '');
  if (!newNick) return;
  const n = newNick.trim().slice(0, 32);
  if (!n) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'nick', nick: n }));
  }
});

// 清屏按钮逻辑
if (clearBtn) {
  clearBtn.addEventListener('click', () => {
    // 清空日志内容并显示一条提示
    logEl.innerHTML = '';
    appendLine('<div class="meta">🧹 已清屏</div>');
  });
}