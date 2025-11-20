const logEl = document.getElementById('log');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const changeNickBtn = document.getElementById('changeNick');
const clearBtn = document.getElementById('clearBtn');
const recordBtn = document.getElementById('recordBtn');
const toggleUsersBtn = document.getElementById('toggleUsers');
const usersPanel = document.getElementById('usersPanel');
const usersListEl = document.getElementById('usersList');

let nick = null;
let ws = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimeout = null;
let currentRoom = null;

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
  // 保持页面可能传入的 room 参数（?room=...），默认不传则在服务端使用 'default'
  const params = new URLSearchParams(loc.search);
  const roomParam = params.get('room');
  currentRoom = roomParam || null;
  const url = protocol + '://' + loc.host + (roomParam ? `/?room=${encodeURIComponent(roomParam)}` : '/');
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
  const roomPart = msg.room ? ` <span style="color:#999;font-size:12px">[room:${escapeHtml(msg.room)}]</span>` : '';
  appendLine(`<div class="meta"><strong>${nick}</strong>${ipPart}${roomPart} 已加入（你的昵称）</div>`);
  // 显示房间在状态栏
  if (msg.room) {
    statusEl.textContent = `已连接（房间 ${msg.room}）`;
    currentRoom = msg.room;
  }
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

    if (msg.type === 'audio') {
      // msg.data is a data URL (audio/webm or audio/ogg)
      const isSelf = msg.nick === nick;
      const d = document.createElement('div');
      d.className = 'msg ' + (isSelf ? 'self' : 'other');
      const t = new Date(msg.ts).toLocaleTimeString();
      const ipPart = msg.ip ? ` <span style="color:#999;font-size:12px">(${escapeHtml(msg.ip)})</span>` : '';
      d.innerHTML = `<div class="bubble"><div class="meta">[${t}] <strong>${escapeHtml(msg.nick)}</strong>${ipPart}</div><div class="text"><audio controls src="${escapeHtml(msg.data)}"></audio></div></div>`;
      logEl.appendChild(d);
      logEl.scrollTop = logEl.scrollHeight;
      return;
    }

    if (msg.type === 'nick') {
      // 显示昵称变更事件
  const ipPart = msg.ip ? ` <span style="color:#999;font-size:12px">(${escapeHtml(msg.ip)})</span>` : '';
  appendLine(`<div class="meta">🔁 <strong>${escapeHtml(msg.oldNick)}</strong> 改名为 <strong>${escapeHtml(msg.newNick)}</strong>${ipPart}</div>`);
      return;
    }
    
    if (msg.type === 'presence') {
      // msg.users = [{nick, ip}]
      renderUsers(msg.users || []);
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

// 生成简单随机 UUID v4（浏览器环境）
function generateUuid() {
  // 使用 crypto API 如果可用
  if (window.crypto && window.crypto.getRandomValues) {
    const buf = new Uint8Array(16);
    window.crypto.getRandomValues(buf);
    // set version bits
    buf[6] = (buf[6] & 0x0f) | 0x40;
    buf[8] = (buf[8] & 0x3f) | 0x80;
    const hex = Array.from(buf).map(b => b.toString(16).padStart(2, '0')).join('');
    return `${hex.substr(0,8)}-${hex.substr(8,4)}-${hex.substr(12,4)}-${hex.substr(16,4)}-${hex.substr(20,12)}`;
  }
  // 回退实现
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function reconnectToRoom(room) {
  // 更新地址栏但不刷新页面
  const url = new URL(window.location.href);
  if (room) url.searchParams.set('room', room);
  else url.searchParams.delete('room');
  history.pushState({}, '', url.toString());
  // 关闭现有连接并重连
  try { if (ws) ws.close(); } catch (e) {}
  // 小延时确保 close 触发
  setTimeout(connect, 200);
}

// 新建房间按钮
const newRoomBtn = document.getElementById('newRoom');
if (newRoomBtn) newRoomBtn.addEventListener('click', () => {
  const id = generateUuid();
  reconnectToRoom(id);
  appendLine(`<div class="meta">🆕 已创建房间 <strong>${escapeHtml(id)}</strong></div>`);
});

// 复制链接按钮
const copyLinkBtn = document.getElementById('copyLink');
if (copyLinkBtn) copyLinkBtn.addEventListener('click', async () => {
  const url = window.location.href;
  try {
    await navigator.clipboard.writeText(url);
    appendLine(`<div class="meta">🔗 已复制链接：${escapeHtml(url)}</div>`);
  } catch (e) {
    // 回退：使用临时输入
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.value = url;
    input.select();
    try { document.execCommand('copy'); appendLine('<div class="meta">🔗 链接已复制（回退方式）</div>'); } catch (ee) { appendLine('<div class="meta">⚠️ 复制失败，请手动复制链接</div>'); }
    document.body.removeChild(input);
  }
});

// 在线人员面板交互
if (toggleUsersBtn) {
  toggleUsersBtn.addEventListener('click', () => {
    if (!usersPanel) return;
    usersPanel.style.display = usersPanel.style.display === 'none' || usersPanel.style.display === '' ? 'block' : 'none';
    if (usersPanel.style.display === 'block' && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'request_presence' }));
    }
  });
}
const closeUsersBtn = document.getElementById('closeUsers');
if (closeUsersBtn) closeUsersBtn.addEventListener('click', () => { if (usersPanel) usersPanel.style.display = 'none'; });

function renderUsers(list) {
  if (!usersListEl) return;
  usersListEl.innerHTML = '';
  if (!Array.isArray(list) || list.length === 0) {
    usersListEl.innerHTML = '<div style="color:#777">暂无在线人员</div>';
    return;
  }
  list.forEach(u => {
    const div = document.createElement('div');
    div.style.padding = '6px 4px';
    div.style.borderBottom = '1px solid #f0f0f0';
    div.innerHTML = `<strong>${escapeHtml(u.nick || '')}</strong> <span style="color:#999;font-size:12px">${u.ip || ''}</span>`;
    usersListEl.appendChild(div);
  });
}

// 清屏按钮逻辑
if (clearBtn) {
  clearBtn.addEventListener('click', () => {
    // 清空日志内容并显示一条提示
    logEl.innerHTML = '';
    appendLine('<div class="meta">🧹 已清屏</div>');
  });
}

// 录音逻辑：按住录音，松开发送
if (recordBtn) {
  function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      appendLine('<div class="meta">⚠️ 浏览器不支持录音</div>');
      return;
    }
  recordBtn.textContent = '正在录音...';
    recordBtn.style.background = '#c33';
    audioChunks = [];
  // 记录录音开始时间，用于判断时长
  recordBtn._recordStart = Date.now();
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) audioChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const durationMs = Date.now() - (recordBtn._recordStart || 0);
        // 最短录音时长：1000ms
        const minMs = 1000;
        if (durationMs < minMs) {
          appendLine(`<div class="meta">⚠️ 录音过短（${Math.round(durationMs)}ms），需至少 ${minMs}ms</div>`);
          // 停止所有音轨以释放麦克风
          stream.getTracks().forEach(t => t.stop());
          return;
        }

        const blob = new Blob(audioChunks, { type: audioChunks[0] ? audioChunks[0].type : 'audio/webm' });
        const reader = new FileReader();
        reader.onload = () => {
          const dataUrl = reader.result;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'audio', data: dataUrl }));
            // 本地回显一条占位（服务端会广播给其他客户端）
            appendLine('<div class="meta">🎤 已发送语音</div>');
          } else {
            appendLine('<div class="meta">⚠️ 未连接，发送失败</div>');
          }
        };
        reader.readAsDataURL(blob);
        // 停止所有音轨以释放麦克风
        stream.getTracks().forEach(t => t.stop());
      };
      mediaRecorder.start();
    }).catch(err => {
      appendLine('<div class="meta">⚠️ 无法访问麦克风：' + escapeHtml(err.message || String(err)) + '</div>');
      recordBtn.textContent = '按住录音';
      recordBtn.style.background = '#f44';
    });
  }

  function stopRecordingAndSend() {
    recordBtn.textContent = '按住录音';
    recordBtn.style.background = '#f44';
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      try { mediaRecorder.stop(); } catch (e) {}
    }
  }

  // 鼠标/触摸事件支持
  recordBtn.addEventListener('mousedown', e => { e.preventDefault(); startRecording(); });
  recordBtn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
  window.addEventListener('mouseup', e => { if (mediaRecorder && mediaRecorder.state === 'recording') stopRecordingAndSend(); });
  window.addEventListener('touchend', e => { if (mediaRecorder && mediaRecorder.state === 'recording') stopRecordingAndSend(); });
}