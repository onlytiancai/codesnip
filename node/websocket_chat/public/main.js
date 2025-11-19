const logEl = document.getElementById('log');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const changeNickBtn = document.getElementById('changeNick');
const clearBtn = document.getElementById('clearBtn');
const recordBtn = document.getElementById('recordBtn');

let nick = null;
let ws = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimeout = null;

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
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) audioChunks.push(e.data); };
      mediaRecorder.onstop = () => {
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