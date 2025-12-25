<template>
  <div class="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
    <h1 class="text-4xl font-bold text-gray-800 mb-4">奥赛罗棋</h1>
    
    <!-- 统计信息 -->
    <div v-if="!isInRoom" class="mb-6 text-center">
      <p class="text-gray-600 mb-2">当前房间数: {{ stats.roomCount }}</p>
      <p class="text-gray-600">当前在线人数: {{ stats.onlinePlayerCount }}</p>
    </div>
    

    
    <!-- 房间创建/加入界面 -->
    <div v-if="!isInRoom && !showNameInput" class="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
      <div class="mb-6">
        <button 
          @click="createRoom" 
          class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
        >
          创建新房间
        </button>
      </div>
      
      <div class="border-t border-gray-200 pt-6">
        <h2 class="text-xl font-semibold mb-4">加入房间</h2>
        <div class="flex gap-3">
          <input
            v-model="roomId"
            type="text"
            placeholder="房间ID"
            class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            @click="joinRoom"
            class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300"
          >
            加入
          </button>
        </div>
      </div>
    </div>
    
    <!-- 房间介绍和昵称输入界面 -->
    <div v-if="showNameInput" class="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
      <h2 class="text-2xl font-bold text-gray-800 mb-4">加入游戏</h2>
      <div class="bg-blue-50 p-4 rounded-lg mb-6">
        <h3 class="font-semibold text-gray-800 mb-2">房间信息</h3>
        <p class="text-gray-600">房间ID: <span class="font-mono bg-white px-2 py-1 rounded">{{ pendingRoomId }}</span></p>
        <p class="text-gray-600 mt-2">欢迎加入奥赛罗棋游戏！请输入您的昵称开始游戏。</p>
      </div>
      

      <div class="mb-4">
        <label for="playerName" class="block text-sm font-medium text-gray-700 mb-1">昵称</label>
        <input
          id="playerName"
          v-model="playerName"
          type="text"
          placeholder="请输入您的昵称（2-10个字符）"
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          maxlength="10"
        />
      </div>
      <div class="flex gap-4">
        <button
          @click="confirmJoinRoom"
          class="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
          :disabled="!playerName.trim()"
        >
          开始游戏
        </button>
        <button
          @click="cancelJoinRoom"
          class="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-3 px-4 rounded-lg transition duration-300"
        >
          取消
        </button>
      </div>
    </div>
    
    <!-- 通知提示 -->
    <div v-if="notification" :class="['fixed bottom-8 left-1/2 transform -translate-x-1/2 p-6 rounded-lg shadow-2xl transition-all duration-500 z-50 opacity-100 bg-white border-2', notification.type === 'join' ? 'border-green-500 bg-green-100' : 'border-red-500 bg-red-100']">
      <p class="font-bold text-lg text-gray-900">{{ notification.message }}</p>
    </div>
    
    <!-- 游戏界面 -->
    <div v-if="isInRoom" class="w-full max-w-4xl">
      
      <div class="bg-white p-6 rounded-lg shadow-lg mb-6">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4">
          <div>
            <h2 class="text-2xl font-bold">房间ID: {{ currentRoomId }}</h2>
            <p class="text-gray-600">当前回合: {{ currentPlayer === 1 ? '黑棋' : '白棋' }}</p>
            <p class="text-gray-500 text-sm" v-if="!gameOver">游戏状态: {{ roomInfo?.gameStarted ? '已开始' : '等待玩家加入' }}</p>
            <p class="text-gray-500 text-sm" v-if="isSpectator">您是: 游客</p>
            <p class="text-gray-500 text-sm" v-else-if="playerColor">您是: {{ playerColor === 1 ? '黑方' : '白方' }}</p>
            <p class="text-yellow-500 text-sm" :class="{ 'invisible': gameOver || currentPlayer === playerColor }">
              等待对方落子: {{ waitTime }}秒
            </p>
            <!-- 房间邀请URL -->
            <div class="mt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
              <input 
                type="text" 
                :value="roomUrl" 
                readonly 
                class="flex-1 px-3 py-2 text-sm border border-gray-300 rounded bg-gray-100"
              />
              <button 
                @click="copyRoomUrl" 
                class="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition"
              >
                复制邀请链接
              </button>
            </div>
          </div>
          <div class="flex flex-wrap items-center gap-3 sm:gap-6">
            <div class="text-center">
              <p class="font-semibold">黑棋</p>
              <p class="text-xs sm:text-sm text-gray-600">{{ getPlayerName(1) }}</p>
              <p class="text-xl sm:text-2xl font-bold text-gray-800">{{ scores.black }}</p>
            </div>
            <div class="text-center">
              <p class="font-semibold">白棋</p>
              <p class="text-xs sm:text-sm text-gray-600">{{ getPlayerName(2) }}</p>
              <p class="text-xl sm:text-2xl font-bold text-gray-800">{{ scores.white }}</p>
            </div>
            <div class="mt-2 sm:mt-0">
              <button 
                @click="leaveRoom" 
                class="w-full sm:w-auto bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300"
              >
                退出房间
              </button>
            </div>
          </div>
        </div>
        
        <!-- 棋盘组件 -->
        <Board 
          :board="board" 
          :currentPlayer="currentPlayer" 
          :validMoves="validMoves" 
          :playerColor="playerColor"
          :gameOver="gameOver"
          :flippedPieces="flippedPieces"
          @makeMove="makeMove"
        />
        
        <!-- 游戏结束信息 -->
        <div v-if="gameOver" class="mt-6 text-center">
          <h3 class="text-2xl font-bold mb-2">游戏结束</h3>
          <p class="text-xl" :class="winner === playerColor ? 'text-green-600' : winner ? 'text-red-600' : 'text-gray-600'">
            {{ winner ? (winner === playerColor ? '你赢了！' : '你输了！') : '平局！' }}
          </p>
          <button
            @click="restartGame"
            class="mt-4 px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
          >
            重新开始游戏
          </button>
        </div>
        
        <!-- 聊天系统 -->
        <div class="mt-6">
          <h3 class="text-lg font-semibold mb-3">聊天</h3>
          <!-- 聊天消息列表 -->
          <div 
            id="chat-messages"
            class="h-40 overflow-y-auto p-3 bg-gray-50 rounded-lg border border-gray-200 mb-3"
          >
            <div 
              v-for="(msg, index) in chatMessages" 
              :key="index"
              class="mb-2 p-2 rounded"
              :class="msg.playerColor === playerColor ? 'bg-blue-50' : 'bg-gray-100'"
            >
              <div class="flex items-center gap-2 text-sm">
                <span 
                  class="font-medium"
                  :class="msg.playerColor === 1 ? 'text-gray-800' : msg.playerColor === 2 ? 'text-gray-800' : 'text-gray-600'"
                >
                  {{ msg.playerName }} ({{ msg.playerColor === 1 ? '黑' : msg.playerColor === 2 ? '白' : '游客' }})
                </span>
                <span class="text-xs text-gray-500">
                  {{ new Date(msg.timestamp).toLocaleTimeString() }}
                </span>
              </div>
              <div class="text-sm mt-1">
                {{ msg.message }}
              </div>
            </div>
            <div v-if="chatMessages.length === 0" class="text-center text-gray-500 py-8">
              暂无聊天记录
            </div>
          </div>
          <!-- 常用语发送按钮 -->
          <div class="flex flex-wrap gap-2 justify-start">
            <button
              v-for="(msg, index) in commonMessages"
              :key="index"
              @click="sendChat(msg)"
              class="px-4 py-2 text-sm bg-gray-200 hover:bg-gray-300 text-gray-800 rounded transition whitespace-nowrap"
            >
              {{ msg }}
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 游戏规则介绍 -->
    <div v-if="!isInRoom" class="mt-10 bg-white p-6 rounded-lg shadow-lg w-full max-w-md">
      <div class="flex justify-between items-center cursor-pointer" @click="showRules = !showRules">
        <h2 class="text-xl font-semibold text-gray-800">游戏规则</h2>
        <svg :class="{'transform rotate-180': showRules}" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
      <div v-if="showRules" class="mt-4 text-gray-700 space-y-3">
        <p>1. 游戏在8x8的棋盘上进行，黑白双方交替落子</p>
        <p>2. 每一步必须翻转至少一个对方的棋子（夹在自己的棋子之间）</p>
        <p>3. 如果某一方无法落子，则跳过该回合</p>
        <p>4. 当双方都无法落子或棋盘填满时，游戏结束</p>
        <p>5. 棋盘上棋子数量多的一方获胜</p>
        <p>6. 黑方先行，初始时棋盘中心放置2黑2白四个棋子</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';
import Board from './components/Board.vue';

// 游戏状态
const isInRoom = ref(false);
const roomId = ref('');
const currentRoomId = ref('');
const playerColor = ref<number | null>(null);
const isSpectator = ref(false); // 添加游客标识
// 初始化8x8空棋盘
const board = ref<number[][]>(Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => 0)));
const currentPlayer = ref(1);
const scores = ref({ black: 2, white: 2 });
const gameOver = ref(false);
const winner = ref<number | null>(null);
const validMoves = ref<{ row: number; col: number }[]>([]);
const roomInfo = ref<any>(null);
const notification = ref<{ message: string; type: 'join' | 'leave' } | null>(null);
// 翻转的棋子，用于动画效果
const flippedPieces = ref<{ row: number; col: number }[]>([]);

// 昵称输入界面控制
const showNameInput = ref(false);
const pendingRoomId = ref('');
const playerName = ref('');

// 聊天系统
const chatMessages = ref<{ playerName: string; playerColor: number | null; message: string; timestamp: number }[]>([]);
const commonMessages = [
  '快点下呀',
  '下的不错',
  '不好意思，我有事要离开',
  '好的',
  '请稍等',
  '这一步很妙',
  '我想想',
  '加油'
];

// 统计信息
const stats = ref({ roomCount: 0, onlinePlayerCount: 0 });

// 游戏规则显示控制
const showRules = ref(true);
const showRulesInJoin = ref(true);

// 计时器
const waitTime = ref(0);
let waitTimer: number | null = null;

// WebSocket连接
const ws = ref<WebSocket | null>(null);
// 使用全局配置的WebSocket URL
const wsUrl = window.wsUrl || 'ws://localhost:3001';

// 创建WebSocket连接
const connectWebSocket = () => {
  ws.value = new WebSocket(wsUrl);
  
  ws.value.onopen = () => {
    console.log('WebSocket connected');
    // WebSocket连接建立后立即请求统计信息
    requestStats();
    
    // 解析URL参数中的roomId，在连接建立后自动加入房间
    const urlParams = new URLSearchParams(window.location.search);
    const roomIdFromUrl = urlParams.get('roomId');
    if (roomIdFromUrl) {
      pendingRoomId.value = roomIdFromUrl;
      showNameInput.value = true;
    }
  };
  
  ws.value.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleWebSocketMessage(message);
  };
  
  ws.value.onclose = () => {
    console.log('WebSocket disconnected');
    // 尝试重连
    setTimeout(connectWebSocket, 3000);
  };
  
  ws.value.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
};

// 处理WebSocket消息
const handleWebSocketMessage = (message: any) => {
  console.log('Received WebSocket message:', message);
  switch (message.type) {
    case 'ROOM_CREATED':
      console.log('Room created:', message.payload.roomId);
      pendingRoomId.value = message.payload.roomId;
      showNameInput.value = true;
      break;
      
    case 'JOINED_ROOM':
      console.log('Joined room:', message.payload.roomId);
      isInRoom.value = true;
      currentRoomId.value = message.payload.roomId;
      playerColor.value = message.payload.playerColor;
      isSpectator.value = playerColor.value === null; // 设置游客标识
      const room = message.payload.room;
      roomInfo.value = room;
      updateGameState(room.gameState);
      // 更新房间URL
      roomUrl.value = `${window.location.origin}${window.location.pathname}?roomId=${currentRoomId.value}`;
      break;
      
    case 'ROOM_UPDATED':
      console.log('Room updated:', message.payload.room.id);
      const updatedRoom = message.payload.room;
      roomInfo.value = updatedRoom;
      updateGameState(updatedRoom.gameState);
      break;
      
    case 'MOVE_MADE':
      console.log('Move made:', message.payload);
      if (message.payload.success) {
        // 记录翻转的棋子
        if (message.payload.flipped) {
          flippedPieces.value = message.payload.flipped;
          // 300ms后清除翻转状态（与动画时间匹配）
          setTimeout(() => {
            flippedPieces.value = [];
          }, 300);
        }
        updateGameState({
          board: message.payload.board,
          currentPlayer: message.payload.currentPlayer,
          scores: message.payload.scores,
          gameOver: message.payload.gameOver,
          winner: message.payload.winner,
          validMoves: [] // 会在ROOM_UPDATED中获取
        });
      }
      break;
      
    case 'ROOM_CLOSED':
      console.log('Room closed:', message.payload.message);
      showNotification(message.payload.message, 'leave');
      isInRoom.value = false;
      currentRoomId.value = '';
      playerColor.value = null;
      stopWaitTimer();
      break;
      
    case 'STATS':
      console.log('Received stats:', message.payload);
      stats.value = message.payload;
      break;
      
    case 'PLAYER_JOINED':
      console.log('Player joined:', message.payload);
      const joinMessage = message.payload.isSpectator 
        ? `${message.payload.playerName} 作为游客加入了房间`
        : `${message.payload.playerName} (${message.payload.playerColor === 1 ? '黑方' : '白方'}) 加入了房间`;
      showNotification(joinMessage, 'join');
      break;
      
    case 'PLAYER_LEFT':
      console.log('Player left:', message.payload);
      const leaveMessage = `${message.payload.playerName} (${message.payload.playerColor === 1 ? '黑方' : '白方'}) 离开了房间`;
      showNotification(leaveMessage, 'leave');
      // 将离开通知添加到聊天历史
      chatMessages.value.push({
        playerName: '系统',
        playerColor: 0, // 使用0表示系统消息
        message: leaveMessage,
        timestamp: Date.now()
      });
      // 保持最新消息可见
      setTimeout(() => {
        const chatContainer = document.getElementById('chat-messages');
        if (chatContainer) {
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
      }, 100);
      break;
      
    case 'ERROR':
      console.error('WebSocket error:', message.payload.message);
      // 添加详细的调试信息
      
      showNotification(message.payload.message, 'leave');
      
      break;
      
    case 'CHAT_MESSAGE':
      console.log('Chat message received:', message.payload);
      chatMessages.value.push(message.payload);
      // 显示聊天消息通知
      showNotification(`${message.payload.playerName}: ${message.payload.message}`, 'join');
      // 保持最新消息可见
      setTimeout(() => {
        const chatContainer = document.getElementById('chat-messages');
        if (chatContainer) {
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
      }, 100);
      break;
      
    case 'GAME_RESET':
      console.log('Game reset:', message.payload);
      const resetRoom = message.payload.room;
      roomInfo.value = resetRoom;
      updateGameState(resetRoom.gameState);
      showNotification('游戏已重新开始！', 'join');
      break;
  }
};

// 发送聊天消息
const sendChat = (message: string) => {
  if (!message.trim()) return;
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'SEND_CHAT',
      payload: {
        roomId: currentRoomId.value,
        message: message
      }
    }));
  }
};

// 更新游戏状态
const updateGameState = (gameState: any) => {
  board.value = gameState.board;
  const previousPlayer = currentPlayer.value;
  currentPlayer.value = gameState.currentPlayer;
  scores.value = gameState.scores;
  gameOver.value = gameState.gameOver;
  winner.value = gameState.winner;
  validMoves.value = gameState.validMoves || [];

  // 控制计时器
  if (gameOver.value) {
    stopWaitTimer();
  } else if (playerColor.value && currentPlayer.value !== playerColor.value) {
    // 对方回合，启动计时器
    startWaitTimer();
  } else {
    // 自己回合，停止计时器
    stopWaitTimer();
  }
};

// 启动等待计时器
const startWaitTimer = () => {
  stopWaitTimer();
  waitTime.value = 0;
  waitTimer = window.setInterval(() => {
    waitTime.value++;
  }, 1000);
};

// 停止等待计时器
const stopWaitTimer = () => {
  if (waitTimer) {
    clearInterval(waitTimer);
    waitTimer = null;
  }
  waitTime.value = 0;
};

// 显示通知提示
const showNotification = (message: string, type: 'join' | 'leave') => {
  notification.value = { message, type };
  // 3秒后自动隐藏通知
  setTimeout(() => {
    notification.value = null;
  }, 3000);
};

// 创建房间
const createRoom = () => {
  console.log('Creating new room');
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'CREATE_ROOM',
      payload: {}
    }));
  }
};

// 加入房间
const joinRoom = () => {
  if (!roomId.value.trim()) {
    showNotification('请输入房间ID', 'leave');
    return;
  }
  
  pendingRoomId.value = roomId.value;
  showNameInput.value = true;
};

// 确认加入房间
const confirmJoinRoom = () => {
  if (!playerName.value.trim()) {
    showNotification('请输入昵称', 'leave');
    return;
  }
  
  console.log('Joining room:', pendingRoomId.value, 'as', playerName.value);
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'JOIN_ROOM',
      payload: {
        roomId: pendingRoomId.value,
        playerName: playerName.value
      }
    }));
    
    // 重置状态
    playerName.value = '';
    showNameInput.value = false;
  }
};

// 取消加入房间
const cancelJoinRoom = () => {
  showNameInput.value = false;
  playerName.value = '';
  // 清除URL中的roomId参数，返回到首页
  const url = new URL(window.location.href);
  url.searchParams.delete('roomId');
  window.history.replaceState({}, '', url);
};

// 退出房间
const leaveRoom = () => {
  console.log('Leaving room:', currentRoomId.value);
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'LEAVE_ROOM',
      payload: {
        roomId: currentRoomId.value
      }
    }));
  }
  // 停止计时器
  stopWaitTimer();
  // 重置游戏状态
  isInRoom.value = false;
  currentRoomId.value = '';
  playerColor.value = null;
  roomInfo.value = null;
  // 清空聊天记录
  chatMessages.value = [];
  // 重置游戏板和分数
  board.value = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => 0));
  scores.value = { black: 2, white: 2 };
  gameOver.value = false;
  winner.value = null;
  validMoves.value = [];
  // 清除URL中的roomId参数，返回到首页
  const url = new URL(window.location.href);
  url.searchParams.delete('roomId');
  window.history.replaceState({}, '', url);
};

// 根据颜色获取玩家昵称
const getPlayerName = (color: number) => {
  if (!roomInfo.value?.players) return '未加入';
  const player = roomInfo.value.players.find((p: any) => p.color === color);
  return player ? player.name : '未加入';
};

// 生成房间邀请URL
const roomUrl = ref('');
onMounted(() => {
  // 设置初始URL
  roomUrl.value = `${window.location.origin}${window.location.pathname}?roomId=${currentRoomId.value}`;
});

// 复制房间URL到剪贴板
const copyRoomUrl = () => {
  navigator.clipboard.writeText(roomUrl.value)
    .then(() => {
      showNotification('邀请链接已复制到剪贴板！', 'join');
    })
    .catch(err => {
      console.error('复制失败:', err);
      showNotification('复制失败，请手动复制！', 'leave');
    });
};

// 重新开始游戏
const restartGame = () => {
  console.log('restartGame function called');
  console.log('WebSocket state:', ws.value?.readyState);
  console.log('Current room ID:', currentRoomId.value);
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    const message = JSON.stringify({
      type: 'RESTART_GAME',
      payload: {
        roomId: currentRoomId.value
      }
    });
    console.log('Sending restart game message:', message);
    ws.value.send(message);
  } else {
    console.error('WebSocket not open or not initialized');
  }
};

// 落子
const makeMove = (row: number, col: number) => {
  if (gameOver.value || playerColor.value !== currentPlayer.value) {
    return;
  }
  
  console.log('Making move at:', row, col);
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'MAKE_MOVE',
      payload: {
        roomId: currentRoomId.value,
        row: row,
        col: col
      }
    }));
  }
};

// 请求统计信息的函数
const requestStats = () => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN && !isInRoom.value) {
    ws.value.send(JSON.stringify({ type: 'GET_STATS', payload: {} }));
  }
};

// 生命周期钩子
onMounted(() => {
  connectWebSocket();
  
  // 每30秒更新一次统计信息
  const statsInterval = setInterval(() => {
    requestStats();
  }, 30000);

  onBeforeUnmount(() => {
    clearInterval(statsInterval);
  });
});

onBeforeUnmount(() => {
  if (ws.value) {
    ws.value.close();
  }
  stopWaitTimer();
});
</script>