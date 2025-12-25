<template>
  <div class="min-h-screen bg-gray-100">
    <!-- 顶部标题栏 -->
    <header v-if="!isInRoom" class="bg-white shadow-sm border-b border-gray-200">
      <div class="container mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
          <h1 class="text-2xl md:text-3xl font-bold text-gray-800 ">奥赛罗棋</h1>
          <div class="text-sm text-gray-600">
            <span class="mr-4">房间: {{ stats.roomCount }}</span>
            <span>在线: {{ stats.onlinePlayerCount }}</span>
          </div>
        </div>
      </div>
    </header>

    <!-- 等待对方落子计时器 - 固定在顶部 -->
    <div v-if="isInRoom && !gameOver && !opponentOffline && playerColor && currentPlayer !== playerColor" 
         class="fixed top-0 left-1/2 transform -translate-x-1/2 z-50 bg-yellow-100 border border-yellow-400 text-yellow-800 px-4 py-2 rounded-lg shadow-lg max-w-sm">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
        <span class="font-medium">等待对方落子: {{ waitTime }}秒</span>
      </div>
    </div>
    
    <!-- 房间创建/加入界面 - 响应式布局 -->
    <div v-if="!isInRoom && !showNameInput" class="container mx-auto px-4 py-8">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <!-- 左侧：创建/加入房间 -->
        <div class="lg:col-span-2">
          <div class="bg-white p-6 md:p-8 rounded-lg shadow-lg">
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
        </div>
        
        <!-- 右侧：游戏规则 -->
        <div class="lg:col-span-1">
          <div class="bg-white p-6 rounded-lg shadow-lg">
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
      </div>
    </div>
    
    <!-- 房间介绍和昵称输入界面 - 响应式布局 -->
    <div v-if="showNameInput" class="container mx-auto px-4 py-8">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <!-- 左侧：房间信息和昵称输入 -->
        <div class="lg:col-span-2">
          <div class="bg-white p-6 md:p-8 rounded-lg shadow-lg">
            <h2 class="text-2xl font-bold text-gray-800 mb-6">加入游戏</h2>
            <div class="bg-blue-50 p-4 rounded-lg mb-6">
              <h3 class="font-semibold text-gray-800 mb-2">房间信息</h3>
              <p class="text-gray-600">房间ID: <span class="font-mono bg-white px-2 py-1 rounded">{{ pendingRoomId }}</span></p>
              <p class="text-gray-600 mt-2">欢迎加入奥赛罗棋游戏！请输入您的昵称开始游戏。</p>
            </div>
            
            <div class="mb-6">
              <label for="playerName" class="block text-sm font-medium text-gray-700 mb-2">昵称</label>
              <input
                id="playerName"
                v-model="playerName"
                type="text"
                placeholder="请输入您的昵称（2-10个字符）"
                class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
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
        </div>
        
        <!-- 右侧：游戏规则 -->
        <div class="lg:col-span-1">
          <div class="bg-white p-6 rounded-lg shadow-lg">
            <div class="flex justify-between items-center cursor-pointer" @click="showRulesInJoin = !showRulesInJoin">
              <h2 class="text-xl font-semibold text-gray-800">游戏规则</h2>
              <svg :class="{'transform rotate-180': showRulesInJoin}" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div v-if="showRulesInJoin" class="mt-4 text-gray-700 space-y-3">
              <p>1. 游戏在8x8的棋盘上进行，黑白双方交替落子</p>
              <p>2. 每一步必须翻转至少一个对方的棋子（夹在自己的棋子之间）</p>
              <p>3. 如果某一方无法落子，则跳过该回合</p>
              <p>4. 当双方都无法落子或棋盘填满时，游戏结束</p>
              <p>5. 棋盘上棋子数量多的一方获胜</p>
              <p>6. 黑方先行，初始时棋盘中心放置2黑2白四个棋子</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 通知提示 -->
    <div v-if="notification" :class="['fixed bottom-8 left-1/2 transform -translate-x-1/2 p-6 rounded-lg shadow-2xl transition-all duration-500 z-50 opacity-100 bg-white border-2', notification.type === 'join' ? 'border-green-500 bg-green-100' : 'border-red-500 bg-red-100']">
      <p class="font-bold text-lg text-gray-900">{{ notification.message }}</p>
    </div>
    
    <!-- 游戏界面 - 响应式布局 -->
    <div v-if="isInRoom" class="container mx-auto px-4 py-4 lg:py-8">
      <!-- 窄屏布局：垂直堆叠 -->
      <div class="lg:hidden space-y-4">
        <!-- 比分显示 - 紧凑一行 -->
        <div class="bg-white p-3 rounded-lg shadow-lg">
          <div class="flex justify-between items-center">
            <div class="text-center">
              <div class="font-semibold text-sm">黑棋 {{ getPlayerName(1) }}</div>
              <div class="text-xl font-bold text-gray-800">{{ scores.black }}</div>
            </div>
            <div class="text-2xl font-bold text-gray-600">VS</div>
            <div class="text-center">
              <div class="font-semibold text-sm">白棋 {{ getPlayerName(2) }}</div>
              <div class="text-xl font-bold text-gray-800">{{ scores.white }}</div>
            </div>
          </div>
        </div>
        
        <!-- 棋盘区域 -->
        <div class="bg-white p-4 rounded-lg shadow-lg">
          <div class="w-full flex justify-center">
            <div class="w-full max-w-sm">
              <Board 
                :board="board" 
                :currentPlayer="currentPlayer" 
                :validMoves="validMoves" 
                :playerColor="playerColor"
                :gameOver="gameOver"
                :flippedPieces="flippedPieces"
                @makeMove="makeMove"
              />
            </div>
          </div>
          
          <!-- 游戏结束信息 -->
          <div v-if="gameOver" class="mt-4 text-center">
            <h3 class="text-lg font-bold mb-2">游戏结束</h3>
            <p class="text-lg" :class="winner === playerColor ? 'text-green-600' : winner ? 'text-red-600' : 'text-gray-600'">
              {{ winner ? (winner === playerColor ? '你赢了！' : '你输了！') : '平局！' }}
            </p>
            <button
              @click="restartGame"
              class="mt-3 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition text-sm"
            >
              重新开始游戏
            </button>
          </div>
        </div>
        
        <!-- 房间信息 - 紧凑两行 -->
        <div class="bg-white p-3 rounded-lg shadow-lg">
          <div class="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span class="text-gray-600">房间ID:</span>
              <div class="font-mono text-xs bg-gray-100 px-2 py-1 rounded">{{ currentRoomId }}</div>
            </div>
            <div>
              <span class="text-gray-600">当前回合:</span>
              <div class="font-semibold">{{ currentPlayer === 1 ? '黑棋' : '白棋' }}</div>
            </div>
            <div>
              <span class="text-gray-600">玩家身份:</span>
              <div class="text-xs" v-if="isSpectator">游客</div>
              <div class="text-xs" v-else-if="playerColor">{{ playerColor === 1 ? '黑方' : '白方' }}</div>
            </div>
            <div>
              <span class="text-gray-600">游戏状态:</span>
              <div class="text-xs" v-if="!gameOver">{{ getGameStatus() }}</div>
              <div class="text-xs text-gray-500" v-else>游戏结束</div>
            </div>
          </div>
          <div v-if="opponentOffline" class="mt-2 text-red-500 text-xs">
            等待对方加入...
          </div>
        </div>
        
        <!-- 常用语 -->
        <div class="bg-white p-3 rounded-lg shadow-lg">
          <h3 class="text-sm font-semibold mb-2">常用语</h3>
          <div class="flex flex-wrap gap-1 justify-start">
            <button
              v-for="(msg, index) in commonMessages"
              :key="index"
              @click="sendChat(msg)"
              class="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 text-gray-800 rounded transition whitespace-nowrap"
            >
              {{ msg }}
            </button>
          </div>
        </div>
        
        <!-- 邀请好友和退出房间 -->
        <div class="space-y-3">
          <!-- 房间邀请 -->
          <div class="bg-white p-3 rounded-lg shadow-lg">
            <h3 class="text-sm font-semibold mb-2">邀请好友</h3>
            <div class="space-y-2">
              <input 
                type="text" 
                :value="roomUrl" 
                readonly 
                class="w-full px-2 py-1 text-xs border border-gray-300 rounded bg-gray-100"
              />
              <button 
                @click="copyRoomUrl" 
                class="w-full px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 transition"
              >
                复制邀请链接
              </button>
            </div>
          </div>
          
          <!-- 退出房间按钮 -->
          <div class="bg-white p-3 rounded-lg shadow-lg">
            <button 
              @click="leaveRoom" 
              class="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300 text-sm"
            >
              退出房间
            </button>
          </div>
        </div>
      </div>
      
      <!-- 宽屏布局：保持原三列布局 -->
      <div class="hidden lg:grid lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
        
        <!-- 左侧：房间信息 -->
        <div class="lg:col-span-1 space-y-6">
          <div class="bg-white p-6 rounded-lg shadow-lg">
            <h2 class="text-xl font-bold mb-4">房间信息</h2>
            <div class="space-y-3">
              <div>
                <label class="text-sm text-gray-600">房间ID:</label>
                <div class="font-mono text-sm bg-gray-100 px-2 py-1 rounded">{{ currentRoomId }}</div>
              </div>
              
              <div>
                <label class="text-sm text-gray-600">当前回合:</label>
                <div class="font-semibold">{{ currentPlayer === 1 ? '黑棋' : '白棋' }}</div>
              </div>
              
              <div>
                <label class="text-sm text-gray-600">游戏状态:</label>
                <div class="text-sm" v-if="!gameOver">{{ getGameStatus() }}</div>
                <div class="text-sm text-gray-500" v-else>游戏结束</div>
              </div>
              
              <div>
                <label class="text-sm text-gray-600">玩家身份:</label>
                <div class="text-sm" v-if="isSpectator">游客</div>
                <div class="text-sm" v-else-if="playerColor">{{ playerColor === 1 ? '黑方' : '白方' }}</div>
              </div>
              
              <div v-if="opponentOffline" class="text-red-500 text-sm">
                等待对方加入...
              </div>
            </div>
          </div>
          
          <!-- 分数显示 -->
          <div class="bg-white p-6 rounded-lg shadow-lg">
            <h3 class="text-lg font-semibold mb-4">比分</h3>
            <div class="space-y-4">
              <div class="flex justify-between items-center">
                <div class="text-center">
                  <div class="font-semibold">黑棋</div>
                  <div class="text-xs text-gray-600">{{ getPlayerName(1) }}</div>
                </div>
                <div class="text-2xl font-bold text-gray-800">{{ scores.black }}</div>
              </div>
              <div class="flex justify-between items-center">
                <div class="text-center">
                  <div class="font-semibold">白棋</div>
                  <div class="text-xs text-gray-600">{{ getPlayerName(2) }}</div>
                </div>
                <div class="text-2xl font-bold text-gray-800">{{ scores.white }}</div>
              </div>
            </div>
          </div>
          

        </div>
        
        <!-- 中间：棋盘区域 -->
        <div class="lg:col-span-2">
          <div class="bg-white p-4 lg:p-6 rounded-lg shadow-lg">
            <!-- 棋盘组件 - 响应式尺寸 -->
            <div class="w-full flex justify-center">
              <div class="w-full max-w-md lg:max-w-lg xl:max-w-xl">
                <Board 
                  :board="board" 
                  :currentPlayer="currentPlayer" 
                  :validMoves="validMoves" 
                  :playerColor="playerColor"
                  :gameOver="gameOver"
                  :flippedPieces="flippedPieces"
                  @makeMove="makeMove"
                />
              </div>
            </div>
            
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
          </div>
        </div>
        
        <!-- 右侧：常用语和操作面板 -->
        <div class="lg:col-span-1 space-y-6">
          <!-- 房间邀请 -->
          <div class="bg-white p-4 lg:p-6 rounded-lg shadow-lg">
            <h3 class="text-lg font-semibold mb-4">邀请好友</h3>
            <div class="space-y-3">
              <input 
                type="text" 
                :value="roomUrl" 
                readonly 
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded bg-gray-100"
              />
              <button 
                @click="copyRoomUrl" 
                class="w-full px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition"
              >
                复制邀请链接
              </button>
            </div>
          </div>
          
          <!-- 常用语面板 -->
          <div class="bg-white p-4 lg:p-6 rounded-lg shadow-lg">
            <h3 class="text-lg font-semibold mb-4">常用语</h3>
            
            <!-- 常用语按钮 -->
            <div class="flex flex-wrap gap-2 justify-start">
              <button
                v-for="(msg, index) in commonMessages"
                :key="index"
                @click="sendChat(msg)"
                class="px-3 py-2 text-sm bg-gray-200 hover:bg-gray-300 text-gray-800 rounded transition whitespace-nowrap"
              >
                {{ msg }}
              </button>
            </div>
          </div>
          
          <!-- 退出房间按钮 -->
          <div class="bg-white p-4 lg:p-6 rounded-lg shadow-lg">
            <button 
              @click="leaveRoom" 
              class="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
            >
              退出房间
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import Board from './components/Board.vue';

// 游戏状态
const isInRoom = ref(false);
const roomId = ref('');
const currentRoomId = ref('');
const playerColor = ref<number | null>(null);
const isSpectator = ref(false); // 添加游客标识

// 心跳计时器
let heartbeatTimer: number | null = null;
// 昵称输入期间的心跳计时器
let nameInputHeartbeatTimer: number | null = null;
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

// 掉线检测
const opponentOffline = ref(false);
let offlineDetectionTimer: number | null = null;

// 计时器
const ws = ref<WebSocket | null>(null);
// 使用全局配置的WebSocket URL
const wsUrl = (window as any).wsUrl || 'ws://localhost:3001';

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
    
    // 检查是否需要重连
    if (reconnectInfo.value.isReconnecting && reconnectInfo.value.roomId && ws.value) {
      // 自动重连到之前的房间
      console.log('尝试重连到房间:', reconnectInfo.value.roomId);
      ws.value.send(JSON.stringify({
        type: 'RECONNECT_ROOM',
        payload: {
          roomId: reconnectInfo.value.roomId,
          playerName: reconnectInfo.value.playerName,
          playerColor: reconnectInfo.value.playerColor
        }
      }));
      
      // 显示重连提示
      showNotification('正在尝试重新连接到游戏...', 'join');
    } else if (roomIdFromUrl) {
      pendingRoomId.value = roomIdFromUrl;
      showNameInput.value = true;
      // 启动昵称输入期间的心跳保护
      startNameInputHeartbeat();
    }
  };
  
  ws.value.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleWebSocketMessage(message);
  };
  
  ws.value.onclose = (event) => {
    console.log('WebSocket disconnected:', event.code, event.reason);
    
    // 停止所有心跳计时器
    stopHeartbeatTimer();
    stopNameInputHeartbeat();
    
    // 显示重连提示
    if (isInRoom.value) {
      showNotification('连接已断开，请刷新页面重新加入房间', 'leave');
    }
    
    // 不再自动重连，让用户手动处理
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
      // 启动昵称输入期间的心跳保护
      startNameInputHeartbeat();
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
      // 启动心跳计时器
      startHeartbeatTimer();
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
      stopHeartbeatTimer(); // 停止心跳计时器
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
      
      // 检查是否是对手重新加入，清除离线状态
      if (playerColor.value && message.payload.playerColor === (playerColor.value === 1 ? 2 : 1)) {
        opponentOffline.value = false;
        stopOfflineDetectionTimer();
      }
      break;
      
    case 'PLAYER_LEFT':
      console.log('Player left:', message.payload);
      const leaveMessage = `${message.payload.playerName} (${message.payload.playerColor === 1 ? '黑方' : '白方'}) 离开了房间`;
      showNotification(leaveMessage, 'leave');
      
      // 检查是否是对手离开，更新离线状态
      if (playerColor.value && message.payload.playerColor === (playerColor.value === 1 ? 2 : 1)) {
        opponentOffline.value = true;
        startOfflineDetectionTimer();
      }
      
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
  } else if (opponentOffline.value) {
    // 对方离线，停止计时器
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

// 获取游戏状态描述
const getGameStatus = () => {
  if (!roomInfo.value?.gameStarted) {
    return '等待玩家加入';
  }
  
  if (gameOver.value) {
    return '游戏已结束';
  }
  
  if (isOpponentOffline.value) {
    return '等待对方重新加入';
  }
  
  if (isSpectator.value) {
    return '观战模式';
  }
  
  return '游戏进行中';
};

// 检测对方是否离线
const isOpponentOffline = computed(() => {
  if (!roomInfo.value?.players || isSpectator.value || !playerColor.value) {
    return false;
  }
  
  // 获取对手颜色
  const opponentColor = playerColor.value === 1 ? 2 : 1;
  const opponent = roomInfo.value.players.find((p: any) => p.color === opponentColor);
  
  // 如果没有对手玩家，或者对手是游客，说明对方离线
  if (!opponent || opponent.isSpectator) {
    return true;
  }
  
  return opponentOffline.value;
});

// 启动掉线检测计时器
const startOfflineDetectionTimer = () => {
  stopOfflineDetectionTimer();
  // 30秒后如果对方还没有重新加入，显示更强的离线提示
  offlineDetectionTimer = window.setTimeout(() => {
    if (opponentOffline.value) {
      showNotification('对方可能已离线，请耐心等待或考虑寻找新对手', 'leave');
    }
  }, 30000);
};

// 停止掉线检测计时器
const stopOfflineDetectionTimer = () => {
  if (offlineDetectionTimer) {
    clearTimeout(offlineDetectionTimer);
    offlineDetectionTimer = null;
  }
};

// 房间验证函数（仅用于确认加入时）
const validateRoom = async (roomIdToValidate: string) => {
  if (!roomIdToValidate.trim() || !ws.value || ws.value.readyState !== WebSocket.OPEN) {
    return false;
  }

  return new Promise<boolean>((resolve) => {
    const timeoutId = setTimeout(() => {
      resolve(false);
    }, 5000);

    const checkRoom = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'ROOM_INFO') {
          clearTimeout(timeoutId);
          ws.value!.removeEventListener('message', checkRoom);
          resolve(true);
        } else if (message.type === 'ERROR' && message.payload.message === 'Room not found') {
          clearTimeout(timeoutId);
          ws.value!.removeEventListener('message', checkRoom);
          resolve(false);
        }
      } catch (error) {
        console.error('Error parsing room validation message:', error);
      }
    };

    // 监听响应
    ws.value!.addEventListener('message', checkRoom);

    // 发送验证请求
    ws.value!.send(JSON.stringify({
      type: 'GET_ROOM_INFO',
      payload: { roomId: roomIdToValidate }
    }));
  });
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
  
  // 直接进入昵称输入界面，不做实时验证
  pendingRoomId.value = roomId.value;
  showNameInput.value = true;
  // 启动昵称输入期间的心跳保护
  startNameInputHeartbeat();
};

// 确认加入房间
const confirmJoinRoom = () => {
  if (!playerName.value.trim()) {
    showNotification('请输入昵称', 'leave');
    return;
  }
  
  console.log('Joining room:', pendingRoomId.value, 'as', playerName.value);
  
  // 在开始游戏时验证房间是否存在
  validateRoom(pendingRoomId.value).then((isValid) => {
    if (isValid) {
      if (ws.value && ws.value.readyState === WebSocket.OPEN) {
        ws.value.send(JSON.stringify({
          type: 'JOIN_ROOM',
          payload: {
            roomId: pendingRoomId.value,
            playerName: playerName.value
          }
        }));
        
        // 停止昵称输入期间的心跳保护
        stopNameInputHeartbeat();
        // 重置状态
        playerName.value = '';
        showNameInput.value = false;
      }
    } else {
      showNotification('房间不存在或已被清理，请检查房间ID', 'leave');
      // 保持在昵称输入界面，允许用户重新输入
    }
  });
};

// 取消加入房间
const cancelJoinRoom = () => {
  // 停止昵称输入期间的心跳保护
  stopNameInputHeartbeat();
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
  // 停止所有计时器
  stopWaitTimer();
  stopHeartbeatTimer();
  stopOfflineDetectionTimer();
  // 重置游戏状态
  isInRoom.value = false;
  currentRoomId.value = '';
  playerColor.value = null;
  roomInfo.value = null;
  opponentOffline.value = false;
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

// 发送心跳消息
const sendHeartbeat = () => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN && isInRoom.value && currentRoomId.value) {
    console.log(`发送心跳消息到房间 ${currentRoomId.value}`);
    ws.value.send(JSON.stringify({
      type: 'HEARTBEAT',
      payload: {
        roomId: currentRoomId.value
      }
    }));
  }
};

// 发送房间验证心跳消息
const sendRoomValidationHeartbeat = () => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN && isInRoom.value && currentRoomId.value) {
    console.log(`发送房间验证心跳到房间 ${currentRoomId.value}`);
    
    // 设置超时检测
    const timeoutId = setTimeout(() => {
      showNotification('房间可能已被清理，请考虑重建房间', 'leave');
      // 尝试离开当前房间
      if (ws.value && ws.value.readyState === WebSocket.OPEN) {
        ws.value.send(JSON.stringify({
          type: 'LEAVE_ROOM',
          payload: {
            roomId: currentRoomId.value
          }
        }));
      }
    }, 5000);
    
    // 监听响应
    const handleHeartbeatResponse = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'HEARTBEAT_RESPONSE' && message.payload.roomId === currentRoomId.value) {
          clearTimeout(timeoutId);
          ws.value!.removeEventListener('message', handleHeartbeatResponse);
          console.log('房间验证心跳成功');
        } else if (message.type === 'ROOM_CLOSED') {
          clearTimeout(timeoutId);
          ws.value!.removeEventListener('message', handleHeartbeatResponse);
          console.log('房间已被关闭');
        } else if (message.type === 'ERROR' && message.payload.message === 'Room not found') {
          clearTimeout(timeoutId);
          ws.value!.removeEventListener('message', handleHeartbeatResponse);
          showNotification('房间已被清理，请重建房间', 'leave');
          // 自动离开无效房间
          if (ws.value && ws.value.readyState === WebSocket.OPEN) {
            ws.value.send(JSON.stringify({
              type: 'LEAVE_ROOM',
              payload: {
                roomId: currentRoomId.value
              }
            }));
          }
        }
      } catch (error) {
        console.error('Error parsing heartbeat response:', error);
      }
    };
    
    ws.value.addEventListener('message', handleHeartbeatResponse);
    
    ws.value.send(JSON.stringify({
      type: 'HEARTBEAT',
      payload: {
        roomId: currentRoomId.value,
        validateRoom: true // 添加验证标志
      }
    }));
  }
};

// 发送昵称输入期间的心跳保护消息
const sendNameInputHeartbeat = () => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN && showNameInput.value && pendingRoomId.value) {
    console.log(`发送昵称输入心跳保护到房间 ${pendingRoomId.value}`);
    
    ws.value.send(JSON.stringify({
      type: 'GET_ROOM_INFO',
      payload: { roomId: pendingRoomId.value }
    }));
  }
};

// 启动昵称输入期间的心跳保护
const startNameInputHeartbeat = () => {
  stopNameInputHeartbeat();
  // 每30秒发送一次心跳保护
  nameInputHeartbeatTimer = window.setInterval(() => {
    sendNameInputHeartbeat();
  }, 30 * 1000); // 30秒
};

// 停止昵称输入期间的心跳保护
const stopNameInputHeartbeat = () => {
  if (nameInputHeartbeatTimer) {
    clearInterval(nameInputHeartbeatTimer);
    nameInputHeartbeatTimer = null;
  }
};

// 启动心跳计时器
const startHeartbeatTimer = () => {
  // 每1分钟发送一次心跳，确保房间不会因为用户没有操作而被误清理
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
  }
  heartbeatTimer = window.setInterval(() => {
    // 使用房间验证心跳功能
    sendRoomValidationHeartbeat();
  }, 10 * 1000); // 1分钟
};

// 停止心跳计时器
const stopHeartbeatTimer = () => {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
};

// 重连状态保存
const reconnectInfo = ref({
  roomId: '',
  playerName: '',
  playerColor: null as number | null,
  isReconnecting: false
});

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

  // 监听页面卸载前事件，尝试发送离开消息
  const handleBeforeUnload = () => {
    if (ws.value && ws.value.readyState === WebSocket.OPEN && isInRoom.value) {
      // 发送离开房间消息
      ws.value.send(JSON.stringify({
        type: 'LEAVE_ROOM',
        payload: {
          roomId: currentRoomId.value
        }
      }));
    }
  };

  window.addEventListener('beforeunload', handleBeforeUnload);

  onBeforeUnmount(() => {
    window.removeEventListener('beforeunload', handleBeforeUnload);
    // 清理所有计时器
    stopNameInputHeartbeat();
    stopHeartbeatTimer();
    stopWaitTimer();
    stopOfflineDetectionTimer();
  });
});

onBeforeUnmount(() => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN && isInRoom.value) {
    // 保存重连信息
    reconnectInfo.value = {
      roomId: currentRoomId.value,
      playerName: roomInfo.value?.players?.find((p: any) => p.color === playerColor.value)?.name || '',
      playerColor: playerColor.value,
      isReconnecting: true
    };
  }
  if (ws.value) {
    ws.value.close();
  }
  stopWaitTimer();
  stopHeartbeatTimer(); // 停止心跳计时器
});
</script>