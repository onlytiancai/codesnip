<template>
  <div class="bg-green-700 p-2 rounded-lg shadow-lg w-full max-w-2xl mx-auto">
    <div class="grid grid-cols-8 gap-0.5">
      <!-- 棋盘格子 -->
      <div
        v-for="(_, rowIndex) in 8"
        :key="`row-${rowIndex}`"
        class="flex flex-col gap-0.5"
      >
        <div
          v-for="(_, colIndex) in 8"
          :key="`cell-${rowIndex}-${colIndex}`"
          class="aspect-square bg-green-600 hover:bg-green-500 cursor-pointer rounded-sm transition-colors"
          :class="{
            'bg-blue-400 hover:bg-blue-500': isValidMove(rowIndex, colIndex),
            'cursor-not-allowed': gameOver || currentPlayer !== playerColor
          }"
          @click="handleCellClick(rowIndex, colIndex)"
        >
          <!-- 棋子 -->
          <div
            v-if="board[rowIndex][colIndex] !== 0"
            class="w-full h-full rounded-full shadow-lg flex items-center justify-center"
            :class="{
              'bg-black text-white': board[rowIndex][colIndex] === 1,
              'bg-white text-black': board[rowIndex][colIndex] === 2
            }"
          >
            <span class="text-2xl font-bold">{{ board[rowIndex][colIndex] === 1 ? '●' : '○' }}</span>
          </div>
          <!-- 有效落子标记 -->
          <div
            v-else-if="isValidMove(rowIndex, colIndex)"
            class="w-3 h-3 bg-blue-300 rounded-full mx-auto mt-3"
          ></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// 定义props类型
interface Props {
  board: number[][];
  currentPlayer: number;
  validMoves: { row: number; col: number }[];
  playerColor: number | null;
  gameOver: boolean;
}

// 定义emit类型
interface Emits {
  makeMove: [row: number, col: number];
}

// 接收props和emit
const props = defineProps<Props>();
const emit = defineEmits<Emits>();

// 检查是否是有效落子
const isValidMove = (row: number, col: number): boolean => {
  return props.validMoves.some((move: { row: number; col: number }) => move.row === row && move.col === col);
};

// 处理格子点击
const handleCellClick = (row: number, col: number) => {
  // 如果游戏结束，或者不是当前玩家的回合，或者不是有效落子，则不处理
  if (props.gameOver || props.currentPlayer !== props.playerColor || !isValidMove(row, col)) {
    return;
  }
  
  // 发送落子请求
  emit('makeMove', row, col);
};
</script>