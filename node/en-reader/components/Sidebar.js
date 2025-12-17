import { ref, onMounted, watchEffect } from '../lib/vue/vue.esm-browser.js';

export default {
  name: 'Sidebar',
  props: {
    isOpen: {
      type: Boolean,
      default: false
    }
  },
  emits: ['toggle', 'select-text'],
  setup(props, { emit }) {
    const recentTexts = ref([]);
    const isLoading = ref(true);
    
    // IndexedDB相关
    const dbName = 'RecentTextsDB';
    const dbVersion = 1;
    let db = null;
    
    // 初始化IndexedDB
    function initDB() {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open(dbName, dbVersion);
        
        request.onerror = (event) => {
          console.error('IndexedDB error:', event.target.error);
          reject(event.target.error);
        };
        
        request.onsuccess = (event) => {
          db = event.target.result;
          resolve(db);
        };
        
        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          
          // 创建存储对象
          const objectStore = db.createObjectStore('recentTexts', { keyPath: 'id', autoIncrement: true });
          
          // 创建索引
          objectStore.createIndex('timestamp', 'timestamp', { unique: false });
        };
      });
    }
    
    // 获取所有最近使用的文本
    async function getRecentTexts() {
      if (!db) {
        await initDB();
      }
      
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(['recentTexts'], 'readonly');
        const objectStore = transaction.objectStore('recentTexts');
        const index = objectStore.index('timestamp');
        const request = index.getAll(null, 10); // 获取最近10条
        
        request.onerror = (event) => {
          reject(event.target.error);
        };
        
        request.onsuccess = (event) => {
          // 按时间倒序排列
          const texts = event.target.result.sort((a, b) => b.timestamp - a.timestamp);
          resolve(texts);
        };
      });
    }
    
    // 保存文本到IndexedDB
    async function saveText(text) {
      if (!db) {
        await initDB();
      }
      
      // 检查是否已存在相同文本
      const existingTexts = await getRecentTexts();
      const existingIndex = existingTexts.findIndex(item => item.text === text);
      
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(['recentTexts'], 'readwrite');
        const objectStore = transaction.objectStore('recentTexts');
        
        if (existingIndex !== -1) {
          // 更新现有记录的时间戳
          const existingText = existingTexts[existingIndex];
          existingText.timestamp = Date.now();
          const request = objectStore.put(existingText);
          
          request.onerror = (event) => reject(event.target.error);
          request.onsuccess = () => resolve();
        } else {
          // 添加新记录
          const newText = {
            text,
            timestamp: Date.now()
          };
          const request = objectStore.add(newText);
          
          request.onerror = (event) => reject(event.target.error);
          request.onsuccess = () => resolve();
        }
      });
    }
    
    // 加载最近使用的文本
    async function loadRecentTexts() {
      isLoading.value = true;
      try {
        recentTexts.value = await getRecentTexts();
      } catch (error) {
        console.error('Failed to load recent texts:', error);
      } finally {
        isLoading.value = false;
      }
    }
    
    // 处理文本选择
    function handleSelectText(text) {
      emit('select-text', text);
    }
    
    // 切换侧边栏
    function toggleSidebar() {
      emit('toggle');
    }
    
    // 组件挂载时初始化
    onMounted(async () => {
      await initDB();
      await loadRecentTexts();
    });
    
    // 格式化日期
    function formatDate(timestamp) {
      return new Date(timestamp).toLocaleString();
    }

    // 暴露方法给父组件
    return {
      recentTexts,
      isLoading,
      toggleSidebar,
      handleSelectText,
      saveText,
      loadRecentTexts,
      formatDate
    };
  },
  template: `
    <div class="sidebar-container" :class="{ 'open': isOpen }">
      <div class="sidebar-toggle" @click="toggleSidebar">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline v-if="isOpen" points="15 18 9 12 15 6"></polyline>
          <polyline v-else points="9 18 15 12 9 6"></polyline>
        </svg>
      </div>
      <div class="sidebar-content">
        <div class="sidebar-header">
          <h2>最近使用</h2>
        </div>
        <div class="sidebar-body">
          <div v-if="isLoading" class="loading">
            <div class="spinner"></div>
          </div>
          <div v-else-if="recentTexts.length === 0" class="empty-text">
            暂无最近使用的文本
          </div>
          <div v-else class="recent-texts-list">
        <div 
          v-for="(item, index) in recentTexts" 
          :key="item.id" 
          class="recent-text-item"
          @click="handleSelectText(item.text)"
        >
          <div class="text-preview">{{ item.text.substring(0, 100) }}{{ item.text.length > 100 ? '...' : '' }}</div>
          <div class="text-time">{{ formatDate(item.timestamp) }}</div>
        </div>
      </div>
        </div>
      </div>
    </div>
  `
};
