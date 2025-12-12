// WordTooltip.js - Word tooltip component
import { ref, reactive, onMounted, onUnmounted, defineComponent } from '../lib/vue/vue.esm-browser.js';

export default defineComponent({
  name: 'WordTooltip',
  props: {
    visible: {
      type: Boolean,
      default: false
    },
    position: {
      type: Object,
      default: () => ({ x: 0, y: 0 })
    },
    isLoading: {
      type: Boolean,
      default: false
    },
    wordInfo: {
      type: Object,
      default: null
    }
  },
  setup(props, { emit }) {
    const isHovered = ref(false);
    const showDefinition = ref(false);

    // Handle tooltip mouse enter
    function handleMouseEnter() {
      isHovered.value = true;
      emit('mouse-enter');
    }

    // Handle tooltip mouse leave
    function handleMouseLeave() {
      isHovered.value = false;
      emit('mouse-leave');
    }

    // Close tooltip
    function closeTooltip() {
      emit('close');
    }

    // Handle click outside tooltip to close it
    function handleClickOutside(event) {
      if (props.visible) {
        const tooltipElement = document.querySelector('.word-tooltip:not(.hidden)');
        // Check if click came from a word element to prevent immediate closing when clicking word
        const wordElement = event.target.closest('.word-block');
        if (tooltipElement && !tooltipElement.contains(event.target) && !wordElement) {
          closeTooltip();
        }
      }
    }

    // Handle ESC key to close tooltip
    function handleKeyDown(event) {
      if (event.code === 'Escape' && props.visible) {
        closeTooltip();
      }
    }

    // Setup event listeners
    onMounted(() => {
      document.addEventListener('click', handleClickOutside);
      window.addEventListener('keydown', handleKeyDown);
    });

    // Clean up event listeners
    onUnmounted(() => {
      document.removeEventListener('click', handleClickOutside);
      window.removeEventListener('keydown', handleKeyDown);
    });

    return {
      isHovered,
      showDefinition,
      handleMouseEnter,
      handleMouseLeave,
      closeTooltip
    };
  },
  template: `
    <div 
      class="word-tooltip" 
      :class="{ 'hidden': !visible }"
      :style="{ left: position.x + 'px', top: position.y + 'px' }"
      @mouseenter="handleMouseEnter"
      @mouseleave="handleMouseLeave"
    >
      <button class="tooltip-close-btn" @click="closeTooltip">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
      <div v-if="isLoading" class="loading-spinner">
        <div class="spinner"></div>
      </div>
      <div v-else-if="wordInfo">
        <div class="word-info-header">
          <h2 class="word-info-word">{{ wordInfo.word }}</h2>
          <span v-if="wordInfo.phonetic" class="word-info-phonetic">{{ wordInfo.phonetic }}</span>
        </div>
        
        <div v-if="wordInfo.translation" class="word-info-section">
          <div class="word-info-label">翻译</div>
          <div class="word-info-content">{{ wordInfo.translation }}</div>
        </div>
        
        <div class="word-info-section">
          <a @click="showDefinition = !showDefinition" class="toggle-definition-link" style="cursor: pointer; color: #4f46e5; text-decoration: underline;">
            {{ showDefinition ? '隐藏定义' : '显示定义' }}
          </a>
        </div>
        
        <div v-if="showDefinition && wordInfo.definition" class="word-info-section">
          <div class="word-info-label">定义</div>
          <div class="word-info-content">{{ wordInfo.definition }}</div>
        </div>
        
        <div v-if="wordInfo.exchange" class="word-info-section">
          <div class="word-info-label">变化形式</div>
          <div class="word-info-content">{{ wordInfo.exchange }}</div>
        </div>
      </div>
      <div v-else class="word-info-section">
        <div class="word-info-label">未找到</div>
        <div class="word-info-content">词典中未找到该单词的信息</div>
      </div>
    </div>
  `
});
