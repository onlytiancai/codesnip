// components/WordBlock.js - Word block component
import { defineComponent } from '../lib/vue/vue.esm-browser.js';

export default defineComponent({
  props: {
    word: {
      type: Object,
      required: true
    },
    showIpa: {
      type: Boolean,
      default: false
    },
    sentenceIndex: {
      type: Number,
      required: true
    },
    wordIndex: {
      type: Number,
      required: true
    },
    onClick: {
      type: Function,
      default: () => {}
    }
  },
  setup(props) {
    return {
      handleClick: () => {
        props.onClick(props.word.wordIndex);
      }
    };
  },
  template: `
    <span class="word-block">
      <span :id="'word-'+sentenceIndex+'-'+wordIndex" class="word-top" :class="word.highlight ? 'hl' : ''" @click="handleClick">
        {{ word.word }}
      </span>
      <span v-if="showIpa" class="ipa">{{ word.ipa || '—' }}</span>
    </span>
  `
});
