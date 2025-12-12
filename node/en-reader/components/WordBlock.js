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
    },
    onMouseEnter: {
      type: Function,
      default: () => {}
    },
    onMouseLeave: {
      type: Function,
      default: () => {}
    }
  },
  setup(props) {
    return {
      handleClick: () => {
        props.onClick(props.word.wordIndex, props.sentenceIndex, props.wordIndex);
      },
      handleMouseEnter: () => {
        props.onMouseEnter(props.word.wordIndex, props.sentenceIndex, props.wordIndex);
      },
      handleMouseLeave: () => {
        props.onMouseLeave(props.word.wordIndex);
      }
    };
  },
  template: `
    <span class="word-block">
      <span :id="'word-'+sentenceIndex+'-'+wordIndex" class="word-top" :class="{'hl': word.highlight, 'hover-hl': word.hover}" @click="handleClick" @mouseenter="handleMouseEnter" @mouseleave="handleMouseLeave">
        {{ word.word }}
      </span>
      <span v-if="showIpa" class="ipa">{{ word.ipa || '—' }}</span>
    </span>
  `
});
