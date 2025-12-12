// components/Sentence.js - Sentence component
import { defineComponent } from '../lib/vue/vue.esm-browser.js';
import WordBlock from './WordBlock.js';

export default defineComponent({
  components: {
    WordBlock
  },
  props: {
    sentence: {
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
    isCurrent: {
      type: Boolean,
      default: false
    },
    onWordClick: {
      type: Function,
      default: () => {}
    },
    onWordMouseEnter: {
      type: Function,
      default: () => {}
    },
    onWordMouseLeave: {
      type: Function,
      default: () => {}
    }
  },
  setup(props) {
    return {};
  },
  template: `
    <span v-if="sentence.words.length > 0" class="sentence" :class="{'sentence-hl': isCurrent}">
      <WordBlock
          v-for="(w, idx) in sentence.words"
          :key="idx"
          :word="w"
          :show-ipa="showIpa"
          :sentence-index="sentenceIndex"
          :word-index="idx"
          :on-click="onWordClick"
          :on-mouse-enter="onWordMouseEnter"
          :on-mouse-leave="onWordMouseLeave"
        />
    </span>
  `
});
