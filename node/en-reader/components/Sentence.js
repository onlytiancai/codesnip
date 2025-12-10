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
    showIPA: {
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
    }
  },
  setup(props) {
    return {};
  },
  template: `
    <template v-if="sentence.isNewline">
      <span class="sentence">
        <span v-for="n in sentence.newline_count" :key="n" class="line-break"><br /></span>
      </span>
    </template>
    <span v-else class="sentence" :class="{'sentence-hl': isCurrent}">
      <WordBlock
        v-for="(w, idx) in sentence.words"
        :key="idx"
        :word="w"
        :show-ipa="showIPA"
        :sentence-index="sentenceIndex"
        :word-index="idx"
        :on-click="onWordClick"
      />
    </span>
  `
});
