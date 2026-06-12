// Formula.js — 公式块（KaTeX 渲染）
// 适用于显示复杂多行公式，比行内 $...$ 更突出
import { onMounted, ref } from "vue";

export const Formula = {
  props: ["data"],
  setup(props) {
    const ref0 = ref(null);

    onMounted(() => {
      if (typeof window.katex === "undefined") {
        setTimeout(() => onMounted(), 50);
        return;
      }
      try {
        window.katex.render(props.data.body, ref0.value, {
          throwOnError: false,
          displayMode: props.data.display !== false,
        });
      } catch (e) {
        ref0.value.textContent = props.data.body;
      }
    });

    return { ref0 };
  },
  template: `<div ref="ref0" style="text-align:center; margin:16px 0; padding:8px"></div>`,
};
