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
      // ::: formula 块的 body 形如 "$$\n...\n$$"（display）或 "$...$"（inline）。
      // 把整段喂给 KaTeX 会让它把首尾的 $ 当函数 token 报错，
      // 因此 display 模式先剥掉首尾的 $$\n...\n$$ 包裹符再渲染。
      const display = props.data.display !== false;
      const raw = props.data.body;
      const tex = display
        ? (raw.match(/^\$\$([\s\S]*?)\$\$$/) ? RegExp.$1.trim() : raw)
        : raw;
      try {
        window.katex.render(tex, ref0.value, {
          throwOnError: false,
          displayMode: display,
        });
      } catch (e) {
        ref0.value.textContent = raw;
      }
    });

    return { ref0 };
  },
  template: `<div ref="ref0" style="text-align:center; margin:16px 0; padding:8px"></div>`,
};
