// app.js — Vue 应用入口
// 1. 加载 chapters.json
// 2. 创建 store
// 3. 创建 router
// 4. 挂载全局组件
// 5. createApp + 挂载到 #app

import { createApp } from "vue";
import { createStore, provideStore } from "./store.js";
import { createAppRouter } from "./router.js";
import { AppShell } from "./components/AppShell.js";

import { Quiz } from "./components/Quiz.js";
import { ChartBlock } from "./components/ChartBlock.js";
import { ComputeGraph } from "./components/ComputeGraph.js";
import { NetworkViz } from "./components/NetworkViz.js";
import { TrainDemo } from "./components/TrainDemo.js";
import { Formula } from "./components/Formula.js";
import { SigmoidSlider } from "./components/SigmoidSlider.js";
import { MathSlider } from "./components/MathSlider.js";
import { PerceptronPlayground } from "./components/PerceptronPlayground.js";

// 给 KaTeX 默认关掉 strict 警告
// （公式里 \text{...} 出现 → 等 Unicode 时，KaTeX 0.16.9 默认 strict="warn" 会刷屏，
//  对正确渲染无影响，关掉即可）
(function patchKaTeX() {
  function tryPatch() {
    if (!window.katex || window.__katexPatched) return;
    const k = window.katex;
    const origRender = k.render;
    const origRenderToString = k.renderToString;
    function withStrict(options) {
      const o = options || {};
      if (o.strict === undefined) o.strict = "ignore";
      return o;
    }
    k.render = function (tex, node, options) {
      return origRender.call(this, tex, node, withStrict(options));
    };
    k.renderToString = function (tex, options) {
      return origRenderToString.call(this, tex, withStrict(options));
    };
    window.__katexPatched = true;
  }
  if (window.katex) tryPatch();
  else window.addEventListener("DOMContentLoaded", tryPatch);
  // katex 标签带 defer，等下一个 tick 再补一次
  setTimeout(tryPatch, 0);
  setTimeout(tryPatch, 200);
})();

async function bootstrap() {
  // 1) 加载 chapters.json
  const res = await fetch("chapters.json");
  if (!res.ok) throw new Error("Failed to load chapters.json");
  const data = await res.json();

  // 2) 初始化 store
  createStore(data.chapters, data.certificate);
  console.log(`[009] 加载 ${data.chapters.length} 章 + 证书配置`);

  // 3) 创建 router
  const router = createAppRouter();

  // 4) 创建 app
  const app = createApp(AppShell);
  app.use(router);
  provideStore(app);

  // 5) 注册全局组件（容器扩展要用）
  app.component("Quiz", Quiz);
  app.component("ChartBlock", ChartBlock);
  app.component("ComputeGraph", ComputeGraph);
  app.component("NetworkViz", NetworkViz);
  app.component("TrainDemo", TrainDemo);
  app.component("Formula", Formula);
  app.component("SigmoidSlider", SigmoidSlider);
  app.component("MathSlider", MathSlider);
  app.component("PerceptronPlayground", PerceptronPlayground);

  // 6) 挂载
  app.mount("#app");
  console.log("[009] 启动完成");
}

bootstrap().catch((e) => {
  console.error("[009] 启动失败:", e);
  document.getElementById("app").innerHTML = `
    <div style="padding:40px; text-align:center; color:#ef4444; font-family:sans-serif;">
      <h2>😢 启动失败</h2>
      <p>${e.message}</p>
      <p style="color:#64748b; font-size:13px;">请检查 chapters.json 是否存在 / 控制台是否有详细错误</p>
    </div>
  `;
});
