// ChartBlock.js — 静态配图 + caption
// 用法：
//   ::: chart caption="..." width="600" alt="..."
//   ![描述](assets/images/xxx.png)
//   :::
import { computed } from "vue";
import { useStore } from "../store.js";

export const ChartBlock = {
  props: ["data"],
  setup(props) {
    const { state } = useStore();

    const args = computed(() => {
      const a = {};
      const re = /(\w+)="([^"]*)"/g;
      let m;
      while ((m = re.exec(props.data.args)) !== null) a[m[1]] = m[2];
      return a;
    });

    // 从 body 提取 markdown 图片 ![alt](url)
    const imgMatch = computed(() => {
      const m = props.data.body.match(/!\[([^\]]*)\]\(([^)]+)\)/);
      return m ? { alt: m[1], url: m[2] } : null;
    });

    const caption = computed(() => args.value.caption || imgMatch.value?.alt || "");
    const alt = computed(() => args.value.alt || imgMatch.value?.alt || "illustration");

    return { state, args, imgMatch, caption, alt };
  },
  template: `
    <figure v-if="imgMatch" style="margin:24px 0; text-align:center">
      <img :src="imgMatch.url" :alt="alt" :style="'max-width:' + (args.width || '100%')" loading="lazy" />
      <figcaption v-if="caption" style="margin-top:8px; color:var(--muted); font-size:13px; font-style:italic">
        {{ caption }}
      </figcaption>
    </figure>
    <div v-else style="padding:14px; background:var(--panel2); border-radius:8px; color:var(--muted); font-size:13px">
      ⚠️ Chart block missing image (expected ![alt](url) in body)
    </div>
  `,
};
