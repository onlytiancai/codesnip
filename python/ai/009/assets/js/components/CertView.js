// CertView.js — 证书页
// 1. 检查所有章节是否完成
// 2. 收集学员姓名
// 3. 渲染 SVG 证书
// 4. 提供"下载 PNG"（html2canvas）和"打印"按钮
import { ref, computed, onMounted } from "vue";
import { useStore } from "../store.js";
import { I18N, pick } from "../i18n.js";

export const CertView = {
  setup() {
    const { state, actions } = useStore();
    const t = (k) => pick(I18N.ui[k] || { zh: k, en: k }, state.language);
    const isZh = computed(() => state.language === "zh");

    const completedCount = computed(() =>
      Object.values(state.progress?.chapters || {}).filter(
        (c) => c[state.language]?.status === "completed"
      ).length
    );
    const totalChapters = computed(() => state.chapters.filter((c) => !c.is_optional).length);
    const allCompleted = computed(() => completedCount.value >= totalChapters.value);
    const overallScore = computed(() => {
      const s = state.progress?.summary;
      if (!s?.quizzes_total) return 0;
      return s.quizzes_correct / s.quizzes_total;
    });

    // 证书 = 完成证明：只看章节完成，不看准确率
    const eligible = computed(() => allCompleted.value);

    const name = ref(state.studentName || "");
    const certId = ref("");
    const issuedAt = ref("");

    function generateCertId() {
      // 简单 hash：学员名 + 完成时间戳
      const s = (name.value || "anonymous") + Date.now();
      let h = 0;
      for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
      const code = Math.abs(h).toString(36).slice(0, 6).toUpperCase();
      const d = new Date();
      const date = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
      return `mlp-${date}-${code}`;
    }

    function issue() {
      if (!name.value.trim()) {
        alert(isZh.value ? "请先填写你的名字" : "Please enter your name first");
        return;
      }
      actions.setStudentName(name.value.trim());
      certId.value = generateCertId();
      issuedAt.value = new Date().toISOString();
      const cert = {
        issued: true,
        issued_at: issuedAt.value,
        student_name: name.value.trim(),
        overall_score: overallScore.value,
        certificate_id: certId.value,
        lang_at_issue: state.language,
      };
      actions.issueCert(cert);
    }

    async function downloadPng() {
      if (typeof window.html2canvas !== "function") {
        alert(isZh.value ? "html2canvas 未加载" : "html2canvas not loaded");
        return;
      }
      const el = document.getElementById("cert-svg-wrap");
      if (!el) return;
      try {
        const canvas = await window.html2canvas(el, {
          backgroundColor: getComputedStyle(document.body).getPropertyValue("--bg") || "#fafbff",
          scale: 2,
        });
        const link = document.createElement("a");
        link.download = `mlp-certificate-${certId.value}.png`;
        link.href = canvas.toDataURL("image/png");
        link.click();
      } catch (e) {
        console.error(e);
        alert(isZh.value ? "下载失败：" + e.message : "Download failed: " + e.message);
      }
    }

    function print() {
      window.print();
    }

    // 自动恢复已颁发证书
    onMounted(() => {
      const existing = actions.getCert();
      if (existing && existing.issued) {
        certId.value = existing.certificate_id;
        issuedAt.value = existing.issued_at;
      }
    });

    const issueDate = computed(() => {
      if (!issuedAt.value) return "—";
      const d = new Date(issuedAt.value);
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    });

    return {
      state, t, isZh,
      completedCount, totalChapters, allCompleted, overallScore, eligible,
      name, certId, issuedAt, issueDate,
      issue, downloadPng, print,
    };
  },
  template: `
    <div v-if="!eligible" class="cert-locked">
      <div class="lock-icon">🔒</div>
      <h2>{{ t('lockTitle') }}</h2>
      <p>{{ t('lockDesc') }}</p>
      <div class="progress-info">
        <div>{{ t('chaptersCompleted') }}：<strong>{{ completedCount }} / {{ totalChapters }}</strong></div>
      </div>
    </div>

    <div v-else class="cert-page">
      <div id="cert-svg-wrap" class="cert-bg">
        <div style="font-size:48px; margin-bottom:8px">🎓</div>
        <h1>{{ state.language === 'en' ? state.certMeta.title_en : state.certMeta.title_zh }}</h1>
        <p class="cert-subtitle">
          {{ state.language === 'en' ? state.certMeta.subtitle_en : state.certMeta.subtitle_zh }}
        </p>

        <div v-if="!certId" style="margin:20px 0">
          <p style="margin-bottom:8px; color:var(--muted); font-size:13px">
            {{ state.language === 'en' ? 'Enter your name to issue the certificate:' : '请输入你的名字以颁发证书：' }}
          </p>
          <input v-model="name" :placeholder="t('enterName')" style="width:300px; max-width:80%; text-align:center; font-size:16px; padding:10px" />
          <div style="margin-top:14px">
            <button class="primary" @click="issue" style="font-size:14px; padding:10px 28px">
              {{ state.language === 'en' ? '🎉 Issue Certificate' : '🎉 颁发证书' }}
            </button>
          </div>
        </div>

        <div v-else>
          <p style="color:var(--muted); margin:8px 0 4px; font-size:13px">
            {{ state.language === 'en' ? 'This is to certify that' : '兹证明' }}
          </p>
          <div class="cert-name">{{ name }}</div>
          <p style="color:var(--muted); margin:4px 0 8px; font-size:13px">
            {{ state.language === 'en' ? 'has successfully completed' : '已成功完成' }}
          </p>
          <div class="cert-score">{{ (overallScore * 100).toFixed(0) }}%</div>

          <div class="cert-meta">
            <div>
              <div class="lbl">{{ t('issuedAt') }}</div>
              <div class="val">{{ issueDate }}</div>
            </div>
            <div>
              <div class="lbl">{{ t('chaptersCompleted') }}</div>
              <div class="val">{{ completedCount }} / {{ totalChapters }}</div>
            </div>
            <div>
              <div class="lbl">{{ t('overallScore') }}</div>
              <div class="val">{{ (overallScore * 100).toFixed(0) }}%</div>
            </div>
          </div>

          <div class="cert-chapters">
            <h3>{{ state.language === 'en' ? 'Chapters Completed' : '完成章节' }}</h3>
            <ul>
              <li v-for="ch in state.chapters.filter(c => !c.is_optional)" :key="ch.id">
                {{ state.language === 'en' ? ch.title_en : ch.title_zh }}
              </li>
            </ul>
          </div>

          <div class="cert-id">{{ t('certificateId') }}: {{ certId }}</div>
        </div>
      </div>

      <div v-if="certId" class="cert-actions no-print">
        <button class="primary" @click="downloadPng">📥 {{ t('download') }}</button>
        <button @click="print">🖨 {{ t('print') }}</button>
      </div>
    </div>
  `,
};
