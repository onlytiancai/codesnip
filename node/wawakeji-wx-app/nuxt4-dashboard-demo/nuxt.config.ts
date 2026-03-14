export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: true },
  modules: [["@nuxt/ui", { fonts: false }], "@nuxt/icon", "nuxt-auth-utils"],
  css: ["~/assets/css/main.css"],
  runtimeConfig: {
    oauth: {
      wechat: {
        clientId: process.env.NUXT_OAUTH_WECHAT_CLIENT_ID,
        clientSecret: process.env.NUXT_OAUTH_WECHAT_CLIENT_SECRET,
      },
      wechatMp: {
        appid: process.env.NUXT_OAUTH_WECHAT_MP_APPID,
        secret: process.env.NUXT_OAUTH_WECHAT_MP_SECRET,
      },
    },
    wechatPay: {
      mchId: process.env.NUXT_WECHAT_PAY_MCH_ID,
      apiV3Key: process.env.NUXT_WECHAT_PAY_API_V3_KEY,
      serialNo: process.env.NUXT_WECHAT_PAY_SERIAL_NO,
      privateKey: process.env.NUXT_WECHAT_PAY_PRIVATE_KEY,
      notifyUrl: process.env.NUXT_WECHAT_PAY_NOTIFY_URL,
    },
  },
  vite: {
    server: {
      allowedHosts: ["tools.myapp1024.com"],
    },
  },
});