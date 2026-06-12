// router.js — vue-router 配置（hash 路由）
// 视图：HomeView / ChapterView / CertView
import { createRouter, createWebHashHistory } from "vue-router";
import { HomeView } from "./components/HomeView.js";
import { ChapterView } from "./components/ChapterView.js";
import { CertView } from "./components/CertView.js";

export function createAppRouter() {
  const routes = [
    { path: "/", name: "home", component: HomeView },
    { path: "/ch/:id", name: "chapter", component: ChapterView, props: true },
    { path: "/cert", name: "cert", component: CertView },
  ];

  return createRouter({
    history: createWebHashHistory(),
    routes,
  });
}
