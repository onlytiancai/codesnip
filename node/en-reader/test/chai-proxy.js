// chai-proxy.js: 动态选择 Chai 导入源的代理模块

// 检测是否在浏览器环境中运行
const isBrowser = typeof window !== 'undefined' && typeof window.document !== 'undefined';

// 使用 Promise.resolve() 来确保无论哪种情况都返回一个 Promise
const getChaiModule = async () => {
  if (isBrowser && window.chai) {
    // 浏览器环境：使用全局作用域中的 Chai（从 HTML 中通过 script 标签引入）
    return Promise.resolve(window.chai);
  } else {
    // 动态导入 Chai（适用于 Node.js 或未预加载 Chai 的浏览器环境）
    return await (isBrowser 
      ? import('https://cdn.jsdelivr.net/npm/chai@4.3.7/+esm') 
      : import('chai'));
  }
};

// 获取 Chai 模块
const chaiModule = await getChaiModule();

// 导出功能
export const expect = chaiModule.expect;
export const assert = chaiModule.assert;
export const should = chaiModule.should;
export const chai = chaiModule;
export default chaiModule;
