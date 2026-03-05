<template>
  <div class="min-h-screen bg-gradient-to-b from-gray-50 to-white">
    <!-- Header -->
    <header class="bg-white/80 backdrop-blur-md border-b border-gray-100 sticky top-0 z-50">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <NuxtLink to="/" class="flex items-center space-x-2">
            <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
              <span class="text-white font-bold text-sm">Eng</span>
            </div>
            <span class="text-xl font-bold text-gray-900">程序员英语</span>
          </NuxtLink>

          <nav class="hidden md:flex items-center space-x-8">
            <NuxtLink to="/" class="text-gray-600 hover:text-gray-900 transition-colors">首页</NuxtLink>
            <NuxtLink to="/articles" class="text-gray-600 hover:text-gray-900 transition-colors">文章</NuxtLink>
            <NuxtLink to="/categories" class="text-gray-600 hover:text-gray-900 transition-colors">分类</NuxtLink>
          </nav>

          <div class="flex items-center space-x-4">
            <NuxtLink to="/login" class="text-gray-600 hover:text-gray-900 transition-colors">
              登录
            </NuxtLink>
            <NuxtLink
              to="/register"
              class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              注册
            </NuxtLink>
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1">
      <slot />
    </main>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white mt-16">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <div class="flex items-center space-x-2 mb-4">
              <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                <span class="text-white font-bold text-sm">Eng</span>
              </div>
              <span class="text-xl font-bold">程序员英语</span>
            </div>
            <p class="text-gray-400 text-sm">
              专为程序员设计的英语阅读学习应用，提升技术英语阅读和口语能力。
            </p>
          </div>

          <div>
            <h3 class="font-semibold mb-4">功能</h3>
            <ul class="space-y-2 text-gray-400 text-sm">
              <li>沉浸式阅读</li>
              <li>逐句跟读练习</li>
              <li>AI 语音反馈</li>
              <li>单词即点即查</li>
            </ul>
          </div>

          <div>
            <h3 class="font-semibold mb-4">链接</h3>
            <ul class="space-y-2 text-gray-400 text-sm">
              <li><NuxtLink to="/articles" class="hover:text-white">文章列表</NuxtLink></li>
              <li><NuxtLink to="/categories" class="hover:text-white">分类浏览</NuxtLink></li>
              <li><NuxtLink to="/about" class="hover:text-white">关于我们</NuxtLink></li>
            </ul>
          </div>
        </div>

        <div class="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400 text-sm">
          <p>&copy; 2026 程序员英语。All rights reserved.</p>
        </div>
      </div>
    </footer>
  </div>
</template>
