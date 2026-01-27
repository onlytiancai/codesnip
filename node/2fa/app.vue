<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-green-50 flex flex-col items-center justify-center p-4">
    <NuxtRouteAnnouncer />
    <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-green-600 mb-12">
      Tailwind 3 样式测试
    </h1>
    <div class="bg-white rounded-xl shadow-xl p-8 w-full max-w-md">
      <h2 class="text-2xl font-bold text-gray-800 mb-6 text-center">按钮样式测试</h2>
      <div class="space-y-4">
        <button class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105">
          主要按钮
        </button>
        <button class="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105">
          成功按钮
        </button>
        <button class="w-full bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105">
          警告按钮
        </button>
        <button class="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105">
          危险按钮
        </button>
        <button class="w-full bg-gray-500 hover:bg-gray-600 text-white font-bold py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105">
          次要按钮
        </button>
        <button class="w-full bg-transparent border border-blue-500 text-blue-500 hover:bg-blue-50 font-bold py-3 px-4 rounded-lg transition-all duration-300 transform hover:scale-105">
          轮廓按钮
        </button>
      </div>
      <div class="mt-8 p-4 bg-gray-50 rounded-lg">
        <p class="text-gray-600">
          如果您看到彩色按钮和渐变标题，则说明 Tailwind CSS 3 已成功集成并正常工作。
        </p>
      </div>
    </div>
  </div>
</template>
