<template>
  <div class="slidev-layout">
    <div>
      <slot />
    </div>
    <div class="w-full h-full grid grid-cols-[2fr_1fr] ">
      <div class="p-4">
        <slot name="left" />
      </div>
      <div class="p-4">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>