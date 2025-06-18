
<template>
  <div class="slidev-layout w-full h-full grid grid-cols-[2fr_1fr] " >
    <div class="p-4" >
      <slot />
    </div>
    <div class="p-4">
      <slot name="right" />
    </div>
  </div>
</template>