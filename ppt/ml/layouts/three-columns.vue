<template>
  <div class="slidev-layout">
    <div>
      <slot />
    </div>
    <div class="w-full grid grid-cols-3">
      <div>
        <slot name="left" />
      </div>
      <div>
        <slot name="middle" />
      </div>
      <div>
        <slot name="right" />
      </div>
    </div>
  </div>
</template>