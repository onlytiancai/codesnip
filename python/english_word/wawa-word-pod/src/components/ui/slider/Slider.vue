<script setup>
import { watch } from "vue";
import { cn } from "@/lib/utils";

const props = defineProps({
  modelValue: { type: Array, required: true },
  min: { type: Number, required: false, default: 0 },
  max: { type: Number, required: false, default: 100 },
  step: { type: Number, required: false, default: 1 },
  disabled: { type: Boolean, required: false, default: false },
  class: { type: [String, Object, Array], required: false, default: "" },
});

const emits = defineEmits(["update:modelValue"]);

// 处理滑块值变化
const handleInput = (event) => {
  const newValue = [...props.modelValue];
  newValue[0] = Number(event.target.value);
  emits("update:modelValue", newValue);
};
</script>

<template>
  <div :class="cn('w-full', props.class)">
    <input
      type="range"
      :min="props.min"
      :max="props.max"
      :step="props.step"
      :value="props.modelValue[0]"
      :disabled="props.disabled"
      @input="handleInput"
      class="w-full h-2 bg-primary/20 rounded-lg appearance-none cursor-pointer accent-primary focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
    />
  </div>
</template>
