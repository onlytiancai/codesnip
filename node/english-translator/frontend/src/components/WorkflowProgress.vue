<script setup>
import { computed } from 'vue'

const props = defineProps({
  currentStep: String,
  mode: String,
  message: String
})

const steps = {
  fast: [
    { key: 'extract', label: 'Extract Content' },
    { key: 'translate', label: 'Translate' },
    { key: 'done', label: 'Complete' }
  ],
  normal: [
    { key: 'extract', label: 'Extract Content' },
    { key: 'analyze', label: 'Analyze' },
    { key: 'terminology', label: 'Terminology' },
    { key: 'prompt_gen', label: 'Generate Prompt' },
    { key: 'segment', label: 'Segment' },
    { key: 'translate', label: 'Translate' },
    { key: 'done', label: 'Complete' }
  ],
  fine: [
    { key: 'extract', label: 'Extract Content' },
    { key: 'analyze', label: 'Analyze' },
    { key: 'terminology', label: 'Terminology' },
    { key: 'prompt_gen', label: 'Generate Prompt' },
    { key: 'segment', label: 'Segment' },
    { key: 'translate', label: 'Translate' },
    { key: 'review', label: 'Review' },
    { key: 'revise', label: 'Revise' },
    { key: 'polish', label: 'Polish' },
    { key: 'done', label: 'Complete' }
  ]
}

const currentSteps = computed(() => {
  return steps[props.mode] || steps.normal
})

function getStepStatus(stepKey) {
  const stepIndex = currentSteps.value.findIndex(s => s.key === stepKey)
  const currentIndex = currentSteps.value.findIndex(s => s.key === props.currentStep)

  if (stepIndex < currentIndex) return 'completed'
  if (stepIndex === currentIndex) return 'current'
  return 'pending'
}

function getStepClass(status) {
  switch (status) {
    case 'completed':
      return 'bg-green-500'
    case 'current':
      return 'bg-primary-500 animate-pulse'
    default:
      return 'bg-gray-300'
  }
}

function getLineClass(status) {
  switch (status) {
    case 'completed':
      return 'bg-green-500'
    case 'current':
      return 'bg-primary-500'
    default:
      return 'bg-gray-300'
  }
}
</script>

<template>
  <div class="bg-white shadow rounded-lg p-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-gray-900">Translation Progress</h2>
      <span v-if="message" class="text-sm text-gray-500">{{ message }}</span>
    </div>

    <div class="flex items-center">
      <template v-for="(step, index) in currentSteps" :key="step.key">
        <!-- Step Circle -->
        <div class="flex flex-col items-center">
          <div
            :class="getStepClass(getStepStatus(step.key))"
            class="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium"
          >
            <svg v-if="getStepStatus(step.key) === 'completed'" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
            <span v-else>{{ index + 1 }}</span>
          </div>
          <span class="mt-2 text-xs text-gray-500 text-center whitespace-nowrap">
            {{ step.label }}
          </span>
        </div>

        <!-- Connector Line -->
        <div
          v-if="index < currentSteps.length - 1"
          :class="getLineClass(getStepStatus(step.key))"
          class="flex-1 h-1 mx-2"
        ></div>
      </template>
    </div>
  </div>
</template>