<template>
  <component
    :is="iconComponent"
    :class="classes"
    :aria-hidden="true"
  />
</template>

<script setup lang="ts">
import {
  CheckCircleIcon, ChevronLeftIcon, ClipboardIcon, ArrowDownIcon, ExclamationCircleIcon,
  EyeIcon, ArrowRightEndOnRectangleIcon, PlusIcon, QrCodeIcon, ArrowPathIcon, ShieldCheckIcon,
  TrashIcon, ArrowUpIcon, XCircleIcon, ClockIcon, UserIcon, KeyIcon, HomeIcon
} from '@heroicons/vue/24/outline'

const iconMap = {
  'check-circle': CheckCircleIcon, 'chevron-left': ChevronLeftIcon, 'copy': ClipboardIcon,
  'download': ArrowDownIcon, 'exclamation-circle': ExclamationCircleIcon, 'eye': EyeIcon,
  'logout': ArrowRightEndOnRectangleIcon, 'plus': PlusIcon, 'qr-code': QrCodeIcon, 'refresh': ArrowPathIcon,
  'shield': ShieldCheckIcon, 'trash': TrashIcon, 'upload': ArrowUpIcon, 'x-circle': XCircleIcon,
  'clock': ClockIcon, 'user': UserIcon, 'key': KeyIcon, 'home': HomeIcon
}

const props = defineProps({
  name: {
    type: String,
    required: true
  },
  size: {
    type: String,
    default: 'h-5 w-5'
  },
  color: {
    type: String,
    default: ''
  },
  class: {
    type: String,
    default: ''
  }
})

const iconComponent = iconMap[props.name as keyof typeof iconMap]
const classes = [
  props.size,
  props.color,
  props.class
].filter(Boolean).join(' ')
</script>