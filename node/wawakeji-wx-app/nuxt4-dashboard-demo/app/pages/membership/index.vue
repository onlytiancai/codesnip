<template>
  <NuxtLayout name="default">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <!-- Header -->
      <div class="text-center mb-12">
        <UBadge color="warning" variant="subtle" class="mb-4">
          <UIcon name="i-lucide-crown" class="w-4 h-4 mr-1" />
          Premium Membership
        </UBadge>
        <h1 class="text-4xl font-bold mb-4">Unlock Your Full Potential</h1>
        <p class="text-lg text-gray-500 dark:text-gray-400 max-w-2xl mx-auto">
          Get unlimited access to all articles, vocabulary tools, and premium features to accelerate your English learning.
        </p>
      </div>

      <!-- Current Plan Status -->
      <ClientOnly>
        <div v-if="loggedIn && membership" class="max-w-md mx-auto mb-8">
          <UCard class="bg-primary/5 border-primary/20">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm text-gray-500 dark:text-gray-400">Current Plan</p>
                <p class="text-lg font-semibold capitalize">{{ membership.plan }}</p>
              </div>
              <UBadge :color="membership.plan === 'free' ? 'neutral' : 'primary'">
                {{ membership.plan === 'free' ? 'Free' : 'Premium' }}
              </UBadge>
            </div>
            <p v-if="membership.endDate" class="text-sm text-gray-500 mt-2">
              Valid until {{ formatDate(membership.endDate) }}
            </p>
          </UCard>
        </div>
      </ClientOnly>

      <!-- Pricing Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <!-- Free Plan -->
        <UCard>
          <template #header>
            <h3 class="text-lg font-semibold">Free</h3>
            <div class="mt-4">
              <span class="text-4xl font-bold">$0</span>
              <span class="text-gray-500 dark:text-gray-400">/month</span>
            </div>
          </template>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>5 articles per month</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Basic audio playback</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Reading progress tracking</span>
            </li>
            <li class="flex items-center gap-2 text-gray-400">
              <UIcon name="i-lucide-x" class="w-5 h-5" />
              <span>Limited vocabulary (10 words)</span>
            </li>
            <li class="flex items-center gap-2 text-gray-400">
              <UIcon name="i-lucide-x" class="w-5 h-5" />
              <span>Ads included</span>
            </li>
          </ul>
          <template #footer>
            <ClientOnly>
              <UButton
                block
                variant="outline"
                :loading="upgrading === 'free'"
                :disabled="membership?.plan === 'free'"
                @click="handleDowngrade"
              >
                {{ membership?.plan === 'free' ? 'Current Plan' : 'Downgrade' }}
              </UButton>
            </ClientOnly>
          </template>
        </UCard>

        <!-- Pro Plan -->
        <UCard class="ring-2 ring-primary relative overflow-visible">
          <div class="absolute -top-3 left-1/2 -translate-x-1/2 z-10">
            <UBadge color="primary">Most Popular</UBadge>
          </div>
          <template #header>
            <h3 class="text-lg font-semibold">Pro</h3>
            <div class="mt-4">
              <span class="text-4xl font-bold">¥68</span>
              <span class="text-gray-500 dark:text-gray-400">/month</span>
            </div>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Billed monthly
            </p>
          </template>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Unlimited articles</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>HD audio with variable speed</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Full vocabulary features</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Spaced repetition system</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>No ads</span>
            </li>
          </ul>
          <template #footer>
            <ClientOnly>
              <UButton
                block
                color="primary"
                :loading="paying === 'pro'"
                :disabled="membership?.plan === 'premium' || membership?.plan === 'annual'"
                @click="handlePayment('pro')"
              >
                {{ membership?.plan === 'premium' || membership?.plan === 'annual' ? 'Current Plan' : 'Upgrade to Pro' }}
              </UButton>
            </ClientOnly>
          </template>
        </UCard>

        <!-- Annual Plan -->
        <UCard>
          <template #header>
            <div class="flex items-center justify-between">
              <h3 class="text-lg font-semibold">Annual</h3>
              <UBadge color="success" variant="subtle">Save 40%</UBadge>
            </div>
            <div class="mt-4">
              <span class="text-4xl font-bold">¥468</span>
              <span class="text-gray-500 dark:text-gray-400">/year</span>
            </div>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
              ¥39/month, billed annually
            </p>
          </template>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Everything in Pro</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Priority support</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Early access to new features</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>Exclusive content</span>
            </li>
            <li class="flex items-center gap-2">
              <UIcon name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <span>2 months free</span>
            </li>
          </ul>
          <template #footer>
            <UButton
              block
              variant="outline"
              :loading="paying === 'annual'"
              :disabled="membership?.plan === 'annual'"
              @click="handlePayment('annual')"
            >
              {{ membership?.plan === 'annual' ? 'Current Plan' : 'Get Annual Plan' }}
            </UButton>
          </template>
        </UCard>
      </div>

      <!-- Payment Status Modal -->
      <UModal v-model:open="showPaymentModal" :ui="{ footer: 'justify-end' }">
        <template #title>
          <div class="flex items-center gap-2">
            <UIcon v-if="paymentStatus === 'processing'" name="i-lucide-loader-2" class="w-5 h-5 animate-spin" />
            <UIcon v-else-if="paymentStatus === 'success'" name="i-lucide-check-circle" class="w-5 h-5 text-green-500" />
            <UIcon v-else-if="paymentStatus === 'failed'" name="i-lucide-x-circle" class="w-5 h-5 text-red-500" />
            <span>{{ paymentModalTitle }}</span>
          </div>
        </template>
        <template #body>
          <p class="text-gray-500 dark:text-gray-400">{{ paymentModalMessage }}</p>
        </template>
        <template #footer>
          <UButton
            v-if="paymentStatus === 'success'"
            color="primary"
            @click="closePaymentModal"
          >
            Done
          </UButton>
          <UButton
            v-else-if="paymentStatus === 'failed'"
            @click="closePaymentModal"
          >
            Close
          </UButton>
        </template>
      </UModal>

      <!-- Feature Comparison -->
      <div class="mb-12">
        <h2 class="text-2xl font-bold text-center mb-6">Feature Comparison</h2>
        <UCard>
          <UTable :data="featureComparison" :columns="columns" class="w-full">
            <template #feature-cell="{ row }">
              <span class="font-medium">{{ row.original.feature }}</span>
            </template>
            <template #free-cell="{ row }">
              <UIcon v-if="row.original.free === true" name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <UIcon v-else-if="row.original.free === false" name="i-lucide-x" class="w-5 h-5 text-gray-300 dark:text-gray-600" />
              <span v-else class="text-sm">{{ row.original.free }}</span>
            </template>
            <template #pro-cell="{ row }">
              <UIcon v-if="row.original.pro === true" name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <UIcon v-else-if="row.original.pro === false" name="i-lucide-x" class="w-5 h-5 text-gray-300 dark:text-gray-600" />
              <span v-else class="text-sm">{{ row.original.pro }}</span>
            </template>
            <template #annual-cell="{ row }">
              <UIcon v-if="row.original.annual === true" name="i-lucide-check" class="w-5 h-5 text-green-500" />
              <UIcon v-else-if="row.original.annual === false" name="i-lucide-x" class="w-5 h-5 text-gray-300 dark:text-gray-600" />
              <span v-else class="text-sm">{{ row.original.annual }}</span>
            </template>
          </UTable>
        </UCard>
      </div>

      <!-- FAQ -->
      <div class="max-w-3xl mx-auto">
        <h2 class="text-2xl font-bold text-center mb-8">Frequently Asked Questions</h2>
        <UAccordion :items="faqItems" class="space-y-4" />
      </div>

      <!-- Trust Badges -->
      <div class="flex flex-wrap justify-center gap-8 mt-12 pt-12 border-t border-gray-200 dark:border-gray-700">
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-shield-check" class="w-5 h-5" />
          <span>Secure Payment</span>
        </div>
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-refresh-ccw" class="w-5 h-5" />
          <span>30-day Money Back</span>
        </div>
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-headphones" class="w-5 h-5" />
          <span>24/7 Support</span>
        </div>
        <div class="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <UIcon name="i-lucide-credit-card" class="w-5 h-5" />
          <span>Cancel Anytime</span>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
const { loggedIn, user } = useUserSession()
const toast = useToast()
const router = useRouter()

const upgrading = ref<string | null>(null)
const paying = ref<string | null>(null)
const membership = ref<any>(null)

// Payment status
const showPaymentModal = ref(false)
const paymentStatus = ref<'processing' | 'success' | 'failed'>('processing')
const currentOrderNo = ref<string | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

const paymentModalTitle = computed(() => {
  switch (paymentStatus.value) {
    case 'processing':
      return 'Processing Payment...'
    case 'success':
      return 'Payment Successful!'
    case 'failed':
      return 'Payment Failed'
    default:
      return ''
  }
})

const paymentModalMessage = computed(() => {
  switch (paymentStatus.value) {
    case 'processing':
      return 'Please complete the payment in WeChat. We will notify you once the payment is successful.'
    case 'success':
      return 'Your membership has been upgraded successfully! Enjoy your premium features.'
    case 'failed':
      return 'The payment was not completed. Please try again.'
    default:
      return ''
  }
})

// Fetch user's membership
watchEffect(async () => {
  if (loggedIn.value && user.value) {
    try {
      const profile = await $fetch('/api/user/profile')
      membership.value = profile.membership
    } catch (e) {
      // Ignore
    }
  }
})

// Check if running in WeChat browser
const isInWeChat = () => {
  if (typeof window === 'undefined') return false
  const ua = navigator.userAgent.toLowerCase()
  return ua.includes('micromessenger')
}

const handleDowngrade = async () => {
  if (!loggedIn.value) {
    toast.add({
      title: 'Please login',
      description: 'You need to login to change your membership',
      color: 'warning'
    })
    router.push('/login')
    return
  }

  upgrading.value = 'free'
  try {
    await $fetch('/api/user/membership', {
      method: 'POST',
      body: { plan: 'free' }
    })
    toast.add({
      title: 'Success',
      description: 'You have successfully downgraded to free plan',
      color: 'success'
    })
    // Refresh membership
    const profile = await $fetch('/api/user/profile')
    membership.value = profile.membership
  } catch (error: any) {
    toast.add({
      title: 'Error',
      description: error?.data?.message || 'Failed to downgrade membership. Please try again.',
      color: 'error'
    })
  } finally {
    upgrading.value = null
  }
}

const handlePayment = async (plan: string) => {
  if (!loggedIn.value) {
    toast.add({
      title: 'Please login',
      description: 'You need to login to upgrade your membership',
      color: 'warning'
    })
    router.push('/login')
    return
  }

  // Check if in WeChat browser
  if (!isInWeChat()) {
    toast.add({
      title: 'WeChat Required',
      description: 'Please open this page in WeChat to complete the payment',
      color: 'warning'
    })
    return
  }

  paying.value = plan
  paymentStatus.value = 'processing'
  showPaymentModal.value = true

  try {
    // Create order and get payment params
    const response = await $fetch('/api/orders', {
      method: 'POST',
      body: { plan }
    })

    if (!response.success || !response.paymentParams) {
      throw new Error('Failed to create order')
    }

    currentOrderNo.value = response.orderNo

    // Call WeChat JS-SDK to invoke payment
    const payParams = response.paymentParams

    // WeChat JS-SDK invoke payment
    // @ts-ignore
    if (typeof wx !== 'undefined' && wx.chooseWXPay) {
      // @ts-ignore
      wx.chooseWXPay({
        timestamp: payParams.timeStamp,
        nonceStr: payParams.nonceStr,
        package: payParams.package,
        signType: payParams.signType,
        paySign: payParams.paySign,
        success: () => {
          // Start polling for payment result
          startPolling()
        },
        cancel: () => {
          paymentStatus.value = 'failed'
          clearPollTimer()
        },
        fail: () => {
          paymentStatus.value = 'failed'
          clearPollTimer()
        }
      })
    } else {
      // If wx is not available, start polling directly
      // (user might complete payment in another way)
      startPolling()
    }
  } catch (error: any) {
    console.error('Payment error:', error)
    paymentStatus.value = 'failed'
    toast.add({
      title: 'Error',
      description: error?.data?.message || 'Failed to initiate payment. Please try again.',
      color: 'error'
    })
  } finally {
    paying.value = null
  }
}

const startPolling = () => {
  if (!currentOrderNo.value) return

  // Poll every 2 seconds for up to 60 times (2 minutes)
  let pollCount = 0
  const maxPolls = 60

  pollTimer = setInterval(async () => {
    pollCount++

    if (pollCount > maxPolls) {
      clearPollTimer()
      paymentStatus.value = 'failed'
      return
    }

    try {
      const order = await $fetch(`/api/orders/${currentOrderNo.value}`)

      if (order.status === 'paid') {
        clearPollTimer()
        paymentStatus.value = 'success'

        // Refresh membership
        const profile = await $fetch('/api/user/profile')
        membership.value = profile.membership
      } else if (order.status === 'failed') {
        clearPollTimer()
        paymentStatus.value = 'failed'
      }
    } catch (e) {
      console.error('Poll error:', e)
    }
  }, 2000)
}

const clearPollTimer = () => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

const closePaymentModal = () => {
  showPaymentModal.value = false
  clearPollTimer()
}

// Cleanup on unmount
onUnmounted(() => {
  clearPollTimer()
})

const formatDate = (date: string | Date) => {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

const columns = [
  { id: 'feature', header: 'Feature', size: 200 },
  { id: 'free', header: 'Free' },
  { id: 'pro', header: 'Pro' },
  { id: 'annual', header: 'Annual' }
]

const featureComparison = [
  { feature: 'Articles per month', free: '5', pro: 'Unlimited', annual: 'Unlimited' },
  { feature: 'Audio quality', free: 'Standard', pro: 'HD', annual: 'HD' },
  { feature: 'Audio playback speed', free: false, pro: true, annual: true },
  { feature: 'Vocabulary limit', free: '10 words', pro: 'Unlimited', annual: 'Unlimited' },
  { feature: 'Spaced repetition', free: false, pro: true, annual: true },
  { feature: 'Offline access', free: false, pro: true, annual: true },
  { feature: 'Reading progress sync', free: true, pro: true, annual: true },
  { feature: 'Bookmark articles', free: true, pro: true, annual: true },
  { feature: 'Highlight & notes', free: false, pro: true, annual: true },
  { feature: 'Vocabulary export', free: false, pro: true, annual: true },
  { feature: 'Dark mode', free: true, pro: true, annual: true },
  { feature: 'Ad-free experience', free: false, pro: true, annual: true },
  { feature: 'Priority support', free: false, pro: false, annual: true },
  { feature: 'Early access to features', free: false, pro: false, annual: true },
  { feature: 'Exclusive content', free: false, pro: false, annual: true }
]

const faqItems = [
  {
    label: 'Can I cancel my subscription anytime?',
    content: 'Yes, you can cancel your subscription at any time. Your access will continue until the end of your billing period.'
  },
  {
    label: 'What payment methods do you accept?',
    content: 'We accept WeChat Pay for now. More payment methods will be added in the future.'
  },
  {
    label: 'Is there a refund policy?',
    content: 'Yes, we offer a 30-day money-back guarantee. If you are not satisfied, contact us for a full refund.'
  },
  {
    label: 'Can I switch between plans?',
    content: 'Yes, you can upgrade your plan at any time. The new plan will take effect immediately after payment.'
  }
]
</script>