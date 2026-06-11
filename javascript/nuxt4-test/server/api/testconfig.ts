export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig(event)

  return { myTestConfig: config.myTestConfig }
})
