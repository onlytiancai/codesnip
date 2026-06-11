export default defineEventHandler((event) => {
  const id = Number.parseInt(event.context.params.id) as number

  if (!Number.isInteger(id)) {
    throw createError({
      status: 400,
      statusText: 'ID should be an integer',
    })
  }
  return 'All good'
})
