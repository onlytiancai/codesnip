// Admin API: Delete a dictionary word
import { useDictionaryDb } from '../../../utils/dictionary'

export default defineEventHandler(async (event) => {
  const word = getRouterParam(event, 'word') as string

  // Check if word exists
  const existing = await useDictionaryDb().dictionary.findUnique({
    where: { word: word.toLowerCase() }
  })

  if (!existing) {
    throw createError({
      statusCode: 404,
      message: `Word "${word}" not found`
    })
  }

  await useDictionaryDb().dictionary.delete({
    where: { word: word.toLowerCase() }
  })

  return { success: true, message: `Word "${word}" deleted` }
})