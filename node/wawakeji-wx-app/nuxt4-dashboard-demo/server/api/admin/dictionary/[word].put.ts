// Admin API: Update a dictionary word
import { z } from 'zod'
import { useDictionaryDb } from '../../../utils/dictionary'

const wordSchema = z.object({
  phonetic: z.string().optional(),
  definition: z.string().optional(),
  translation: z.string().optional(),
  pos: z.string().optional(),
  collins: z.number().int().min(1).max(5).optional().nullable(),
  oxford: z.number().int().min(0).max(1).optional().nullable(),
  tag: z.string().optional(),
  bnc: z.number().int().optional().nullable(),
  frq: z.number().int().optional().nullable(),
  exchange: z.string().optional()
})

export default defineEventHandler(async (event) => {
  const word = getRouterParam(event, 'word') as string
  const body = await readValidatedBody(event, wordSchema.parse)

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

  const updated = await useDictionaryDb().dictionary.update({
    where: { word: word.toLowerCase() },
    data: {
      phonetic: body.phonetic ?? existing.phonetic,
      definition: body.definition ?? existing.definition,
      translation: body.translation ?? existing.translation,
      pos: body.pos ?? existing.pos,
      collins: body.collins ?? existing.collins,
      oxford: body.oxford ?? existing.oxford,
      tag: body.tag ?? existing.tag,
      bnc: body.bnc ?? existing.bnc,
      frq: body.frq ?? existing.frq,
      exchange: body.exchange ?? existing.exchange
    }
  })

  return updated
})