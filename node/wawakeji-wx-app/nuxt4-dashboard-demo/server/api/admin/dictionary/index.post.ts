// Admin API: Create a new dictionary word
import { z } from 'zod'
import { useDictionaryDb } from '../../../utils/dictionary'

const wordSchema = z.object({
  word: z.string().min(1).max(100),
  phonetic: z.string().optional(),
  definition: z.string().optional(),
  translation: z.string().optional(),
  pos: z.string().optional(),
  collins: z.number().int().min(1).max(5).optional(),
  oxford: z.number().int().min(0).max(1).optional(),
  tag: z.string().optional(),
  bnc: z.number().int().optional(),
  frq: z.number().int().optional(),
  exchange: z.string().optional()
})

export default defineEventHandler(async (event) => {
  const body = await readValidatedBody(event, wordSchema.parse)

  // Check if word already exists
  const existing = await useDictionaryDb().dictionary.findUnique({
    where: { word: body.word.toLowerCase() }
  })

  if (existing) {
    throw createError({
      statusCode: 400,
      message: `Word "${body.word}" already exists`
    })
  }

  const word = await useDictionaryDb().dictionary.create({
    data: {
      word: body.word.toLowerCase(),
      phonetic: body.phonetic || null,
      definition: body.definition || null,
      translation: body.translation || null,
      pos: body.pos || null,
      collins: body.collins || null,
      oxford: body.oxford || null,
      tag: body.tag || null,
      bnc: body.bnc || null,
      frq: body.frq || null,
      exchange: body.exchange || null
    }
  })

  return word
})