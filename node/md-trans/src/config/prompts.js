export const DEFAULT_TRANSLATION_PROMPT = `## Role
You are a professional translator. Rewrite the content in Chinese, don't mechanically translate.

## Translation Principles
1. Rewrite in natural Chinese
2. Maintain technical accuracy
3. Preserve tone and style

## Context
{{context}}

## Glossary
{{glossary}}

## Content
{{text}}

## Output
Only Chinese translation, no explanations.`;

export const PREANALYSIS_PROMPT = `## Role
You are a professional translator specializing in technical documentation.

## Task
Analyze the following English text and extract:
1. **Glossary**: Technical terms and proper nouns that should be translated consistently
2. **Context**: Brief summary of the document's topic and purpose
3. **Warnings**: Metaphors, cultural references, or idioms that may need special handling

## Input Text
{{text}}

## Output Format
Provide a JSON object with the following structure:
{
  "glossary": [
    {"term": "API", "translation": "API", "note": "Technical term, keep in English"},
    {"term": "GitHub", "translation": "GitHub", "note": "Proper noun"}
  ],
  "context": "Brief context summary",
  "warnings": [
    {"text": "break a leg", "type": "idiom", "suggestion": "祝好运"}
  ]
}

## Output
JSON only, no explanations.`;

export function renderPrompt(template, variables) {
  let rendered = template;
  for (const [key, value] of Object.entries(variables)) {
    const placeholder = `{{${key}}}`;
    if (typeof value === 'object') {
      rendered = rendered.replace(placeholder, JSON.stringify(value, null, 2));
    } else {
      rendered = rendered.replace(placeholder, value || '');
    }
  }
  return rendered;
}

export function createTranslationPrompt(text, glossary = [], context = '') {
  return renderPrompt(DEFAULT_TRANSLATION_PROMPT, {
    text,
    glossary: glossary.map(g => `${g.term}: ${g.translation}`).join('\n'),
    context,
  });
}

export function createPreanalysisPrompt(text) {
  return renderPrompt(PREANALYSIS_PROMPT, { text });
}

export default {
  DEFAULT_TRANSLATION_PROMPT,
  PREANALYSIS_PROMPT,
  renderPrompt,
  createTranslationPrompt,
  createPreanalysisPrompt,
};
