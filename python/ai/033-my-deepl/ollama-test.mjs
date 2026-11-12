import ollama from 'ollama'

const response = await ollama.chat({
  model: 's2021008840/hy-mt2:1.8b',
  messages: [{role: 'user', content: 'Hello!'}],
})
console.log(response.message.content)
