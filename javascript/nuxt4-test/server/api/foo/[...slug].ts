// const res = await fetch('/api/submit',{method: 'POST',body:JSON.stringify({a:1,b:2})})
// await res.json()
export default defineEventHandler((event) => {
  const slug = getRouterParam(event, 'slug')
  return `Default foo handler: ${slug}`
})
