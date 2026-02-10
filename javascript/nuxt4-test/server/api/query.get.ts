// /api/query?foo=bar&baz=qux
export default defineEventHandler((event) => {
  const query = getQuery(event)

  return { a: query.foo, b: query.baz }
})