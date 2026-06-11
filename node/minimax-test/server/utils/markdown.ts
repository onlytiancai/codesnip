export function markdownToHtml(markdown: string): string {
  let html = markdown

  // Convert images BEFORE links (since images start with ! but are different)
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" />')

  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>')
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>')
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>')

  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')

  // Convert links (but not images which were already converted)
  html = html.replace(/(?<!<img[^>]*)\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')

  html = html.replace(/`(.+?)`/g, '<code>$1</code>')

  html = html.replace(/^\*(.+)$/gm, '<li>$1</li>')
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')

  html = html.replace(/\n\n/g, '</p><p>')
  html = `<p>${html}</p>`
  html = html.replace(/<p><(h[1-6]|ul|li|img|a|code|strong|em)/g, '<$1')
  html = html.replace(/<\/(h[1-6]|ul)><\/p>/g, '</$1>')

  return html
}
