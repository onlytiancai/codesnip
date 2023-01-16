export function parse(text) {
    if (!text) return { tags: [], data: [] }
    var tags = new Set(), data = []
    var lines = text.split('\n').map(x => x.trim()).filter(x => x.length > 0)
    for (const line of lines) {
        const reg = /\[(\w+)\]/g
        let reResult = null, lineTags = new Set();
        while ((reResult = reg.exec(line)) !== null) {
            const tag = reResult[1];
            tags.add(tag)
            lineTags.add(tag)
        }                
        data.push({
            tags: Array.from(lineTags),
            text: line.replace(reg, '')
        })
    }
    return {
        tags: Array.from(tags),
        data: data
    }
}