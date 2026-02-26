const { request, Agent } = require('undici');

// 创建支持重定向的 Agent
const agent = new Agent({
  maxRedirections: 5,
});

/**
 * 从 URL 抓取 HTML 内容
 * @param {string} url - 目标 URL
 * @returns {Promise<string>} HTML 内容
 */
async function fetchHtml(url) {
  try {
    const response = await request(url, {
      dispatcher: agent,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
      },
    });

    if (response.statusCode !== 200) {
      throw new Error(`HTTP ${response.statusCode}: Failed to fetch ${url}`);
    }

    const contentType = response.headers['content-type'] || '';
    if (!contentType.includes('text/html')) {
      console.warn(`Warning: Content-Type is '${contentType}', expected 'text/html'`);
    }

    const chunks = [];
    for await (const chunk of response.body) {
      chunks.push(chunk);
    }

    return Buffer.concat(chunks).toString('utf-8');
  } catch (error) {
    if (error.code === 'ENOTFOUND') {
      throw new Error(`Network error: Unable to resolve ${url}`);
    }
    if (error.code === 'ECONNREFUSED') {
      throw new Error(`Connection refused: Unable to connect to ${url}`);
    }
    throw error;
  }
}

module.exports = { fetchHtml };
