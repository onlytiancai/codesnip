// 翻译服务
export class TranslateService {
  private static instance: TranslateService;
  private translationCache: Record<string, string> = {};
  
  private constructor() {}
  
  static getInstance(): TranslateService {
    if (!TranslateService.instance) {
      TranslateService.instance = new TranslateService();
    }
    return TranslateService.instance;
  }
  
  /**
   * 翻译文本
   * @param text 要翻译的文本
   * @param targetLang 目标语言，默认为中文
   * @returns 翻译结果
   */
  async translate(text: string, targetLang: string = 'zh-CN'): Promise<string> {
    // 检查缓存
    const cacheKey = `${text}_${targetLang}`;
    if (this.translationCache[cacheKey]) {
      return this.translationCache[cacheKey];
    }
    
    try {
      // 使用谷歌翻译API
      const encodedText = encodeURIComponent(text);
      const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=${targetLang}&dt=t&q=${encodedText}`;
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`翻译API请求失败: ${response.status}`);
      }
      
      const data = await response.json();
      // Extract all translations and join them
      const translatedText = data[0].map((item: any) => item[0]).join(' ');
      
      // 存入缓存
      this.translationCache[cacheKey] = translatedText;
      
      return translatedText;
    } catch (error) {
      console.error('翻译失败:', error);
      throw new Error('谷歌翻译服务不可用，请检查网络连接');
    }
  }
  
  /**
   * 清空翻译缓存
   */
  clearCache(): void {
    this.translationCache = {};
  }
}

// 导出单例实例
export const translateService = TranslateService.getInstance();