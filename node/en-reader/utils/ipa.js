// utils/ipa.js - IPA loading and lookup utilities

// Offline IPA map will be loaded from `config/offlineIPA.csv` at runtime
const offlineIPA = {};

// Load offline IPA from CSV file
export async function loadOfflineIPA(isLoadingIPA) {
  isLoadingIPA.value = true;
  try {
    const res = await fetch('./config/offlineIPA.csv', { cache: 'no-store' });
    if (!res.ok) throw new Error('offlineIPA CSV fetch failed');
    const text = await res.text();
    // Parse simple CSV: first header row (word,ipa), subsequent rows word,ipa
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#'));
    // Skip header if present
    const startIdx = (lines[0] && lines[0].toLowerCase().startsWith('word,')) ? 1 : 0;
    for (let i = startIdx; i < lines.length; i++) {
      // Handle quoted CSV fields
      const parts = lines[i].split(',').map(part => {
        // Remove quotes from both sides
        return part.trim().replace(/^"|"$/g, '');
      });
      const w = (parts[0] || '').trim().toLowerCase();
      const p = (parts.slice(1).join(',') || '').trim();
      if (w) offlineIPA[w] = p || null;
    }
    console.log('IPA规则加载完成，共加载', Object.keys(offlineIPA).length, '个单词的IPA信息');
  } catch (e) {
    console.warn('Could not load config/offlineIPA.csv, continuing with built-in empty map.', e);
  } finally {
    isLoadingIPA.value = false;
  }
}

// Query IPA for a word
export async function fetchIPA(word) {
  if (!word) return null;
  
  // First try direct lookup
  let ipa = offlineIPA[word];
  if (ipa) return ipa;
  
  // If not found, try removing common suffixes
  const suffixes = [
    // Third person singular
    { suffix: 'es', length: 2 },
    { suffix: 's', length: 1 },
    // Past tense/past participle
    { suffix: 'ed', length: 2 },
    { suffix: 'ing', length: 3 },
    // Comparative/superlative
    { suffix: 'er', length: 2 },
    { suffix: 'est', length: 3 },
    // Noun plural
    { suffix: 'ies', length: 3, replace: 'y' },
    { suffix: 'ves', length: 3, replace: 'f' },
    { suffix: 'ves', length: 3, replace: 'fe' },
  ];
  
  for (let suffix of suffixes) {
    if (word.endsWith(suffix.suffix)) {
      if (suffix.replace) {
        const baseWord = word.slice(0, -suffix.length) + suffix.replace;
        ipa = offlineIPA[baseWord];
        if (ipa) return ipa;
      } else {
        const baseWord = word.slice(0, -suffix.length);
        ipa = offlineIPA[baseWord];
        if (ipa) return ipa;
      }
    }
  }
  
  // Handle special case: -ly suffix
  if (word.endsWith('ly')) {
    const baseWord = word.slice(0, -2);
    ipa = offlineIPA[baseWord];
    if (ipa) return ipa;
  }
  
  // Handle special case: -ment suffix
  if (word.endsWith('ment')) {
    const baseWord = word.slice(0, -4);
    ipa = offlineIPA[baseWord];
    if (ipa) return ipa;
  }
  
  // Handle special case: -ful suffix
  if (word.endsWith('ful')) {
    const baseWord = word.slice(0, -3);
    ipa = offlineIPA[baseWord];
    if (ipa) return ipa;
  }
  
  // Handle special case: -able suffix
  if (word.endsWith('able')) {
    const baseWord = word.slice(0, -4);
    ipa = offlineIPA[baseWord];
    if (ipa) return ipa;
  }
  
  // Handle special case: -ible suffix
  if (word.endsWith('ible')) {
    const baseWord = word.slice(0, -4);
    ipa = offlineIPA[baseWord];
    if (ipa) return ipa;
  }
  
  // Handle special case: -ity suffix
  if (word.endsWith('ity')) {
    const baseWord = word.slice(0, -3) + 'e';
    ipa = offlineIPA[baseWord];
    if (ipa) return ipa;
  }
  
  return null;
}
