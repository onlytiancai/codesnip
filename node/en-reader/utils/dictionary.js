// utils/dictionary.js - Dictionary lookup utilities

// Global variables to cache index
let cachedIndex = null;
const dbUrl = "./test/dict.db";

// HTTP Range request function
async function httpRange(url, start, end) {
  const res = await fetch(url, {
    headers: { Range: `bytes=${start}-${end - 1}` }
  });
  return new Uint8Array(await res.arrayBuffer());
}

// Load dictionary index
async function loadIndex(url) {
  console.log('Loading dictionary index from', url);
  // Read first 4 bytes = N
  const header = await httpRange(url, 0, 4);
  const N = new DataView(header.buffer).getUint32(0, true);
  console.log('Found', N, 'items in dictionary database');

  // IndexTable = 8 * N bytes (keyHash + offset)
  const indexBytes = await httpRange(url, 4, 4 + 8 * N);
  const dv = new DataView(indexBytes.buffer);

  // Calculate data area offset: 4 bytes (N) + 8*N bytes (index table)
  const dataAreaOffset = 4 + 8 * N;
  console.log('Data area offset:', dataAreaOffset);

  const index = [];
  for (let i = 0; i < N; i++) {
    const h = dv.getUint32(i * 8, true);
    const off = dv.getUint32(i * 8 + 4, true) + dataAreaOffset; // Add data area offset
    index.push({ h, off });
  }
  console.log('Loaded index with adjusted offsets:', index);
  return index;
}

// Simple 32-bit hash function
function hash32(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++)
    h = (h * 31 + str.charCodeAt(i)) >>> 0;
  return h;
}

// Load index once and cache it
async function loadAndCacheIndex() {
  if (!cachedIndex) {
    console.log("Loading dictionary index...");
    cachedIndex = await loadIndex(dbUrl);
    console.log("Index loaded with", cachedIndex.length, "items");
  }
  return cachedIndex;
}

// Lookup a word in the dictionary
export async function lookupWord(word) {
  if (!word || word.trim() === '') {
    return null;
  }
  
  const index = await loadAndCacheIndex();
  const h = hash32(word.trim().toLowerCase());
  
  // Binary search for keyHash in index
  let lo = 0, hi = index.length - 1, pos = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (index[mid].h === h) { 
      pos = mid; 
      break; 
    }
    if (index[mid].h < h) lo = mid + 1;
    else hi = mid - 1;
  }
  
  if (pos < 0) {
    console.log(`Word "${word}" not found in dictionary`);
    return null;
  }

  const start = index[pos].off;
  const end = (pos + 1 < index.length) ? index[pos + 1].off : null; // last one: load until EOF
  
  const rangeEndHeader = end !== null ? end : "";
  const rangeHeader = `bytes=${start}-${rangeEndHeader ? rangeEndHeader - 1 : ""}`;
  
  // Read value block
  const res = await fetch(dbUrl, {
    headers: { Range: rangeHeader }
  });
  
  const buf = new Uint8Array(await res.arrayBuffer());
  const value = new TextDecoder().decode(buf);
  
  try {
    return JSON.parse(value);
  } catch (error) {
    console.error("Failed to parse dictionary entry for", word, error);
    return null;
  }
}
