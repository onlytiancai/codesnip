// ============ Constants ============
export const TOOL_RESULT_MAX_DISPLAY = 100;

// ============ Error Handling ============
export function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `Error: ${error.message}`;
  }
  return `Error: ${String(error)}`;
}

// ============ Text Utilities ============
export function truncateText(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.substring(0, maxLen) + `... (${text.length} chars total)`;
}

export function escapeNewlines(text: string): string {
  return text.replace(/\n/g, '\\n');
}

export function formatToolResult(result: string, maxLen: number = TOOL_RESULT_MAX_DISPLAY): string {
  const escaped = escapeNewlines(result);
  if (escaped.length <= maxLen) return escaped;
  return truncateText(escaped, maxLen);
}
