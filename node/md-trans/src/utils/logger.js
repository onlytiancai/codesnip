import picocolors from 'picocolors';

class Logger {
  constructor(options = {}) {
    this.debugMode = options.debug || false;
    this.prefix = options.prefix || '';
  }

  debug(message, ...args) {
    if (this.debugMode) {
      console.error(picocolors.gray(`[DEBUG${this.prefix ? ' ' + this.prefix : ''}] ${message}`), ...args);
    }
  }

  info(message, ...args) {
    console.log(picocolors.blue(`[INFO${this.prefix ? ' ' + this.prefix : ''}] ${message}`), ...args);
  }

  warn(message, ...args) {
    console.warn(picocolors.yellow(`[WARN${this.prefix ? ' ' + this.prefix : ''}] ${message}`), ...args);
  }

  error(message, ...args) {
    console.error(picocolors.red(`[ERROR${this.prefix ? ' ' + this.prefix : ''}] ${message}`), ...args);
  }

  success(message, ...args) {
    console.log(picocolors.green(`[SUCCESS] ${message}`), ...args);
  }

  setDebug(enabled) {
    this.debugMode = enabled;
  }
}

export function createLogger(options = {}) {
  return new Logger(options);
}

export default Logger;
