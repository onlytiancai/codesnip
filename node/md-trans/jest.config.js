import { resolve } from 'path';
import { fileURLToPath } from 'url';

const rootDir = resolve(fileURLToPath(import.meta.url), '..');

export default {
  testEnvironment: 'node',
  transform: {},
  moduleFileExtensions: ['js', 'mjs'],
  testMatch: ['**/tests/**/*.test.js'],
  verbose: true,
  rootDir: rootDir,
  roots: [rootDir],
  testPathIgnorePatterns: ['/node_modules/'],
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/cli.js',
  ],
};
