import Anthropic from '@anthropic-ai/sdk';
import { config } from 'dotenv';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
config({ path: resolve(__dirname, '.env') });

const baseURL = process.env.ANTHROPIC_BASE_URL || 'https://api.minimaxi.com/anthropic';
const apiKey = process.env.ANTHROPIC_API_KEY;
const model = process.env.ANTHROPIC_MODEL || 'MiniMax-M2.7';

const client = new Anthropic({
  baseURL,
  apiKey,
});

const content = `## Role
You are a professional translator. Rewrite the content in Chinese, don't mechanically translate.

## Translation Principles
1. Rewrite in natural Chinese
2. Maintain technical accuracy
3. Preserve tone and style

## Context


## Glossary


## Content
title: "obsidianmd/obsidian-clipper: Highlight and capture the web in your favorite browser. The official Web Clipper extension for Obsidian."
source: "https://github.com/obsidianmd/obsidian-clipper?tab=readme-ov-file"
author:
published:
created: 2026-04-08
description: "Highlight and capture the web in your favorite browser. The official Web Clipper extension for Obsidian. - obsidianmd/obsidian-clipper"
tags:"clippings"Obsidian Web Clipper helps you highlight and capture the web in your favorite browser. Anything you save is stored as durable Markdown files that you can read offline, and preserve for the long term.Get startedInstall the extension by downloading it from the official directory for your browser:Chrome Web Store for Chrome, Brave, Arc, Orion, and other Chromium-based browsers.Firefox Add-Ons for Firefox and Firefox Mobile.Safari Extensions for macOS, iOS, and iPadOS.Edge Add-Ons for Microsoft Edge.Use the extensionDocumentation is available on the Obsidian Help site, which covers how to use highlighting, templates, variables, filters, and more.ContributeTranslationsYou can help translate Web Clipper into your language. Submit your translation via pull request using the format found in the /_locales folder.Features and bug fixesSee the help wanted tag for issues where contributions are welcome.RoadmapIn no particular order:A separate icon for Web ClipperAnnotate highlightsTemplate directoryTemplate validationTemplate logic (if/for)Save images locally, added in Obsidian 1.8.0Translate UI into more languages — help is welcomedDevelopersTo build the extension:npm run buildThis will create three directories: for the Chromium version for the Firefox version for the Safari versionInstall the extension locallyFor Chromium browsers, such as Chrome, Brave, Edge, and Arc:Open your browser and navigate to Enable Developer modeClick Load unpacked and select the  directoryFor Firefox:Open Firefox and navigate to Click Load Temporary Add-onNavigate to the  directory and select the  fileIf you want to run the extension permanently you can do so with the Nightly or Developer versions of Firefox.Type  in the URL barIn the Search box type Double-click the preference, or right-click and select "Toggle", to set it to .Go to  > gear icon > Install Add-on From File…For iOS Simulator testing on macOS:Run  to build the extensionOpen  in XcodeSelect the Obsidian Web Clipper (iOS) scheme from the scheme selectorChoose an iOS Simulator device and click Run to build and launch the appOnce the app is running on the simulator, open SafariNavigate to a webpage and tap the Extensions button in Safari to access the Web Clipper extensionRun testsnpm testOr run in watch mode during development:npm run test:watchThird-party librarieswebextension-polyfill for browser compatibilitydefuddle for content extraction and Markdown conversiondayjs for date parsing and formattinglz-string to compress templates to reduce storage spacelucide for iconsdompurify for sanitizing HTML

## Output
Only Chinese translation, no explanations.`;

console.log('=== Request ===');
console.log(`URL: ${baseURL}`);
console.log(`Model: ${model}`);
console.log(`Content length: ${content.length} chars`);

async function main() {
  try {
    console.log('\n=== Response ===');
    const message = await client.messages.create({
      model,
      max_tokens: 2000,
      system: 'You are a professional translator.',
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: content,
            },
          ],
        },
      ],
    });

    console.log('Id:', message.id);
    console.log('Model:', message.model);
    console.log('Type:', message.type);
    console.log('Role:', message.role);
    console.log('Content:', JSON.stringify(message.content, null, 2));

    for (const block of message.content) {
      if (block.type === 'thinking') {
        console.log(`\n[Thinking]\n${block.thinking}\n`);
      } else if (block.type === 'text') {
        console.log(`\n[Text]\n${block.text}\n`);
      }
    }
  } catch (error) {
    console.error('\n=== Error ===');
    console.error('Message:', error.message);
    console.error('Status:', error.status);
    console.error('Response:', JSON.stringify(error.response, null, 2));
  }
}

main();
