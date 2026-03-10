import { mkdirSync } from 'node:fs';
import { writeFile, readdir, unlink, stat } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import * as path from 'node:path';
import type { Page, Response, Locator } from 'playwright';
import { logger } from './logger.js';
import { GEMINI_APP_URL } from './browser.js';

const SELECTORS = {
  promptInput: [
    '[data-testid="user-prompt-input-box"]',
    '.input-area textarea',
    'rich-textarea div[contenteditable]',
    'textarea',
  ] as const,
  submitButton: [
    'button[aria-label="Send message"]',
    'button[type="submit"]',
    '.send-button',
  ] as const,
  generatedImage:
    'img[src*="generativelanguage.googleapis.com"], img[src*="blob:"], .response-container img',
  conversationUrlPattern: /gemini\.google\.com\/app\/([a-zA-Z0-9_-]+)/,
} as const;

export const IMAGE_DIR = '/tmp/gemini-images';
export const CONVERSATION_ID_REGEX = /^[a-zA-Z0-9_-]+$/;
const DEFAULT_IMAGE_MIME_TYPE = 'image/webp';
export const LARGE_IMAGE_THRESHOLD = 400_000;
const IMAGE_MAX_AGE_MS = 60 * 60 * 1000; // 1 hour

mkdirSync(IMAGE_DIR, { recursive: true });

/** Delete temp images older than IMAGE_MAX_AGE_MS (fire-and-forget). */
export function pruneImageDir(): void {
  const cutoff = Date.now() - IMAGE_MAX_AGE_MS;
  readdir(IMAGE_DIR)
    .then((files) =>
      Promise.all(
        files.map(async (f) => {
          const fp = path.join(IMAGE_DIR, f);
          const s = await stat(fp).catch(() => null);
          if (s && s.mtimeMs < cutoff) await unlink(fp).catch(() => {});
        }),
      ),
    )
    .catch((err) => { logger.warn('pruneImageDir failed', err); });
}

export interface ImageResult {
  buffer: Buffer | null;
  filePath: string | null;
  mimeType: string;
  conversationId: string;
}

export async function ensureOnGemini(page: Page): Promise<void> {
  const url = page.url();
  if (!url.includes('gemini.google.com')) {
    logger.info('Navigating to Gemini', { currentUrl: url });
    await page.goto('https://gemini.google.com', { waitUntil: 'domcontentloaded' });
  }
}

/** Try each selector in order; return the first Locator that exists (count > 0). */
async function findFirstLocator(
  page: Page,
  selectorList: readonly string[],
): Promise<Locator | null> {
  for (const selector of selectorList) {
    try {
      const locator = page.locator(selector).first();
      if ((await locator.count()) > 0) return locator;
    } catch (e) {
      logger.debug('Selector evaluation failed, trying next', { selector, error: e instanceof Error ? e.message : String(e) });
    }
  }
  return null;
}

export async function fillPrompt(page: Page, prompt: string): Promise<void> {
  const locator = await findFirstLocator(page, SELECTORS.promptInput);
  if (locator) {
    await locator.fill(prompt);
    logger.debug('Filled prompt via selector');
    return;
  }
  await page.keyboard.type(prompt);
  logger.debug('Filled prompt via keyboard.type fallback');
}

export async function submitPrompt(page: Page): Promise<void> {
  const locator = await findFirstLocator(page, SELECTORS.submitButton);
  if (locator) {
    await locator.click();
    logger.debug('Submitted via button click');
    return;
  }
  await page.keyboard.press('Enter');
  logger.debug('Submitted via Enter key fallback');
}

export function armImageCapture(
  page: Page,
): { waitForImage(timeoutMs: number): Promise<ImageResult> } {
  const imageFallbackLocator = page.locator(SELECTORS.generatedImage).first();

  return {
    async waitForImage(timeoutMs: number): Promise<ImageResult> {
      return waitForImage(page, imageFallbackLocator, timeoutMs);
    },
  };
}

async function waitForImage(
  page: Page,
  imageFallbackLocator: Locator,
  timeoutMs: number,
): Promise<ImageResult> {
  let response: Response;
  try {
    response = await page.waitForResponse(
      (r: Response) => {
        const contentType = r.headers()['content-type'] ?? '';
        return contentType.startsWith('image/');
      },
      { timeout: timeoutMs },
    );
  } catch (err) {
    // timeout or no response — fall back to DOM
    logger.warn('Network response not captured, falling back to DOM image', { err });
    await imageFallbackLocator.waitFor({ state: 'visible', timeout: timeoutMs });
    const src = await imageFallbackLocator.getAttribute('src');
    logger.info('DOM fallback image located', { src: src?.slice(0, 80) });
    const conversationId = extractConversationId(page);
    return { buffer: null, filePath: null, mimeType: DEFAULT_IMAGE_MIME_TYPE, conversationId };
  }

  const conversationId = extractConversationId(page);
  const rawContentType = response.headers()['content-type'] ?? DEFAULT_IMAGE_MIME_TYPE;
  const mimeType = (rawContentType.split(';')[0] ?? DEFAULT_IMAGE_MIME_TYPE).trim();
  const bodyBuffer = await response.body();
  const buffer = Buffer.from(bodyBuffer);

  logger.debug('Image response received', { bytes: buffer.byteLength, mimeType });

  if (buffer.byteLength > LARGE_IMAGE_THRESHOLD) {
    const ext = mimeType.includes('png') ? 'png' : mimeType.includes('jpeg') ? 'jpg' : 'webp';
    const filePath = path.join(IMAGE_DIR, randomUUID() + '.' + ext);
    await writeFile(filePath, buffer);
    logger.info('Large image written to file', { filePath, bytes: buffer.byteLength });
    pruneImageDir();
    return { buffer: null, filePath, mimeType, conversationId };
  }

  logger.debug('Small image returned as buffer', { bytes: buffer.byteLength });
  return { buffer, filePath: null, mimeType, conversationId };
}

export function extractConversationId(page: Page): string {
  const match = SELECTORS.conversationUrlPattern.exec(page.url());
  if (match?.[1]) {
    return match[1];
  }
  return `fallback-${randomUUID()}`;
}

export async function navigateToConversation(
  page: Page,
  conversationId: string,
): Promise<void> {
  if (!CONVERSATION_ID_REGEX.test(conversationId)) {
    throw new Error(`Invalid conversationId: ${conversationId}`);
  }
  const url = `https://gemini.google.com/app/${conversationId}`;
  logger.info('Navigating to conversation', { url });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
}

export async function queryImage(
  page: Page,
  prompt: string,
  timeoutMs: number,
): Promise<ImageResult> {
  await ensureOnGemini(page);
  // CRITICAL: arm capture BEFORE submitting
  const capture = armImageCapture(page);
  await fillPrompt(page, prompt);
  await submitPrompt(page);
  return capture.waitForImage(timeoutMs);
}

export async function iterateImage(
  page: Page,
  conversationId: string,
  prompt: string,
  timeoutMs: number,
): Promise<ImageResult> {
  await navigateToConversation(page, conversationId);
  // CRITICAL: arm capture BEFORE submitting
  const capture = armImageCapture(page);
  await fillPrompt(page, prompt);
  await submitPrompt(page);
  return capture.waitForImage(timeoutMs);
}

/**
 * Poll the page URL until it contains GEMINI_APP_URL (login confirmed) or
 * the deadline is reached. Returns true if login was detected, false on timeout.
 */
export async function waitForLogin(page: Page, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  const pollIntervalMs = 2000;
  while (Date.now() < deadline) {
    if (page.url().includes(GEMINI_APP_URL)) {
      return true;
    }
    await new Promise<void>((resolve) => setTimeout(resolve, pollIntervalMs));
  }
  return false;
}
