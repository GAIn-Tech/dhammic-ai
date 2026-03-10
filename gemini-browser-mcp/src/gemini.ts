import { mkdirSync } from 'fs';
import { writeFile, readdir, unlink, stat } from 'node:fs/promises';
import { randomUUID } from 'crypto';
import type { Page, Response } from 'playwright';
import { logger } from './logger.js';

const SELECTORS = {
  promptInput:
    '[data-testid="user-prompt-input-box"], .input-area textarea, rich-textarea div[contenteditable], textarea',
  submitButton:
    'button[aria-label="Send message"], button[type="submit"], .send-button',
  generatedImage:
    'img[src*="generativelanguage.googleapis.com"], img[src*="blob:"], .response-container img',
  conversationUrlPattern: /gemini\.google\.com\/app\/([a-zA-Z0-9_-]+)/,
} as const;

const IMAGE_DIR = '/tmp/gemini-images';
const LARGE_IMAGE_THRESHOLD = 400_000;
const IMAGE_MAX_AGE_MS = 60 * 60 * 1000; // 1 hour

mkdirSync(IMAGE_DIR, { recursive: true });

/** Delete temp images older than IMAGE_MAX_AGE_MS (fire-and-forget). */
export function pruneImageDir(): void {
  const cutoff = Date.now() - IMAGE_MAX_AGE_MS;
  readdir(IMAGE_DIR)
    .then((files) =>
      Promise.all(
        files.map(async (f) => {
          const fp = `${IMAGE_DIR}/${f}`;
          const s = await stat(fp).catch(() => null);
          if (s && s.mtimeMs < cutoff) await unlink(fp).catch(() => {});
        }),
      ),
    )
    .catch(() => {});
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

export async function fillPrompt(page: Page, prompt: string): Promise<void> {
  const selectors = SELECTORS.promptInput.split(', ');
  for (const selector of selectors) {
    try {
      const locator = page.locator(selector).first();
      const count = await locator.count();
      if (count > 0) {
        await locator.fill(prompt);
        logger.debug('Filled prompt via selector', { selector });
        return;
      }
    } catch {
      // try next selector
    }
  }
  // fallback: type into whatever is focused
  await page.keyboard.type(prompt);
  logger.debug('Filled prompt via keyboard.type fallback');
}

export async function submitPrompt(page: Page): Promise<void> {
  const submitSelectors = SELECTORS.submitButton.split(', ');
  for (const selector of submitSelectors) {
    try {
      const locator = page.locator(selector).first();
      const count = await locator.count();
      if (count > 0) {
        await locator.click();
        logger.debug('Submitted via button click', { selector });
        return;
      }
    } catch {
      // try next selector
    }
  }
  // fallback: press Enter
  await page.keyboard.press('Enter');
  logger.debug('Submitted via Enter key fallback');
}

export function armImageCapture(
  page: Page,
): { waitForImage(timeoutMs: number): Promise<ImageResult> } {
  // Register listener BEFORE submitting so we never miss a fast response
  const responsePromise = page.waitForResponse(
    (response: Response) => {
      const url = response.url();
      const contentType = response.headers()['content-type'] ?? '';
      return (
        url.includes('generativelanguage.googleapis.com') ||
        contentType.startsWith('image/')
      );
    },
  );

  // DOM fallback locator (resolved lazily inside waitForImage)
  const imageFallbackLocator = page.locator(SELECTORS.generatedImage).first();

  return {
    async waitForImage(timeoutMs: number): Promise<ImageResult> {
      return waitForImage(page, responsePromise, imageFallbackLocator, timeoutMs);
    },
  };
}

async function waitForImage(
  page: Page,
  responsePromise: Promise<Response>,
  imageFallbackLocator: ReturnType<Page['locator']>,
  timeoutMs: number,
): Promise<ImageResult> {
  const conversationId = extractConversationId(page);

  let response: Response;
  try {
    response = await Promise.race([
      responsePromise,
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Image response timeout')), timeoutMs),
      ),
    ]);
  } catch (err) {
    logger.warn('Network response not captured, falling back to DOM image', { err });

    // DOM fallback: wait for an img element to appear
    await imageFallbackLocator.waitFor({ state: 'visible', timeout: timeoutMs });
    const src = await imageFallbackLocator.getAttribute('src');
    logger.info('DOM fallback image located', { src: src?.slice(0, 80) });

    return {
      buffer: null,
      filePath: null,
      mimeType: 'image/webp',
      conversationId,
    };
  }

  const mimeType = response.headers()['content-type'] ?? 'image/webp';
  const bodyBuffer = await response.body();
  const buffer = Buffer.from(bodyBuffer);

  logger.debug('Image response received', { bytes: buffer.byteLength, mimeType });

  if (buffer.byteLength > LARGE_IMAGE_THRESHOLD) {
    const ext = mimeType.includes('png') ? 'png' : mimeType.includes('jpeg') ? 'jpg' : 'webp';
    const filePath = `${IMAGE_DIR}/${randomUUID()}.${ext}`;
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
  return `fallback-${Date.now()}`;
}

export async function navigateToConversation(
  page: Page,
  conversationId: string,
): Promise<void> {
  if (!/^[a-zA-Z0-9_-]+$/.test(conversationId)) {
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
