import { chromium, type BrowserContext, type Page } from 'playwright';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { logger } from './logger.js';

// Module-level state (no class)
let context: BrowserContext | null = null;
let initPromise: Promise<BrowserContext> | null = null;

const BASE_PROFILE_DIR = path.join(os.homedir(), '.gemini-mcp');

function sanitizeProfileDir(input?: string): string {
  const base = BASE_PROFILE_DIR;
  if (!input) {
    return path.join(base, 'profile');
  }
  const resolved = path.resolve(input);
  if (!resolved.startsWith(base + path.sep) && resolved !== base) {
    throw new Error(
      `Profile dir "${resolved}" is outside allowed base "${base}"`
    );
  }
  return resolved;
}

// Honour GEMINI_MCP_PROFILE_DIR env var; fall back to default on invalid value.
let _profileDir: string = (() => {
  const envDir = process.env['GEMINI_MCP_PROFILE_DIR'];
  if (envDir) {
    try {
      return sanitizeProfileDir(envDir);
    } catch {
      // Invalid env var — use default silently (logged at first browser init)
    }
  }
  return path.join(BASE_PROFILE_DIR, 'profile');
})();

async function initBrowser(): Promise<BrowserContext> {
  logger.info('Initializing browser', { profileDir: _profileDir });

  await fs.mkdir(_profileDir, { recursive: true, mode: 0o700 });

  const browser = await chromium.launchPersistentContext(_profileDir, {
    channel: 'chrome',
    headless: false,
    ignoreDefaultArgs: ['--enable-automation'],
    args: [
      '--disable-blink-features=AutomationControlled',
      '--disable-extensions',
      '--disable-component-extensions-with-background-pages',
      '--disable-default-apps',
      '--no-first-run',
      '--disable-dev-shm-usage',
      '--no-sandbox',
      '--disk-cache-size=67108864',
      '--media-cache-size=33554432',
    ],
  });

  const ctx = browser;

  // Stealth init script
  await ctx.addInitScript(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en'],
    });
  });

  /** Attach navigation allowlist to a single page (main frame only). */
  function attachAllowlist(page: import('playwright').Page): void {
    page.on('framenavigated', (frame) => {
      // Only enforce on top-level navigations to avoid disrupting iframes.
      if (frame !== frame.page().mainFrame()) return;
      const urlStr = frame.url();
      if (urlStr === 'about:blank' || urlStr.startsWith('chrome://')) return;
      try {
        const url = new URL(urlStr);
        const allowed = ['gemini.google.com', 'accounts.google.com'];
        if (!allowed.includes(url.hostname)) {
          logger.warn('Navigation blocked by allowlist', { url: urlStr });
          frame.page().goBack().catch(() => {});
        }
      } catch {
        /* ignore invalid URLs */
      }
    });
  }

  // Cover the initial page that already exists before 'page' events fire.
  for (const p of ctx.pages()) attachAllowlist(p);
  // Cover future pages (e.g. pop-ups).
  ctx.on('page', attachAllowlist);

  // Reset state when context is closed externally
  ctx.on('close', () => {
    context = null;
    initPromise = null;
    logger.info('Browser context closed, state reset');
  });

  logger.info('Browser initialized');
  return ctx;
}

async function ensureContext(): Promise<BrowserContext> {
  if (context) return context;

  // Assigned SYNCHRONOUSLY before first await to prevent concurrent dual-init
  initPromise ??= initBrowser();
  try {
    context = await initPromise;
  } catch (e) {
    initPromise = null; // allow retry on next call
    throw e;
  }
  return context;
}

export async function getPage(): Promise<Page> {
  const ctx = await ensureContext();
  const pages = ctx.pages();
  if (pages.length > 0) {
    return pages[0]!;
  }
  return ctx.newPage();
}

export async function closeBrowser(): Promise<void> {
  if (context) {
    logger.info('Closing browser context');
    await context.close();
    context = null;
    initPromise = null;
  }
}

export async function isLoggedIn(): Promise<boolean> {
  try {
    const page = await getPage();
    const url = page.url();
    if (url.includes('gemini.google.com/app')) {
      return true;
    }
    await page.goto('https://gemini.google.com/app', {
      waitUntil: 'domcontentloaded',
      timeout: 15000,
    });
    const finalUrl = page.url();
    return finalUrl.includes('gemini.google.com/app');
  } catch (e) {
    logger.warn('isLoggedIn check failed', e);
    return false;
  }
}

export function getProfileDir(): string {
  return _profileDir;
}

// Allow overriding profile dir at startup (before first init)
export function setProfileDir(input?: string): void {
  if (context || initPromise) {
    throw new Error('Cannot change profile dir after browser has been initialized');
  }
  _profileDir = sanitizeProfileDir(input);
}

export { sanitizeProfileDir };

/** Returns true if the browser context is currently active (no side effects) */
export function isBrowserRunning(): boolean {
  return context !== null;
}
