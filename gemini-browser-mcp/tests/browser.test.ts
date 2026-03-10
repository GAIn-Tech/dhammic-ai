import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as os from 'node:os';
import * as path from 'node:path';

// ---------------------------------------------------------------------------
// Mocks must be declared BEFORE any imports from the module under test.
// vi.mock() calls are hoisted to the top of the file by vitest.
// ---------------------------------------------------------------------------

vi.mock('playwright', () => ({
  chromium: {
    launchPersistentContext: vi.fn(),
  },
}));

vi.mock('node:fs/promises', () => ({
  mkdir: vi.fn().mockResolvedValue(undefined),
}));

// We also need to mock the logger so it doesn't write to stderr during tests.
vi.mock('../src/logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import {
  sanitizeProfileDir,
  getProfileDir,
  setProfileDir,
  isBrowserRunning,
  withPage,
  checkLoginStatus,
} from '../src/browser.js';

import { chromium } from 'playwright';

// ---------------------------------------------------------------------------
// File-level constants
// ---------------------------------------------------------------------------

const base = path.join(os.homedir(), '.gemini-mcp');

// ---------------------------------------------------------------------------
// Shared fake browser objects used in multiple test groups
// ---------------------------------------------------------------------------

const fakePage = {
  url: vi.fn().mockReturnValue('https://gemini.google.com/app/test'),
  on: vi.fn(),
  goBack: vi.fn().mockResolvedValue(null),
};

const fakeContext = {
  pages: vi.fn().mockReturnValue([fakePage]),
  newPage: vi.fn().mockResolvedValue(fakePage),
  addInitScript: vi.fn().mockResolvedValue(undefined),
  on: vi.fn(),
  close: vi.fn().mockResolvedValue(undefined),
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resetFakeMocks(): void {
  vi.clearAllMocks();
  fakePage.url.mockReturnValue('https://gemini.google.com/app/test');
  fakePage.on.mockReset();
  fakePage.goBack.mockResolvedValue(null);
  fakeContext.pages.mockReturnValue([fakePage]);
  fakeContext.newPage.mockResolvedValue(fakePage);
  fakeContext.addInitScript.mockResolvedValue(undefined);
  fakeContext.on.mockReset();
  fakeContext.close.mockResolvedValue(undefined);
  vi.mocked(chromium.launchPersistentContext).mockResolvedValue(
    fakeContext as unknown as Awaited<
      ReturnType<typeof chromium.launchPersistentContext>
    >,
  );
}

// ---------------------------------------------------------------------------
// sanitizeProfileDir — pure function, no module state involved
// ---------------------------------------------------------------------------

describe('sanitizeProfileDir', () => {
  it('returns default profile path when no input is given', () => {
    const result = sanitizeProfileDir();
    expect(result).toBe(path.join(base, 'profile'));
  });

  it('accepts a valid path within ~/.gemini-mcp/', () => {
    const validPath = path.join(base, 'custom');
    const result = sanitizeProfileDir(validPath);
    expect(result).toBe(validPath);
  });

  it('rejects a path outside ~/.gemini-mcp/ with an error', () => {
    expect(() => sanitizeProfileDir('/tmp/evil')).toThrow(
      /outside allowed base/,
    );
  });

  it('rejects a path traversal that escapes the base', () => {
    const traversal = path.join(base, '..', 'etc', 'passwd');
    expect(() => sanitizeProfileDir(traversal)).toThrow(/outside allowed base/);
  });

  it('accepts ~/.gemini-mcp itself as the base path', () => {
    const result = sanitizeProfileDir(base);
    expect(result).toBe(base);
  });

  it('rejects an absolute path that starts with base string but is not under base dir', () => {
    // e.g. ~/.gemini-mcp-evil should be rejected
    const sibling = path.join(os.homedir(), '.gemini-mcp-evil');
    expect(() => sanitizeProfileDir(sibling)).toThrow(/outside allowed base/);
  });
});

// ---------------------------------------------------------------------------
// isBrowserRunning — reflects module-level `context` variable
// ---------------------------------------------------------------------------

describe('isBrowserRunning', () => {
  it('returns false when no browser has been launched (module initial state)', () => {
    // The module is imported fresh: context is null at load time.
    expect(isBrowserRunning()).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// getProfileDir / setProfileDir
// ---------------------------------------------------------------------------

describe('getProfileDir / setProfileDir', () => {
  it('getProfileDir returns the default profile path initially', () => {
    // Default is set from env var GEMINI_MCP_PROFILE_DIR (not set in tests)
    // or falls back to ~/.gemini-mcp/profile
    const dir = getProfileDir();
    expect(dir).toBe(path.join(base, 'profile'));
  });

  it('setProfileDir changes the profile directory', () => {
    const newDir = path.join(base, 'test-profile');
    setProfileDir(newDir);
    expect(getProfileDir()).toBe(newDir);
  });

  it('setProfileDir accepts undefined (resets to default)', () => {
    // First set something non-default
    setProfileDir(path.join(base, 'test-profile'));
    // Now reset to default
    setProfileDir(undefined);
    expect(getProfileDir()).toBe(path.join(base, 'profile'));
  });

  it('setProfileDir rejects a path outside ~/.gemini-mcp/', () => {
    expect(() => setProfileDir('/tmp/evil')).toThrow(/outside allowed base/);
  });

  it('setProfileDir accepts ~/.gemini-mcp itself', () => {
    setProfileDir(base);
    expect(getProfileDir()).toBe(base);
    // Reset for other tests
    setProfileDir(undefined);
  });

  afterEach(() => {
    // Restore the default profile dir so other tests are not affected
    try {
      setProfileDir(undefined);
    } catch {
      // Ignore — if browser is running we can't reset, but no test leaves it running
    }
  });
});

// ---------------------------------------------------------------------------
// withPage — serialization gate + browser context initialization
// ---------------------------------------------------------------------------

describe('withPage serialization gate', () => {
  beforeEach(() => {
    resetFakeMocks();
  });

  it('executes a function with the page and returns its value', async () => {
    const result = await withPage(() => Promise.resolve(42));
    expect(result).toBe(42);
  });

  it('passes the Page object to the callback', async () => {
    let receivedPage: unknown;
    await withPage((page) => {
      receivedPage = page;
      return Promise.resolve(undefined);
    });
    // The page returned from getPage() is fakePage
    expect(receivedPage).toBe(fakePage);
  });

  it('rejects a concurrent withPage call while one is already in progress', async () => {
    let resolveFirst!: () => void;
    const firstDone = new Promise<void>((resolve) => {
      resolveFirst = resolve;
    });

    // Start the first call — it holds the lock until we call resolveFirst
    const firstCall = withPage(
      () =>
        new Promise<number>((resolve) => {
          // Signal that the first call has started but not yet released
          resolveFirst();
          // Yield to let the second call attempt to run
          setTimeout(() => resolve(1), 50);
        }),
    );

    // Wait until the first call has acquired the lock
    await firstDone;

    // The second call should be immediately rejected
    await expect(withPage(() => Promise.resolve(2))).rejects.toThrow(
      /already in progress/,
    );

    // Let the first call finish
    await expect(firstCall).resolves.toBe(1);
  });

  it('allows a subsequent withPage call after the previous one completes', async () => {
    // First call
    await withPage(() => Promise.resolve('first'));
    // Second call should succeed
    const second = await withPage(() => Promise.resolve('second'));
    expect(second).toBe('second');
  });

  it('releases the lock even when the callback throws', async () => {
    await expect(
      withPage(() => Promise.reject(new Error('callback error'))),
    ).rejects.toThrow('callback error');

    // The lock must be released — this should not throw "already in progress"
    const result = await withPage(() => Promise.resolve('after error'));
    expect(result).toBe('after error');
  });

  it('calls launchPersistentContext to initialize the browser on first use', async () => {
    // This test must run in isolation with a fresh module so that context is null
    // and launchPersistentContext will actually be invoked.
    const freshBrowser = await importFreshBrowser();

    // Provide a mock for the freshly-imported playwright chromium
    const freshPlaywright = await import('playwright');
    vi.mocked(freshPlaywright.chromium.launchPersistentContext).mockResolvedValue(
      fakeContext as unknown as Awaited<
        ReturnType<typeof chromium.launchPersistentContext>
      >,
    );

    await freshBrowser.withPage(() => Promise.resolve(undefined));
    expect(freshPlaywright.chromium.launchPersistentContext).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// checkLoginStatus
// ---------------------------------------------------------------------------

describe('checkLoginStatus', () => {
  beforeEach(() => {
    resetFakeMocks();
  });

  it('returns false when no browser context is running (context is null)', async () => {
    // At module load context is null; we have NOT yet called withPage to warm it up
    // We need a fresh module state — use vi.resetModules + dynamic import for this test.
    const { checkLoginStatus: freshCheck } = await importFreshBrowser();
    const result = await freshCheck();
    expect(result).toBe(false);
  });

  it('returns true when the page URL contains /app', async () => {
    fakePage.url.mockReturnValue('https://gemini.google.com/app/test123');
    // Warm up the context so it is non-null
    await withPage(() => Promise.resolve(undefined));
    const result = await checkLoginStatus();
    expect(result).toBe(true);
  });

  it('returns false when the page URL does not contain /app', async () => {
    fakePage.url.mockReturnValue('https://accounts.google.com/signin/v2');
    await withPage(() => Promise.resolve(undefined));
    const result = await checkLoginStatus();
    expect(result).toBe(false);
  });

  it('returns false when the page URL is gemini.google.com without /app', async () => {
    fakePage.url.mockReturnValue('https://gemini.google.com/');
    await withPage(() => Promise.resolve(undefined));
    const result = await checkLoginStatus();
    expect(result).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Helpers for module isolation
// ---------------------------------------------------------------------------

/**
 * Import a completely fresh copy of browser.ts by resetting the module registry.
 * The playwright and fs/promises mocks are still in effect because vi.mock() is
 * registered globally, but the module-level state (context, initPromise,
 * _profileDir, _pageBusy) is freshly initialised.
 */
async function importFreshBrowser(): Promise<
  typeof import('../src/browser.js')
> {
  vi.resetModules();

  // Re-apply mocks in the fresh module graph
  vi.mock('playwright', () => ({
    chromium: { launchPersistentContext: vi.fn() },
  }));
  vi.mock('node:fs/promises', () => ({
    mkdir: vi.fn().mockResolvedValue(undefined),
  }));
  vi.mock('../src/logger.js', () => ({
    logger: {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    },
  }));

  return import('../src/browser.js');
}
