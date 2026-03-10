/**
 * Security and regression tests for gemini-browser-mcp
 *
 * Covers:
 *  1. sanitizeOutputPath – path traversal / boundary enforcement
 *  2. navigateToConversation – conversation ID input validation
 *  3. Navigation allowlist logic – hostname-level gating
 *  4. withPage concurrent call rejection – serialisation lock
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// ---------------------------------------------------------------------------
// Top-level mocks (must come before any src imports)
// ---------------------------------------------------------------------------

// Prevent MCP server startup side-effects
vi.mock('@modelcontextprotocol/sdk/server/mcp.js', () => ({
  McpServer: vi.fn().mockImplementation(() => ({
    tool: vi.fn(),
    connect: vi.fn().mockResolvedValue(undefined),
  })),
}));
vi.mock('@modelcontextprotocol/sdk/server/stdio.js', () => ({
  StdioServerTransport: vi.fn(),
}));

// Stub browser module (prevents Playwright/Chrome launch),
// but keep isAllowedNavigation as the real implementation so it can be tested directly.
vi.mock('../src/browser.js', async (importOriginal) => {
  const real = await importOriginal() as typeof import('../src/browser.js');
  return {
    ...real,
    getPage: vi.fn(),
    withPage: vi.fn(),
    closeBrowser: vi.fn(),
    checkLoginStatus: vi.fn(),
    isBrowserRunning: vi.fn().mockReturnValue(false),
    getProfileDir: vi.fn().mockReturnValue('/home/user/.gemini-mcp/profile'),
    setProfileDir: vi.fn(),
    sanitizeProfileDir: vi.fn(),
  };
});

// Partially mock gemini module: keep navigateToConversation real, stub DOM-heavy functions
vi.mock('../src/gemini.js', async (importOriginal) => {
  const actual = await importOriginal() as typeof import('../src/gemini.js');
  return {
    ...actual,
    queryImage: vi.fn(),
    iterateImage: vi.fn(),
    pruneImageDir: vi.fn(),
  };
});

// Stub zod so that src/index.ts tool registrations don't fail during import
vi.mock('zod', () => {
  const schema: any = {
    min: () => schema,
    max: () => schema,
    int: () => schema,
    optional: () => schema,
    default: () => schema,
    regex: () => schema,
    enum: () => schema,
    parse: vi.fn(),
  };
  return {
    z: {
      string: () => schema,
      number: () => schema,
      boolean: () => schema,
      object: () => schema,
      enum: () => schema,
    },
  };
});

// Playwright mock (needed for gemini.ts module-level side-effects when imported
// directly in the navigateToConversation tests below)
vi.mock('playwright', () => ({ chromium: { launchPersistentContext: vi.fn() } }));
vi.mock('node:fs', () => ({ mkdirSync: vi.fn() }));
vi.mock('node:fs/promises', () => ({
  writeFile: vi.fn().mockResolvedValue(undefined),
  readdir: vi.fn().mockResolvedValue([]),
  stat: vi.fn(),
  unlink: vi.fn().mockResolvedValue(undefined),
  mkdir: vi.fn().mockResolvedValue(undefined),
}));

// ---------------------------------------------------------------------------
// Imports (after mocks are registered)
// ---------------------------------------------------------------------------
import { sanitizeOutputPath } from '../src/index.js';
import { navigateToConversation } from '../src/gemini.js';
import { isAllowedNavigation } from '../src/browser.js';
import { makeMockPage, FALLBACK_ID_PATTERN } from './helpers.js';

// ---------------------------------------------------------------------------
// 1. sanitizeOutputPath
// ---------------------------------------------------------------------------
describe('sanitizeOutputPath', () => {
  it('accepts path inside /tmp/gemini-images/', () => {
    const result = sanitizeOutputPath('/tmp/gemini-images/test.webp');
    expect(result).toBe('/tmp/gemini-images/test.webp');
  });

  it('rejects path outside /tmp/gemini-images/', () => {
    expect(() => sanitizeOutputPath('/etc/passwd')).toThrow(/outside allowed directory/);
  });

  it('rejects path traversal: /tmp/gemini-images/../etc/passwd', () => {
    expect(() => sanitizeOutputPath('/tmp/gemini-images/../etc/passwd')).toThrow(
      /outside allowed directory/,
    );
  });

  it('accepts /tmp/gemini-images itself', () => {
    // The base directory itself is allowed (resolved === IMAGE_OUTPUT_BASE)
    expect(() => sanitizeOutputPath('/tmp/gemini-images')).not.toThrow();
    const result = sanitizeOutputPath('/tmp/gemini-images');
    expect(result).toBe('/tmp/gemini-images');
  });

  it('rejects /tmp/gemini-images-evil/x.webp (startsWith check requires trailing sep)', () => {
    expect(() => sanitizeOutputPath('/tmp/gemini-images-evil/x.webp')).toThrow(
      /outside allowed directory/,
    );
  });

  it('accepts deeply nested path inside /tmp/gemini-images/', () => {
    expect(() =>
      sanitizeOutputPath('/tmp/gemini-images/2024/jan/img.webp'),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// 2. navigateToConversation – input validation
// ---------------------------------------------------------------------------
describe('navigateToConversation - input validation', () => {
  let mockPage: ReturnType<typeof makeMockPage>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockPage = makeMockPage();
  });

  it('rejects conversationId with path traversal: ../etc/passwd', async () => {
    await expect(
      navigateToConversation(mockPage as any, '../etc/passwd'),
    ).rejects.toThrow(/Invalid conversationId/);
  });

  it('rejects conversationId with semicolon', async () => {
    await expect(
      navigateToConversation(mockPage as any, 'abc;rm -rf /'),
    ).rejects.toThrow(/Invalid conversationId/);
  });

  it('rejects conversationId with spaces', async () => {
    await expect(
      navigateToConversation(mockPage as any, 'abc 123'),
    ).rejects.toThrow(/Invalid conversationId/);
  });

  it('rejects empty conversationId', async () => {
    await expect(
      navigateToConversation(mockPage as any, ''),
    ).rejects.toThrow(/Invalid conversationId/);
  });

  it('accepts valid alphanumeric with hyphens/underscores and calls goto with correct URL', async () => {
    await expect(
      navigateToConversation(mockPage as any, 'abc-123_XYZ'),
    ).resolves.toBeUndefined();
    expect(mockPage.goto).toHaveBeenCalledWith(
      'https://gemini.google.com/app/abc-123_XYZ',
      expect.anything(),
    );
  });
});

// ---------------------------------------------------------------------------
// 3. Navigation allowlist logic
//    Uses the exported isAllowedNavigation from browser.ts directly.
// ---------------------------------------------------------------------------

describe('navigation allowlist logic', () => {
  it('allows gemini.google.com', () => {
    expect(isAllowedNavigation('https://gemini.google.com/app/abc123')).toBe(true);
  });

  it('allows accounts.google.com', () => {
    expect(isAllowedNavigation('https://accounts.google.com/signin')).toBe(true);
  });

  it('blocks evil.com', () => {
    expect(isAllowedNavigation('https://evil.com/steal')).toBe(false);
  });

  it('blocks subdomain.gemini.google.com (subdomain not in allowlist)', () => {
    expect(isAllowedNavigation('https://subdomain.gemini.google.com/')).toBe(false);
  });

  it('allows chrome:// URLs (special handling)', () => {
    expect(isAllowedNavigation('chrome://newtab')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 4. Concurrent withPage lock – inline testable reimplementation
// ---------------------------------------------------------------------------
function makeTestableWithPage() {
  let _pageBusy = false;
  function testableWithPage<T>(fn: () => Promise<T>): Promise<T> {
    if (_pageBusy) {
      return Promise.reject(new Error('A Gemini operation is already in progress.'));
    }
    _pageBusy = true;
    return fn().finally(() => {
      _pageBusy = false;
    });
  }
  return testableWithPage;
}

describe('withPage concurrent call rejection', () => {
  it('allows sequential withPage calls', async () => {
    const withPage = makeTestableWithPage();
    await expect(withPage(() => Promise.resolve('first'))).resolves.toBe('first');
    await expect(withPage(() => Promise.resolve('second'))).resolves.toBe('second');
  });

  it('rejects concurrent withPage calls with "already in progress"', async () => {
    const withPage = makeTestableWithPage();

    // First call: takes 100 ms to complete
    const first = withPage(
      () =>
        new Promise<string>((resolve) => setTimeout(() => resolve('done'), 100)),
    );

    // Second call fires immediately while the first is still running
    await expect(withPage(() => Promise.resolve('concurrent'))).rejects.toThrow(
      /already in progress/,
    );

    // First call should still resolve normally
    await expect(first).resolves.toBe('done');
  });

  it('releases lock after function throws', async () => {
    const withPage = makeTestableWithPage();

    // First call rejects
    await expect(
      withPage(() => Promise.reject(new Error('test error'))),
    ).rejects.toThrow('test error');

    // Lock must have been released — second call should succeed
    await expect(withPage(() => Promise.resolve('after error'))).resolves.toBe(
      'after error',
    );
  });
});
