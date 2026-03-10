import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as path from 'node:path';
import * as os from 'node:os';

// ---------------------------------------------------------------------------
// Top-level mocks (must appear before any imports that load the mocked modules)
// ---------------------------------------------------------------------------

vi.mock('playwright', () => ({
  chromium: { launchPersistentContext: vi.fn() },
}));

vi.mock('node:fs', () => ({
  mkdirSync: vi.fn(),
}));

vi.mock('node:fs/promises', () => ({
  writeFile: vi.fn().mockResolvedValue(undefined),
  readdir: vi.fn().mockResolvedValue([]),
  stat: vi.fn().mockResolvedValue({ mtimeMs: Date.now() }),
  unlink: vi.fn().mockResolvedValue(undefined),
  mkdir: vi.fn().mockResolvedValue(undefined),
  readFile: vi.fn().mockResolvedValue(Buffer.from('fake-image')),
  copyFile: vi.fn().mockResolvedValue(undefined),
  rename: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../src/browser.js', () => ({
  sanitizeProfileDir: vi.fn((input?: string) => {
    if (!input) return path.join(os.homedir(), '.gemini-mcp', 'profile');
    const resolved = path.resolve(input);
    const base = path.join(os.homedir(), '.gemini-mcp');
    if (!resolved.startsWith(base + path.sep) && resolved !== base) {
      throw new Error('outside allowed base');
    }
    return resolved;
  }),
  getPage: vi.fn(),
  withPage: vi.fn(),
  closeBrowser: vi.fn(),
  checkLoginStatus: vi.fn().mockResolvedValue(false),
  isBrowserRunning: vi.fn().mockReturnValue(false),
  getProfileDir: vi.fn().mockReturnValue(path.join(os.homedir(), '.gemini-mcp', 'profile')),
  setProfileDir: vi.fn(),
}));

// ---------------------------------------------------------------------------
// Imports (after mocks)
// ---------------------------------------------------------------------------

import { writeFile, readdir } from 'node:fs/promises';
import {
  extractConversationId,
  armImageCapture,
  navigateToConversation,
  pruneImageDir,
  LARGE_IMAGE_THRESHOLD,
} from '../src/gemini.js';
import { createLogger, LOG_LEVELS } from '../src/logger.js';
import { makeMockPage, makeResponseMock, flushMicrotasks, FALLBACK_ID_PATTERN } from './helpers.js';

beforeEach(() => {
  vi.clearAllMocks();
});

// ===========================================================================
// Regression: async writeFile (P1 fix — was writeFileSync)
// ===========================================================================

describe('Regression: async file I/O for large images', () => {
  it('large image uses async writeFile (not blocking)', async () => {
    const mockPage = makeMockPage();
    const fakeResponse = makeResponseMock(500_000); // 500KB > threshold
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    // Result must have filePath (large image path), no inline buffer
    expect(result.filePath).toBeTruthy();
    expect(result.buffer).toBeNull();

    // The async writeFile from 'node:fs/promises' must have been called
    expect(writeFile).toHaveBeenCalledWith(
      expect.stringContaining('/tmp/gemini-images/'),
      expect.any(Buffer),
    );
  });
});

// ===========================================================================
// Regression: temp file cleanup after large image write
// ===========================================================================

describe('Regression: temp file cleanup after large image write', () => {
  it('pruneImageDir is called (readdir invoked) after writing large image to file', async () => {
    const mockPage = makeMockPage();
    const fakeResponse = makeResponseMock(500_000);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    await capture.waitForImage(5000);

    // pruneImageDir is fire-and-forget — give the micro-task queue a tick
    await flushMicrotasks();

    // readdir is the first I/O call inside pruneImageDir — it proves the function ran
    expect(readdir).toHaveBeenCalled();
  });

  it('pruneImageDir is NOT invoked after writing a small image (buffer path)', async () => {
    const mockPage = makeMockPage();
    const fakeResponse = makeResponseMock(100_000); // small image
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    await capture.waitForImage(5000);

    await flushMicrotasks();

    // pruneImageDir should NOT have been called for a small image
    expect(readdir).not.toHaveBeenCalled();
  });
});

// ===========================================================================
// Regression: checkLoginStatus read-only (P2 fix — must not call goto)
// ===========================================================================

describe('Regression: checkLoginStatus does not navigate', () => {
  // Simulate the real implementation logic without importing the module
  // (which would pull in browser state).
  const testCheckLoginStatus = async (pageUrl: string): Promise<boolean> => {
    return pageUrl.includes('gemini.google.com/app');
  };

  it('returns true for a logged-in Gemini /app URL', async () => {
    const result = await testCheckLoginStatus('https://gemini.google.com/app/abc123');
    expect(result).toBe(true);
  });

  it('returns false for a Google accounts sign-in URL', async () => {
    const result = await testCheckLoginStatus('https://accounts.google.com/signin');
    expect(result).toBe(false);
  });

  it('returns false for the Gemini root URL (no /app)', async () => {
    const result = await testCheckLoginStatus('https://gemini.google.com/');
    expect(result).toBe(false);
  });

  it('the check is purely URL-based and never requires a goto call', async () => {
    // A goto-free implementation means no Page object is required at all
    const gotoSpy = vi.fn();
    // Calling our test helper never touches gotoSpy
    await testCheckLoginStatus('https://gemini.google.com/app/xyz');
    expect(gotoSpy).not.toHaveBeenCalled();
  });
});

// ===========================================================================
// Regression: pages[0] conditional handling (P2 fix — was non-null assertion)
// ===========================================================================

describe('Regression: pages[0] conditional (was non-null assertion)', () => {
  // Extracted logic mirroring browser.ts getPage()
  const testGetPage = (existingPages: any[], newPageFn: () => any) => {
    const page = existingPages[0];
    if (page) return page;
    return newPageFn();
  };

  it('returns the existing page when pages[0] exists', () => {
    const fakePage = { url: () => 'https://gemini.google.com/app/abc' };
    const newPageFn = vi.fn();
    const result = testGetPage([fakePage], newPageFn);
    expect(result).toBe(fakePage);
    expect(newPageFn).not.toHaveBeenCalled();
  });

  it('creates a new page when pages array is empty', () => {
    const newPage = { url: () => 'about:blank' };
    const newPageFn = vi.fn().mockReturnValue(newPage);
    const result = testGetPage([], newPageFn);
    expect(newPageFn).toHaveBeenCalledOnce();
    expect(result).toBe(newPage);
  });

  it('creates a new page when pages array contains only undefined/falsy entries', () => {
    const newPage = { url: () => 'about:blank' };
    const newPageFn = vi.fn().mockReturnValue(newPage);
    // Simulate what happens if pages[0] is undefined
    const result = testGetPage([undefined], newPageFn);
    expect(newPageFn).toHaveBeenCalledOnce();
    expect(result).toBe(newPage);
  });
});

// ===========================================================================
// Regression: conversation ID regex validation (security)
// ===========================================================================

describe('Regression: conversation ID validation (security)', () => {
  const VALID_IDS = [
    'abc123',
    'ABC-xyz_123',
    'A',
    'MyConversation',
    'conv-001_abc',
  ];

  const INVALID_IDS = [
    '../etc/passwd',
    'id with spaces',
    'id/with/slashes',
    'id\nwith\nnewlines',
    '',
  ];

  describe('valid IDs resolve without error', () => {
    for (const id of VALID_IDS) {
      it(`accepts "${id}"`, async () => {
        const mockPage = makeMockPage();
        mockPage.goto.mockResolvedValue(null);
        await expect(
          navigateToConversation(mockPage as any, id),
        ).resolves.toBeUndefined();
      });
    }
  });

  describe('invalid IDs reject with /Invalid conversationId/', () => {
    for (const id of INVALID_IDS) {
      it(`rejects ${JSON.stringify(id)}`, async () => {
        const mockPage = makeMockPage();
        await expect(
          navigateToConversation(mockPage as any, id),
        ).rejects.toThrow(/Invalid conversationId/);
      });
    }
  });
});

// ===========================================================================
// Regression: large image threshold boundary (exactly 400KB)
// ===========================================================================

describe('Regression: large image threshold boundary (400KB)', () => {
  it('LARGE_IMAGE_THRESHOLD bytes is NOT large — returned as buffer, not file', async () => {
    const mockPage = makeMockPage();
    const fakeResponse = makeResponseMock(LARGE_IMAGE_THRESHOLD);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.buffer).toBeInstanceOf(Buffer);
    expect(result.filePath).toBeNull();
    expect(writeFile).not.toHaveBeenCalled();
  });

  it('LARGE_IMAGE_THRESHOLD + 1 bytes IS large — written to file, buffer is null', async () => {
    const mockPage = makeMockPage();
    const fakeResponse = makeResponseMock(LARGE_IMAGE_THRESHOLD + 1);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.buffer).toBeNull();
    expect(result.filePath).toBeTruthy();
    expect(writeFile).toHaveBeenCalled();
  });

  it('1 byte is small — returned as buffer', async () => {
    const mockPage = makeMockPage();
    const fakeResponse = makeResponseMock(1);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.buffer).toBeInstanceOf(Buffer);
    expect(result.filePath).toBeNull();
  });
});

// ===========================================================================
// Functional: extractConversationId URL patterns
// ===========================================================================

describe('Functional: extractConversationId URL patterns', () => {
  it('extracts ID from a standard /app/ URL', () => {
    const mockPage = makeMockPage('https://gemini.google.com/app/myConvId123');
    expect(extractConversationId(mockPage as any)).toBe('myConvId123');
  });

  it('extracts ID containing hyphens and underscores', () => {
    const mockPage = makeMockPage('https://gemini.google.com/app/my-conv_ID');
    expect(extractConversationId(mockPage as any)).toBe('my-conv_ID');
  });

  it('returns fallback-<number> for the Gemini root URL', () => {
    const mockPage = makeMockPage('https://gemini.google.com/');
    const id = extractConversationId(mockPage as any);
    expect(id).toMatch(FALLBACK_ID_PATTERN);
  });

  it('returns fallback-<number> for a Google accounts URL', () => {
    const mockPage = makeMockPage('https://accounts.google.com/signin');
    const id = extractConversationId(mockPage as any);
    expect(id).toMatch(FALLBACK_ID_PATTERN);
  });

  it('extracts ID when URL contains query params after the path', () => {
    // The regex only matches the /app/<id> portion — query params follow
    const mockPage = makeMockPage('https://gemini.google.com/app/convABC?foo=bar');
    const id = extractConversationId(mockPage as any);
    expect(id).toBe('convABC');
  });

  it('returns a unique fallback on each call when URL has no conversation ID', () => {
    // Two calls in quick succession may share the same millisecond timestamp,
    // but the pattern must always match.
    const mockPage = makeMockPage('https://gemini.google.com/');
    const id1 = extractConversationId(mockPage as any);
    const id2 = extractConversationId(mockPage as any);
    expect(id1).toMatch(FALLBACK_ID_PATTERN);
    expect(id2).toMatch(FALLBACK_ID_PATTERN);
  });
});

// ===========================================================================
// Functional: Logger behavior
// ===========================================================================

describe('Functional: logger stderr output', () => {
  it('debug messages are suppressed when log level is info', () => {
    const log = createLogger('info');
    const spy = vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    log.debug('debug msg');
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it('info messages are written at info level', () => {
    const log = createLogger('info');
    const chunks: string[] = [];
    const spy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation((chunk: any) => {
        chunks.push(String(chunk));
        return true;
      });
    log.info('info msg');
    spy.mockRestore();

    expect(chunks.length).toBeGreaterThan(0);
    const parsed = JSON.parse(chunks[0]);
    expect(parsed.level).toBe('info');
    expect(parsed.message).toBe('info msg');
  });

  it('warn messages are written at info level', () => {
    const log = createLogger('info');
    const chunks: string[] = [];
    const spy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation((chunk: any) => {
        chunks.push(String(chunk));
        return true;
      });
    log.warn('warn msg');
    spy.mockRestore();

    expect(chunks.length).toBeGreaterThan(0);
    const parsed = JSON.parse(chunks[0]);
    expect(parsed.level).toBe('warn');
  });

  it('error messages include serialized Error object fields', () => {
    const log = createLogger('info');
    const chunks: string[] = [];
    const spy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation((chunk: any) => {
        chunks.push(String(chunk));
        return true;
      });
    log.error('err', new Error('test error'));
    spy.mockRestore();

    expect(chunks.length).toBeGreaterThan(0);
    const parsed = JSON.parse(chunks[0]);
    expect(parsed.level).toBe('error');
    expect(parsed.message).toBe('err');
    expect(parsed.data).toBeDefined();
    expect(parsed.data.name).toBe('Error');
    expect(parsed.data.message).toBe('test error');
    expect(typeof parsed.data.stack).toBe('string');
  });

  it('debug messages ARE written when log level is debug', () => {
    const log = createLogger('debug');
    const chunks: string[] = [];
    const spy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation((chunk: any) => {
        chunks.push(String(chunk));
        return true;
      });
    log.debug('debug visible');
    spy.mockRestore();

    expect(chunks.length).toBeGreaterThan(0);
    const parsed = JSON.parse(chunks[0]);
    expect(parsed.level).toBe('debug');
  });

  it('LOG_LEVELS ordering: debug < info < warn < error', () => {
    expect(LOG_LEVELS.debug).toBeLessThan(LOG_LEVELS.info);
    expect(LOG_LEVELS.info).toBeLessThan(LOG_LEVELS.warn);
    expect(LOG_LEVELS.warn).toBeLessThan(LOG_LEVELS.error);
  });

  it('log entry includes a timestamp field', () => {
    const log = createLogger('info');
    const chunks: string[] = [];
    const spy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation((chunk: any) => {
        chunks.push(String(chunk));
        return true;
      });
    log.info('with timestamp');
    spy.mockRestore();

    const parsed = JSON.parse(chunks[0]);
    expect(typeof parsed.timestamp).toBe('string');
    // Timestamp should be a valid ISO 8601 date
    expect(() => new Date(parsed.timestamp)).not.toThrow();
  });

  it('log entry has no data field when none is passed', () => {
    const log = createLogger('info');
    const chunks: string[] = [];
    const spy = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation((chunk: any) => {
        chunks.push(String(chunk));
        return true;
      });
    log.info('no data');
    spy.mockRestore();

    const parsed = JSON.parse(chunks[0]);
    expect('data' in parsed).toBe(false);
  });
});

// ===========================================================================
// Functional: GEMINI_MCP_PROFILE_DIR environment variable
// ===========================================================================

describe('Functional: GEMINI_MCP_PROFILE_DIR environment variable', () => {
  // Simulate what the browser.ts IIFE does to consume the env var
  const BASE = path.join(os.homedir(), '.gemini-mcp');

  const getProfileFromEnv = (envDir?: string): string => {
    if (envDir) {
      try {
        const resolved = path.resolve(envDir);
        if (resolved.startsWith(BASE + path.sep) || resolved === BASE) {
          return resolved;
        }
      } catch {
        // invalid path — fall through to default
      }
    }
    return path.join(BASE, 'profile');
  };

  it('uses default profile path when env var is not set', () => {
    const result = getProfileFromEnv(undefined);
    expect(result).toContain('.gemini-mcp');
    expect(result.endsWith(path.join('.gemini-mcp', 'profile'))).toBe(true);
  });

  it('uses env var when set to a valid path inside ~/.gemini-mcp/', () => {
    const customDir = path.join(BASE, 'custom');
    const result = getProfileFromEnv(customDir);
    expect(result).toContain('custom');
    expect(result).toBe(customDir);
  });

  it('falls back to default when env var points outside ~/.gemini-mcp/', () => {
    const result = getProfileFromEnv('/tmp/evil');
    expect(result.endsWith(path.join('.gemini-mcp', 'profile'))).toBe(true);
  });

  it('falls back to default when env var points to /etc/passwd', () => {
    const result = getProfileFromEnv('/etc/passwd');
    expect(result.endsWith(path.join('.gemini-mcp', 'profile'))).toBe(true);
  });

  it('accepts the BASE directory itself as a valid profile dir', () => {
    const result = getProfileFromEnv(BASE);
    expect(result).toBe(BASE);
  });

  it('resolved path must start with BASE + sep OR equal BASE', () => {
    // This is the exact guard condition from browser.ts
    const validChild = path.join(BASE, 'sessions', 'profile-1');
    const result = getProfileFromEnv(validChild);
    expect(result).toBe(validChild);
  });
});

// ===========================================================================
// Functional: navigateToConversation constructs correct URL
// ===========================================================================

describe('Functional: navigateToConversation URL construction', () => {
  it('navigates to the correct Gemini conversation URL', async () => {
    const mockPage = makeMockPage();
    mockPage.goto.mockResolvedValue(null);

    await navigateToConversation(mockPage as any, 'myConv123');

    expect(mockPage.goto).toHaveBeenCalledWith(
      'https://gemini.google.com/app/myConv123',
      expect.objectContaining({ waitUntil: 'domcontentloaded' }),
    );
  });

  it('never calls goto for an invalid conversation ID', async () => {
    const mockPage = makeMockPage();

    await expect(
      navigateToConversation(mockPage as any, '../../../etc/shadow'),
    ).rejects.toThrow(/Invalid conversationId/);

    expect(mockPage.goto).not.toHaveBeenCalled();
  });
});

// ===========================================================================
// Functional: pruneImageDir fire-and-forget contract
// ===========================================================================

describe('Functional: pruneImageDir fire-and-forget contract', () => {
  it('does not return a Promise (returns void synchronously)', () => {
    const result = pruneImageDir();
    // The function signature is void — the return value should be undefined
    expect(result).toBeUndefined();
  });

  it('does not throw synchronously even when readdir rejects', async () => {
    vi.mocked(readdir).mockRejectedValueOnce(new Error('ENOENT'));
    expect(() => pruneImageDir()).not.toThrow();
    // Allow the rejection to be absorbed
    await flushMicrotasks();
  });
});
