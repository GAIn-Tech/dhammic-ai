import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as os from 'node:os';
import * as path from 'node:path';

// Mock browser.ts before importing gemini.ts
vi.mock('../src/browser.js', () => ({
  sanitizeProfileDir: vi.fn((input?: string) => {
    const base = path.join(os.homedir(), '.gemini-mcp');
    if (!input) return path.join(base, 'profile');
    const resolved = path.resolve(input);
    if (!resolved.startsWith(base + path.sep) && resolved !== base) {
      throw new Error(`Profile dir "${resolved}" is outside allowed base "${base}"`);
    }
    return resolved;
  }),
  getPage: vi.fn(),
  closeBrowser: vi.fn(),
  checkLoginStatus: vi.fn(),
  getProfileDir: vi.fn(),
  setProfileDir: vi.fn(),
}));

// Mock playwright
vi.mock('playwright', () => ({
  chromium: {
    launchPersistentContext: vi.fn(),
  },
}));

// Mock 'node:fs' (sync — used only for mkdirSync at module load)
vi.mock('node:fs', () => ({
  mkdirSync: vi.fn(),
}));

// Mock 'node:fs/promises' (async — used for writeFile in waitForImage)
vi.mock('node:fs/promises', () => ({
  writeFile: vi.fn().mockResolvedValue(undefined),
  readdir: vi.fn().mockResolvedValue([]),
  stat: vi.fn(),
  unlink: vi.fn().mockResolvedValue(undefined),
}));

import {
  sanitizeProfileDir,
} from '../src/browser.js';

import {
  extractConversationId,
  armImageCapture,
  navigateToConversation,
  fillPrompt,
  submitPrompt,
  ensureOnGemini,
  queryImage,
  iterateImage,
  pruneImageDir,
  LARGE_IMAGE_THRESHOLD,
} from '../src/gemini.js';

import { writeFile, readdir, stat, unlink } from 'node:fs/promises';
import { makeMockPage, makeResponseMock, flushMicrotasks, FALLBACK_ID_PATTERN } from './helpers.js';

// ---------------------------------------------------------------------------
// Shared minimal mock Page
// ---------------------------------------------------------------------------
let mockPage: ReturnType<typeof makeMockPage>;

// ---------------------------------------------------------------------------
// Reset mocks between tests
// ---------------------------------------------------------------------------
beforeEach(() => {
  vi.clearAllMocks();
  mockPage = makeMockPage();
  // Default: waitForResponse times out (Playwright rejects on timeout)
  mockPage.waitForResponse.mockRejectedValue(new Error('Timeout exceeded.'));
});

// ---------------------------------------------------------------------------
// sanitizeProfileDir
// ---------------------------------------------------------------------------
describe('sanitizeProfileDir', () => {
  const base = path.join(os.homedir(), '.gemini-mcp');

  it('rejects paths outside ~/.gemini-mcp/ with an error', () => {
    expect(() => sanitizeProfileDir('/tmp/evil-profile')).toThrow(
      /outside allowed base/,
    );
  });

  it('accepts a valid path within ~/.gemini-mcp/', () => {
    const validPath = path.join(base, 'my-profile');
    expect(() => sanitizeProfileDir(validPath)).not.toThrow();
    const result = sanitizeProfileDir(validPath);
    expect(result).toBe(validPath);
  });
});

// ---------------------------------------------------------------------------
// extractConversationId
// ---------------------------------------------------------------------------
describe('extractConversationId', () => {
  it('returns correct ID from a Gemini conversation URL', () => {
    mockPage.url.mockReturnValue('https://gemini.google.com/app/abc123XYZ');
    const id = extractConversationId(mockPage as any);
    expect(id).toBe('abc123XYZ');
  });

  it('returns fallback-* for a non-matching URL', () => {
    mockPage.url.mockReturnValue('https://gemini.google.com/');
    const id = extractConversationId(mockPage as any);
    expect(id).toMatch(FALLBACK_ID_PATTERN);
  });
});

// ---------------------------------------------------------------------------
// armImageCapture
// ---------------------------------------------------------------------------
describe('armImageCapture', () => {
  it('returns an object with a waitForImage method', () => {
    mockPage.waitForResponse.mockReturnValue(new Promise(() => {}));
    const capture = armImageCapture(mockPage as any);
    expect(capture).toBeDefined();
    expect(typeof capture.waitForImage).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Image size routing
// ---------------------------------------------------------------------------
describe('image size routing via armImageCapture.waitForImage', () => {
  it('writes image > 400KB to a file and returns filePath (not buffer)', async () => {
    const LARGE = LARGE_IMAGE_THRESHOLD + 1;
    const fakeResponse = makeResponseMock(LARGE);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.filePath).toBeTruthy();
    expect(result.buffer).toBeNull();
    expect(writeFile).toHaveBeenCalledWith(
      expect.stringContaining('/tmp/gemini-images/'),
      expect.any(Buffer),
    );
  });

  it('returns image <= 400KB as Buffer (not filePath)', async () => {
    const SMALL = 100_000;
    const fakeResponse = makeResponseMock(SMALL);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.buffer).toBeInstanceOf(Buffer);
    expect(result.filePath).toBeNull();
    expect(writeFile).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// navigateToConversation
// ---------------------------------------------------------------------------
describe('navigateToConversation', () => {
  it('rejects an invalid conversationId containing ..', async () => {
    await expect(
      navigateToConversation(mockPage as any, '../etc/passwd'),
    ).rejects.toThrow(/Invalid conversationId/);
  });

  it('navigates to valid conversationId without throwing', async () => {
    await expect(
      navigateToConversation(mockPage as any, 'abc123XYZ'),
    ).resolves.toBeUndefined();
    expect(mockPage.goto).toHaveBeenCalledWith(
      'https://gemini.google.com/app/abc123XYZ',
      expect.anything(),
    );
  });
});

// ---------------------------------------------------------------------------
// fillPrompt behavior
// ---------------------------------------------------------------------------
describe('fillPrompt', () => {
  it('uses first selector when locator returns count > 0', async () => {
    const fillMock = vi.fn().mockResolvedValue(undefined);
    const countMock = vi.fn().mockResolvedValue(1);
    const firstReturnValue = {
      count: countMock,
      fill: fillMock,
      click: vi.fn().mockResolvedValue(undefined),
      waitFor: vi.fn().mockResolvedValue(undefined),
      getAttribute: vi.fn().mockResolvedValue(null),
    };
    mockPage.locator.mockReturnValue({
      first: vi.fn().mockReturnValue(firstReturnValue),
    });

    await fillPrompt(mockPage as any, 'hello world');

    expect(fillMock).toHaveBeenCalledWith('hello world');
    expect(mockPage.keyboard.type).not.toHaveBeenCalled();
  });

  it('falls back to keyboard.type when all locators return count 0', async () => {
    // All locators return count 0 (already set up in beforeEach)
    await fillPrompt(mockPage as any, 'fallback text');

    expect(mockPage.keyboard.type).toHaveBeenCalledWith('fallback text');
  });

  it('calls keyboard.type when locator throws', async () => {
    mockPage.locator.mockImplementation(() => {
      throw new Error('locator error');
    });

    await fillPrompt(mockPage as any, 'typed text');

    expect(mockPage.keyboard.type).toHaveBeenCalledWith('typed text');
  });
});

// ---------------------------------------------------------------------------
// submitPrompt behavior
// ---------------------------------------------------------------------------
describe('submitPrompt', () => {
  it('clicks submit button when count > 0', async () => {
    const clickMock = vi.fn().mockResolvedValue(undefined);
    const countMock = vi.fn().mockResolvedValue(1);
    const firstReturnValue = {
      count: countMock,
      click: clickMock,
      fill: vi.fn().mockResolvedValue(undefined),
      waitFor: vi.fn().mockResolvedValue(undefined),
      getAttribute: vi.fn().mockResolvedValue(null),
    };
    mockPage.locator.mockReturnValue({
      first: vi.fn().mockReturnValue(firstReturnValue),
    });

    await submitPrompt(mockPage as any);

    expect(clickMock).toHaveBeenCalled();
    expect(mockPage.keyboard.press).not.toHaveBeenCalled();
  });

  it('falls back to Enter key when no buttons found', async () => {
    // All locators return count 0 (already set up in beforeEach)
    await submitPrompt(mockPage as any);

    expect(mockPage.keyboard.press).toHaveBeenCalledWith('Enter');
  });
});

// ---------------------------------------------------------------------------
// ensureOnGemini behavior
// ---------------------------------------------------------------------------
describe('ensureOnGemini', () => {
  it('does not navigate when already on gemini.google.com', async () => {
    mockPage.url.mockReturnValue('https://gemini.google.com/app/abc');

    await ensureOnGemini(mockPage as any);

    expect(mockPage.goto).not.toHaveBeenCalled();
  });

  it('navigates to gemini when on different domain', async () => {
    mockPage.url.mockReturnValue('https://google.com');

    await ensureOnGemini(mockPage as any);

    expect(mockPage.goto).toHaveBeenCalledWith(
      'https://gemini.google.com',
      expect.anything(),
    );
  });
});

// ---------------------------------------------------------------------------
// queryImage integration flow
// ---------------------------------------------------------------------------
describe('queryImage', () => {
  it('calls ensureOnGemini, fillPrompt, submitPrompt in order and returns result with conversationId', async () => {
    // Start on a non-gemini URL so ensureOnGemini triggers a goto
    mockPage.url.mockReturnValue('https://google.com');

    const fillMock = vi.fn().mockResolvedValue(undefined);
    const clickMock = vi.fn().mockResolvedValue(undefined);
    mockPage.locator.mockReturnValue({
      first: vi.fn().mockReturnValue({
        count: vi.fn().mockResolvedValue(1),
        fill: fillMock,
        click: clickMock,
        waitFor: vi.fn().mockResolvedValue(undefined),
        getAttribute: vi.fn().mockResolvedValue(null),
      }),
    });

    const fakeResponse = makeResponseMock(100_000);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    // After goto, url changes to gemini
    mockPage.goto.mockImplementation(async () => {
      mockPage.url.mockReturnValue('https://gemini.google.com/app/conv001');
    });

    const result = await queryImage(mockPage as any, 'draw a cat', 5000);

    // ensureOnGemini should have triggered navigation
    expect(mockPage.goto).toHaveBeenCalledWith(
      'https://gemini.google.com',
      expect.anything(),
    );
    // fillPrompt should have used the locator path
    expect(fillMock).toHaveBeenCalledWith('draw a cat');
    // submitPrompt should have clicked
    expect(clickMock).toHaveBeenCalled();
    // Result should have buffer (small image) and conversationId
    expect(result.buffer).toBeInstanceOf(Buffer);
    expect(result.conversationId).toBe('conv001');
  });

  it('returns filePath for large image', async () => {
    mockPage.url.mockReturnValue('https://gemini.google.com/app/conv002');

    const fakeResponse = makeResponseMock(500_000);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const result = await queryImage(mockPage as any, 'draw something big', 5000);

    expect(result.filePath).toMatch(/\/tmp\/gemini-images\//);
    expect(result.buffer).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// iterateImage integration flow
// ---------------------------------------------------------------------------
describe('iterateImage', () => {
  it('navigates to conversation URL then submits prompt and returns correct conversationId', async () => {
    mockPage.url.mockReturnValue('https://gemini.google.com/app/abc123');

    // After goto to the conversation, url reflects the conversation
    mockPage.goto.mockImplementation(async (url: string) => {
      const match = /\/app\/([a-zA-Z0-9_-]+)/.exec(url);
      if (match) {
        mockPage.url.mockReturnValue(`https://gemini.google.com/app/${match[1]}`);
      }
    });

    const fakeResponse = makeResponseMock(100_000);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const result = await iterateImage(mockPage as any, 'conv123', 'make it blue', 5000);

    expect(mockPage.goto).toHaveBeenCalledWith(
      'https://gemini.google.com/app/conv123',
      expect.anything(),
    );
    expect(result.conversationId).toBe('conv123');
  });
});

// ---------------------------------------------------------------------------
// waitForImage DOM fallback
// ---------------------------------------------------------------------------
describe('waitForImage DOM fallback', () => {
  it('uses DOM locator when network response times out', async () => {
    mockPage.url.mockReturnValue('https://gemini.google.com/app/fallback-test');

    const waitForMock = vi.fn().mockResolvedValue(undefined);
    const getAttributeMock = vi.fn().mockResolvedValue('https://example.com/img.webp');

    mockPage.locator.mockReturnValue({
      first: vi.fn().mockReturnValue({
        count: vi.fn().mockResolvedValue(0),
        waitFor: waitForMock,
        getAttribute: getAttributeMock,
        fill: vi.fn().mockResolvedValue(undefined),
        click: vi.fn().mockResolvedValue(undefined),
      }),
    });

    // waitForResponse times out — Playwright rejects with an error
    mockPage.waitForResponse.mockRejectedValue(new Error('Timeout 50ms exceeded.'));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(50);

    expect(waitForMock).toHaveBeenCalled();
    expect(getAttributeMock).toHaveBeenCalled();
    expect(result.buffer).toBeNull();
    expect(result.filePath).toBeNull();
    expect(result.mimeType).toBe('image/webp');
    expect(result.conversationId).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// pruneImageDir
// ---------------------------------------------------------------------------
describe('pruneImageDir', () => {
  const IMAGE_DIR = '/tmp/gemini-images';

  it('calls unlink for files older than 1 hour but not for recent files', async () => {
    const now = Date.now();
    const twoHoursAgo = now - 2 * 60 * 60 * 1000;

    vi.mocked(readdir).mockResolvedValue(['old.webp', 'new.webp'] as any);
    vi.mocked(stat).mockImplementation(async (fp: any) => {
      if (String(fp).includes('old.webp')) {
        return { mtimeMs: twoHoursAgo } as any;
      }
      return { mtimeMs: now } as any;
    });

    pruneImageDir();

    // Let all microtasks/promises resolve
    await flushMicrotasks();
    // Allow nested promise chains to flush
    await flushMicrotasks();

    expect(unlink).toHaveBeenCalledWith(`${IMAGE_DIR}/old.webp`);
    expect(unlink).not.toHaveBeenCalledWith(`${IMAGE_DIR}/new.webp`);
  });

  it('does not throw when readdir fails', async () => {
    vi.mocked(readdir).mockRejectedValue(new Error('ENOENT'));

    // pruneImageDir is fire-and-forget — must not throw synchronously
    expect(() => pruneImageDir()).not.toThrow();

    // Also must not cause unhandled rejection
    await flushMicrotasks();
  });
});

// ---------------------------------------------------------------------------
// Mime type extension routing
// ---------------------------------------------------------------------------
describe('image file extension from mime type', () => {
  const LARGE = 500_000; // > 400KB threshold so file gets written

  it('uses .png extension for image/png mime type', async () => {
    const fakeResponse = makeResponseMock(LARGE, 'image/png');
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.filePath).toBeTruthy();
    expect(result.filePath).toMatch(/\.png$/);
  });

  it('uses .jpg extension for image/jpeg mime type', async () => {
    const fakeResponse = makeResponseMock(LARGE, 'image/jpeg');
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.filePath).toBeTruthy();
    expect(result.filePath).toMatch(/\.jpg$/);
  });

  it('uses .webp extension for image/webp mime type', async () => {
    const fakeResponse = makeResponseMock(LARGE, 'image/webp');
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.filePath).toBeTruthy();
    expect(result.filePath).toMatch(/\.webp$/);
  });
});
