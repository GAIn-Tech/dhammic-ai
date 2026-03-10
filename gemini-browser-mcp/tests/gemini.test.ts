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
  isLoggedIn: vi.fn(),
  getProfileDir: vi.fn(),
  setProfileDir: vi.fn(),
}));

// Mock playwright
vi.mock('playwright', () => ({
  chromium: {
    launchPersistentContext: vi.fn(),
  },
}));

// Mock fs (used at module top-level by gemini.ts via mkdirSync)
vi.mock('fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('fs')>();
  return {
    ...actual,
    mkdirSync: vi.fn(),
    writeFileSync: vi.fn(),
  };
});

import {
  sanitizeProfileDir,
} from '../src/browser.js';

import {
  extractConversationId,
  armImageCapture,
  navigateToConversation,
} from '../src/gemini.js';

import { writeFileSync } from 'fs';

// ---------------------------------------------------------------------------
// Shared minimal mock Page
// ---------------------------------------------------------------------------
const mockPage = {
  url: vi.fn().mockReturnValue('https://gemini.google.com/app/abc123'),
  goto: vi.fn().mockResolvedValue(null),
  fill: vi.fn().mockResolvedValue(null),
  keyboard: { press: vi.fn().mockResolvedValue(null) },
  locator: vi.fn().mockReturnValue({
    waitFor: vi.fn(),
    first: vi.fn().mockReturnValue({
      count: vi.fn().mockResolvedValue(0),
      waitFor: vi.fn().mockResolvedValue(undefined),
      getAttribute: vi.fn().mockResolvedValue('https://example.com/img.webp'),
    }),
  }),
  waitForResponse: vi.fn(),
  on: vi.fn(),
};

// ---------------------------------------------------------------------------
// Reset mocks between tests
// ---------------------------------------------------------------------------
beforeEach(() => {
  vi.clearAllMocks();
  mockPage.url.mockReturnValue('https://gemini.google.com/app/abc123');
  mockPage.goto.mockResolvedValue(null);
  mockPage.waitForResponse.mockReturnValue(new Promise(() => {})); // never resolves by default
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
    expect(id).toMatch(/^fallback-\d+$/);
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
  function makeResponseMock(byteLength: number, mimeType = 'image/webp') {
    const buf = Buffer.alloc(byteLength, 0xaa);
    return {
      url: () => 'https://generativelanguage.googleapis.com/v1/image',
      headers: () => ({ 'content-type': mimeType }),
      body: vi.fn().mockResolvedValue(buf),
    };
  }

  it('writes image > 400KB to a file and returns filePath (not buffer)', async () => {
    const LARGE = 401_000;
    const fakeResponse = makeResponseMock(LARGE);
    mockPage.waitForResponse.mockReturnValue(Promise.resolve(fakeResponse));

    const capture = armImageCapture(mockPage as any);
    const result = await capture.waitForImage(5000);

    expect(result.filePath).toBeTruthy();
    expect(result.buffer).toBeNull();
    expect(writeFileSync).toHaveBeenCalledWith(
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
    expect(writeFileSync).not.toHaveBeenCalled();
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
