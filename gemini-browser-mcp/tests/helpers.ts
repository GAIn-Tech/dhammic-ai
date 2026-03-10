import { vi } from 'vitest';

/**
 * Shared mock Page factory. Pass a URL to override the default.
 * Returns a minimal Playwright Page mock with the locator chain needed by gemini.ts.
 */
export function makeMockPage(url = 'https://gemini.google.com/app/abc123') {
  return {
    url: vi.fn().mockReturnValue(url),
    goto: vi.fn().mockResolvedValue(null),
    fill: vi.fn().mockResolvedValue(null),
    keyboard: {
      press: vi.fn().mockResolvedValue(null),
      type: vi.fn().mockResolvedValue(null),
    },
    locator: vi.fn().mockReturnValue({
      waitFor: vi.fn(),
      first: vi.fn().mockReturnValue({
        count: vi.fn().mockResolvedValue(0),
        fill: vi.fn().mockResolvedValue(null),
        click: vi.fn().mockResolvedValue(null),
        waitFor: vi.fn().mockResolvedValue(undefined),
        getAttribute: vi.fn().mockResolvedValue('https://example.com/img.webp'),
      }),
    }),
    waitForResponse: vi.fn().mockReturnValue(new Promise(() => {})),
    on: vi.fn(),
  };
}

/**
 * Shared mock network response factory.
 * Creates a fake Playwright Response with the given byte length and MIME type.
 */
export function makeResponseMock(byteLength: number, mimeType = 'image/webp') {
  const buf = Buffer.alloc(byteLength, 0xaa);
  return {
    url: () => 'https://generativelanguage.googleapis.com/v1/image',
    headers: () => ({ 'content-type': mimeType }),
    body: vi.fn().mockResolvedValue(buf),
  };
}

/**
 * Flush the microtask queue (await one event loop tick).
 * Useful for waiting for fire-and-forget async operations to settle.
 */
export async function flushMicrotasks(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

/** Fallback conversation ID regex — matches the fallback-<uuid> format. */
export const FALLBACK_ID_PATTERN =
  /^fallback-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
