import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import * as path from 'node:path';
import { logger } from './logger.js';
import {
  getPage,
  withPage,
  closeBrowser,
  checkLoginStatus,
  isBrowserRunning,
  getProfileDir,
  setProfileDir,
} from './browser.js';
import { IMAGE_DIR, CONVERSATION_ID_REGEX, queryImage, iterateImage, waitForLogin } from './gemini.js';
import type { ImageResult } from './gemini.js';
import * as fs from 'node:fs/promises';

/** Validate that a caller-supplied output path stays within IMAGE_DIR. */
export function sanitizeOutputPath(input: string): string {
  const resolved = path.resolve(input);
  if (!resolved.startsWith(IMAGE_DIR + path.sep) && resolved !== IMAGE_DIR) {
    throw new Error(
      `output_path "${resolved}" is outside allowed directory "${IMAGE_DIR}"`,
    );
  }
  return resolved;
}

// ─── Shared constants and types ────────────────────────────────────────────

const MAX_PROMPT_LENGTH = 4096;

const waitTimeoutSchema = z.number().int().min(10000).max(120000).default(90000);

type ContentItem = { type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string };

function buildResponseContent(result: ImageResult): ContentItem[] {
  return [{ type: 'text', text: `conversation_id: ${result.conversationId}` }];
}

// ─── Shared helpers ────────────────────────────────────────────────────────

async function imageResultToContent(
  result: ImageResult,
): Promise<ContentItem[]> {
  if (result.buffer) {
    return [{ type: 'image' as const, data: result.buffer.toString('base64'), mimeType: result.mimeType }];
  }
  if (result.filePath) {
    const buf = await fs.readFile(result.filePath);
    fs.unlink(result.filePath).catch((e) => logger.warn('Failed to delete temp image file', { path: result.filePath, error: e instanceof Error ? e.message : String(e) }));
    return [{ type: 'image' as const, data: buf.toString('base64'), mimeType: result.mimeType }];
  }
  return [
    {
      type: 'text' as const,
      text: 'Image rendered in browser but binary data was not captured (DOM fallback). Use output_format="file" or inspect the browser window.',
    },
  ];
}

function makeErrorResponse(toolName: string, error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  logger.error(`${toolName} failed`, error);
  return {
    isError: true as const,
    content: [{ type: 'text' as const, text: message }],
  };
}

// ─── Server setup ─────────────────────────────────────────────────────────

const server = new McpServer({ name: 'gemini-browser-mcp', version: '1.0.0' });

// Prevent Chromium orphan processes when MCP host closes stdin
process.stdin.on('end', () => {
  logger.info('stdin closed, shutting down');
  closeBrowser().finally(() => process.exit(0));
});

// ─── Tool 1: gemini_login ──────────────────────────────────────────────────

server.tool(
  'gemini_login',
  'Open a headed browser and log in to Gemini. Polls until the URL contains gemini.google.com/app.',
  {
    profile_dir: z.string().max(512).optional(),
    timeout_ms: z.number().int().min(30000).max(300000).default(120000),
  },
  async ({ profile_dir, timeout_ms }) => {
    try {
      if (profile_dir) {
        setProfileDir(profile_dir);
      }

      const page = await getPage();
      await page.goto('https://gemini.google.com', { waitUntil: 'domcontentloaded' });

      const loggedIn = await waitForLogin(page, timeout_ms);
      if (loggedIn) {
        logger.info('Login confirmed', { url: page.url() });
        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify({
                success: true,
                session_valid: true,
                profile_path: getProfileDir(),
              }),
            },
          ],
        };
      }

      // Timed out waiting for login
      return {
        isError: true,
        content: [
          {
            type: 'text' as const,
            text: `Login timed out after ${timeout_ms}ms. Please complete sign-in in the browser window and retry.`,
          },
        ],
      };
    } catch (error) {
      return makeErrorResponse('gemini_login', error);
    }
  },
);

// ─── Tool 2: gemini_query_image ────────────────────────────────────────────

server.tool(
  'gemini_query_image',
  'Send a prompt to Gemini and capture the generated image. Returns base64 data or saves to a file.',
  {
    prompt: z.string().min(1).max(MAX_PROMPT_LENGTH),
    wait_timeout_ms: waitTimeoutSchema,
    output_format: z.enum(['base64', 'file']).default('base64'),
    output_path: z.string().max(512).optional(),
  },
  async ({ prompt, wait_timeout_ms, output_format, output_path }) => {
    try {
      const result = await withPage((page) => queryImage(page, prompt, wait_timeout_ms));

      const content: ContentItem[] = buildResponseContent(result);

      if (output_format === 'file') {
        // Determine and validate destination path (must stay inside IMAGE_DIR).
        const rawDest =
          output_path ??
          (result.filePath ?? `${IMAGE_DIR}/output-${Date.now()}.webp`);
        const destPath = sanitizeOutputPath(rawDest);

        if (result.filePath && result.filePath !== destPath) {
          // Move the already-written file if caller specified a different path
          await fs.rename(result.filePath, destPath).catch(async () => {
            // Cross-device move: copy then delete
            await fs.copyFile(result.filePath!, destPath);
            await fs.unlink(result.filePath!).catch(() => {});
          });
        } else if (result.buffer) {
          await fs.writeFile(destPath, result.buffer);
        }
        content.push({ type: 'text' as const, text: `Image saved to: ${destPath}` });
      } else {
        // base64 output
        const imageContent = await imageResultToContent(result);
        content.push(...imageContent);
      }

      return { content };
    } catch (error) {
      return makeErrorResponse('gemini_query_image', error);
    }
  },
);

// ─── Tool 3: gemini_iterate_image ─────────────────────────────────────────

server.tool(
  'gemini_iterate_image',
  'Continue an existing Gemini conversation to refine a generated image.',
  {
    conversation_id: z
      .string()
      .regex(CONVERSATION_ID_REGEX, 'Invalid conversation ID')
      .max(128),
    refinement_prompt: z.string().min(1).max(MAX_PROMPT_LENGTH),
    wait_timeout_ms: waitTimeoutSchema,
  },
  async ({ conversation_id, refinement_prompt, wait_timeout_ms }) => {
    try {
      const result = await withPage((page) =>
        iterateImage(page, conversation_id, refinement_prompt, wait_timeout_ms),
      );

      const content: ContentItem[] = buildResponseContent(result);

      const imageContent = await imageResultToContent(result);
      content.push(...imageContent);

      return { content };
    } catch (error) {
      return makeErrorResponse('gemini_iterate_image', error);
    }
  },
);

// ─── Tool 4: gemini_get_session_status ────────────────────────────────────

server.tool(
  'gemini_get_session_status',
  'Return the current browser and session state without launching a new browser.',
  {},
  async () => {
    try {
      let currentUrl: string | null = null;
      const browserRunning = isBrowserRunning();

      if (browserRunning) {
        try {
          const page = await getPage();
          currentUrl = page.url();
        } catch {
          // page unavailable
        }
      }

      // Use read-only status check — avoids triggering a page navigation that
      // could abort an in-flight image generation.
      const loggedIn = browserRunning ? await checkLoginStatus() : false;

      return {
        content: [
          {
            type: 'text' as const,
            text: JSON.stringify({
              is_logged_in: loggedIn,
              profile_dir: getProfileDir(),
              browser_running: browserRunning,
              current_url: currentUrl,
            }),
          },
        ],
      };
    } catch (error) {
      return makeErrorResponse('gemini_get_session_status', error);
    }
  },
);

// ─── Entry point ──────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info('gemini-browser-mcp started');
}

main().catch((e) => {
  process.stderr.write(
    JSON.stringify({ level: 'error', message: 'Fatal', error: String(e) }) + '\n',
  );
  process.exit(1);
});
