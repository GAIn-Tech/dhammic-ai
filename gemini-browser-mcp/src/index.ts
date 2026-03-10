import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { logger } from './logger.js';
import {
  getPage,
  closeBrowser,
  isLoggedIn,
  isBrowserRunning,
  getProfileDir,
  setProfileDir,
} from './browser.js';
import { queryImage, iterateImage } from './gemini.js';
import * as fs from 'node:fs/promises';

const server = new McpServer({ name: 'gemini-browser-mcp', version: '1.0.0' });

// Prevent Chromium orphan processes when MCP host closes stdin
process.stdin.on('end', async () => {
  logger.info('stdin closed, shutting down');
  await closeBrowser();
  process.exit(0);
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

      const deadline = Date.now() + timeout_ms;
      const pollIntervalMs = 2000;

      while (Date.now() < deadline) {
        const url = page.url();
        if (url.includes('gemini.google.com/app')) {
          logger.info('Login confirmed', { url });
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
        await new Promise<void>((resolve) => setTimeout(resolve, pollIntervalMs));
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
      const message = error instanceof Error ? error.message : String(error);
      logger.error('gemini_login failed', error);
      return {
        isError: true,
        content: [{ type: 'text' as const, text: message }],
      };
    }
  },
);

// ─── Tool 2: gemini_query_image ────────────────────────────────────────────

server.tool(
  'gemini_query_image',
  'Send a prompt to Gemini and capture the generated image. Returns base64 data or saves to a file.',
  {
    prompt: z.string().min(1).max(4096),
    wait_timeout_ms: z
      .number()
      .int()
      .min(10000)
      .max(120000)
      .default(90000),
    output_format: z.enum(['base64', 'file']).default('base64'),
    output_path: z.string().max(512).optional(),
  },
  async ({ prompt, wait_timeout_ms, output_format, output_path }) => {
    try {
      const page = await getPage();
      const result = await queryImage(page, prompt, wait_timeout_ms);

      const content: Array<
        | { type: 'text'; text: string }
        | { type: 'image'; data: string; mimeType: string }
      > = [];

      // Always include conversation ID
      content.push({
        type: 'text' as const,
        text: `conversation_id: ${result.conversationId}`,
      });

      if (output_format === 'file') {
        // Determine destination path
        const destPath =
          output_path ??
          (result.filePath ?? `/tmp/gemini-images/output-${Date.now()}.webp`);

        if (result.filePath && result.filePath !== destPath) {
          // Move the already-written file if caller specified a different path
          await fs.rename(result.filePath, destPath).catch(async () => {
            // Cross-device move: copy then delete
            await fs.copyFile(result.filePath!, destPath);
            await fs.unlink(result.filePath!);
          });
        } else if (result.buffer) {
          await fs.writeFile(destPath, result.buffer);
        }
        content.push({ type: 'text' as const, text: `Image saved to: ${destPath}` });
      } else {
        // base64 output
        if (result.buffer) {
          content.push({
            type: 'image' as const,
            data: result.buffer.toString('base64'),
            mimeType: result.mimeType,
          });
        } else if (result.filePath) {
          // Large image was written to disk; read it back as base64
          const buf = await fs.readFile(result.filePath);
          content.push({
            type: 'image' as const,
            data: buf.toString('base64'),
            mimeType: result.mimeType,
          });
        } else {
          // DOM fallback: no binary data available
          content.push({
            type: 'text' as const,
            text: 'Image rendered in browser but binary data was not captured (DOM fallback). Use output_format="file" or inspect the browser window.',
          });
        }
      }

      return { content };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      logger.error('gemini_query_image failed', error);
      return {
        isError: true,
        content: [{ type: 'text' as const, text: message }],
      };
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
      .regex(/^[a-zA-Z0-9_-]+$/, 'Invalid conversation ID')
      .max(128),
    refinement_prompt: z.string().min(1).max(4096),
    wait_timeout_ms: z
      .number()
      .int()
      .min(10000)
      .max(120000)
      .default(90000),
  },
  async ({ conversation_id, refinement_prompt, wait_timeout_ms }) => {
    try {
      const page = await getPage();
      const result = await iterateImage(page, conversation_id, refinement_prompt, wait_timeout_ms);

      const content: Array<
        | { type: 'text'; text: string }
        | { type: 'image'; data: string; mimeType: string }
      > = [];

      content.push({
        type: 'text' as const,
        text: `conversation_id: ${result.conversationId}`,
      });

      if (result.buffer) {
        content.push({
          type: 'image' as const,
          data: result.buffer.toString('base64'),
          mimeType: result.mimeType,
        });
      } else if (result.filePath) {
        const buf = await fs.readFile(result.filePath);
        content.push({
          type: 'image' as const,
          data: buf.toString('base64'),
          mimeType: result.mimeType,
        });
      } else {
        content.push({
          type: 'text' as const,
          text: 'Image rendered in browser but binary data was not captured (DOM fallback).',
        });
      }

      return { content };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      logger.error('gemini_iterate_image failed', error);
      return {
        isError: true,
        content: [{ type: 'text' as const, text: message }],
      };
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

      const loggedIn = browserRunning ? await isLoggedIn() : false;

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
      const message = error instanceof Error ? error.message : String(error);
      logger.error('gemini_get_session_status failed', error);
      return {
        isError: true,
        content: [{ type: 'text' as const, text: message }],
      };
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
