# gemini-browser-mcp — Feature Walkthrough

## What It Is

An MCP (Model Context Protocol) server that drives `gemini.google.com` via
Playwright, enabling Claude and other MCP-compatible AI tools to generate and
iterate on images through the Gemini UI without using the REST API.

## Architecture at a Glance

```
MCP Host (Claude / opencode)
   │  JSON-RPC over stdio
   ▼
src/index.ts   ← 4 MCP tools, input validation, error wrapping
   │
   ├── src/browser.ts  ← Chromium singleton, durable OAuth profile,
   │                      navigation allowlist, stealth patches
   └── src/gemini.ts   ← Prompt fill, submit, pre-armed response capture,
                          image size routing, temp file management
```

## One-Time Setup

```bash
# 1. Install dependencies
cd gemini-browser-mcp && bun install

# 2. Install Playwright's system Chrome
bunx playwright install chrome

# 3. Build
bun run build

# 4. Add to MCP config (see claude-mcp-config.json for template)
#    then log in once:
```

## Tool Walkthrough

### 1. `gemini_login` — log in once, session persists forever

```json
{ "tool": "gemini_login" }
```

Opens a headed Chrome window (reuses `~/.gemini-mcp/profile/` if it exists).
The tool polls until the URL matches `gemini.google.com/app`, then returns:

```json
{ "success": true, "session_valid": true, "profile_path": "/home/user/.gemini-mcp/profile" }
```

After this completes, close the browser — the OAuth cookies are persisted to
disk and will be reused automatically on every subsequent call.

---

### 2. `gemini_query_image` — generate an image from a prompt

```json
{
  "tool": "gemini_query_image",
  "prompt": "a tiny glowing banana floating in deep space, cinematic lighting",
  "wait_timeout_ms": 90000
}
```

What happens internally:
1. `ensureOnGemini()` — navigates to gemini.google.com if not already there
2. `armImageCapture()` — registers `waitForResponse` listener **before** typing
3. `fillPrompt()` — types the prompt into the input box
4. `submitPrompt()` — clicks Send (or presses Enter as fallback)
5. `waitForImage()` — waits for `generativelanguage.googleapis.com` image response
6. Images ≤400KB → returned as MCP `image/webp` content block (base64)
   Images >400KB → written to `/tmp/gemini-images/<uuid>.webp`, path returned

Returns:
```
conversation_id: aB3xKm9...   ← use this for iterations
[image content block]
```

---

### 3. `gemini_iterate_image` — refine the previous result

```json
{
  "tool": "gemini_iterate_image",
  "conversation_id": "aB3xKm9...",
  "refinement_prompt": "make it more vibrant, add a rainbow halo",
  "wait_timeout_ms": 90000
}
```

Navigates to `gemini.google.com/app/<conversation_id>`, arms capture, then
submits the refinement prompt. Same image return logic as above.

---

### 4. `gemini_get_session_status` — check without side effects

```json
{ "tool": "gemini_get_session_status" }
```

Returns current state without launching the browser:
```json
{
  "is_logged_in": true,
  "profile_dir": "/home/user/.gemini-mcp/profile",
  "browser_running": true,
  "current_url": "https://gemini.google.com/app/aB3xKm9..."
}
```

## Security Properties

| Property | Implementation |
|----------|---------------|
| Profile path containment | `sanitizeProfileDir()` — must be inside `~/.gemini-mcp/` |
| Output path containment | `sanitizeOutputPath()` — must be inside `/tmp/gemini-images/` |
| Navigation restriction | Allowlist: `gemini.google.com` + `accounts.google.com` only |
| Main-frame-only enforcement | `frame !== frame.page().mainFrame()` guard |
| Initial page coverage | Allowlist attached to pre-existing pages at init |
| Prompt length limit | `z.string().max(4096)` on all prompt inputs |
| Conversation ID validation | `/^[a-zA-Z0-9_-]+$/` regex in 2 places |
| No auth token leakage | Only `buffer` bytes extracted, URL never returned |
| Stderr-only logging | `process.stderr.write` — stdout reserved for MCP JSON-RPC |

## Test Coverage (9/9)

```
sanitizeProfileDir — rejects /tmp/evil, accepts ~/.gemini-mcp/custom
extractConversationId — correct ID from URL, fallback for non-matching
armImageCapture — returns { waitForImage } before submit
image routing: >400KB → writeFile + filePath, ≤400KB → buffer in memory
navigateToConversation — rejects ../etc/passwd, accepts abc123XYZ
```

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `GEMINI_MCP_PROFILE_DIR` | `~/.gemini-mcp/profile` | Override OAuth profile location |
| `GEMINI_MCP_HEADLESS` | `false` | Not yet wired (headed for login visibility) |
| `LOG_LEVEL` | `info` | Set to `debug` for verbose Playwright logging |
