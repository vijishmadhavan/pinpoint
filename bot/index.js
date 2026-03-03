/**
 * Pinpoint — WhatsApp Bot with Gemini AI (48 tools + skills system + file receive)
 *
 * Self-chat mode: message yourself to search documents.
 * Gemini understands natural language → calls tools → replies naturally.
 * Echo prevention: 3 layers (fromMe, message ID dedup, sent-text tracker).
 * Conversation memory: last 20 messages passed to Gemini for context.
 * File receive: send files from phone → saved to computer (Downloads/Pinpoint/).
 *
 * Tools (38): search, files, faces, images, data, write/create, PDF, archive, download, smart-ops, facts
 *   See skills/*.md for full tool descriptions
 */

const {
  default: makeWASocket,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
  makeCacheableSignalKeyStore,
  downloadMediaMessage,
  getContentType,
} = require("@whiskeysockets/baileys");
const { GoogleGenAI } = require("@google/genai");
const pino = require("pino");
const qrcode = require("qrcode-terminal");
const { createHash } = require("crypto");
const { readFileSync, writeFileSync, statSync, existsSync, mkdirSync, readdirSync, unlinkSync } = require("fs");
const pathModule = require("path");

// Load .env from project root
require("dotenv").config({ path: pathModule.join(__dirname, "..", ".env") });

// Silent logger for Baileys
const logger = pino({ level: "silent" });

// --- Config ---
const API_URL = "http://localhost:5123";
const PREFIX = "[pinpoint]";
const AUTH_DIR = "./auth";
const MAX_RESULTS = 5;
const MAX_FILES_TO_SEND = 3;
const MAX_IMAGE_SIZE = 16 * 1024 * 1024;
const MAX_DOC_SIZE = 100 * 1024 * 1024;
const TEXT_CHUNK_LIMIT = 4000;
const GEMINI_MODEL = "gemini-3-flash-preview";
// No tool-calling timeout — batch mode handles long operations in single calls (not token-burning loops)
const IDLE_TIMEOUT_MS = 60 * 60 * 1000; // 60 minutes — auto-reset conversation
const MAX_HISTORY_MESSAGES = 20; // last 20 messages passed to Gemini
const MAX_INLINE_IMAGES = 5; // Max images sent as visual data per turn

const DEBOUNCE_MS = 1500; // Combine rapid messages within 1.5s

// Processing lock: prevent concurrent Gemini calls per chat (causes context loss)
const activeRequests = new Map(); // chatJid → { msg, startTime, id }
const lastImage = new Map(); // chatJid → { mimeType, data (base64), path, ts } — for follow-up re-injection (2 min TTL)
let requestCounter = 0;

// Persistent memory: on by default (all local, single user)
let memoryEnabled = true;
let memoryContext = ""; // Loaded from API, injected into system prompt

// Allowed users: phone numbers/LIDs that can use Pinpoint (managed via /allow, /revoke)
const allowedUsers = new Set();
const allowedSessions = new Map(); // chatJid → last activity timestamp (active session)

// Cost tracking: per-session token usage (OpenCode-inspired)
const sessionCosts = {}; // chatJid → { input, output, rounds, started }
const TOKEN_COST_INPUT = 0.15 / 1_000_000;   // Gemini Flash $/token (input)
const TOKEN_COST_OUTPUT = 0.60 / 1_000_000;   // Gemini Flash $/token (output)

function trackTokens(chatJid, response) {
  const usage = response.usageMetadata;
  if (!usage) return;
  if (!sessionCosts[chatJid]) {
    sessionCosts[chatJid] = { input: 0, output: 0, rounds: 0, started: Date.now() };
  }
  const s = sessionCosts[chatJid];
  s.input += usage.promptTokenCount || 0;
  s.output += usage.candidatesTokenCount || 0;
  s.rounds++;
  return { input: usage.promptTokenCount || 0, output: usage.candidatesTokenCount || 0 };
}

function formatTokens(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function getCostSummary(chatJid) {
  const s = sessionCosts[chatJid];
  if (!s || s.rounds === 0) return "No token usage in this session.";
  const cost = (s.input * TOKEN_COST_INPUT + s.output * TOKEN_COST_OUTPUT);
  const elapsed = Math.round((Date.now() - s.started) / 60000);
  return `*Session tokens:* ${formatTokens(s.input + s.output)} (input: ${formatTokens(s.input)}, output: ${formatTokens(s.output)})\n*Rounds:* ${s.rounds}\n*Estimated cost:* $${cost.toFixed(4)}\n*Duration:* ${elapsed} min`;
}

function isAllowedUser(jid) {
  const number = jid?.split("@")[0];
  return allowedUsers.has(number);
}

async function loadAllowedUsers() {
  try {
    const setting = await apiGet("/setting?key=allowed_users");
    if (setting.value) {
      const numbers = setting.value.split(",").filter(n => n.trim());
      numbers.forEach(n => allowedUsers.add(n.trim()));
    }
  } catch (_) {}
}

async function saveAllowedUsers() {
  try {
    await apiPost(`/setting?key=allowed_users&value=${[...allowedUsers].join(",")}`, {});
  } catch (_) {}
}

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

// Mime type → extension (for received files without filename)
const MIME_TO_EXT = {
  "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
  "image/bmp": ".bmp", "image/gif": ".gif", "image/tiff": ".tiff",
  "video/mp4": ".mp4", "video/mkv": ".mkv", "video/avi": ".avi",
  "video/quicktime": ".mov", "video/webm": ".webm",
  "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/wav": ".wav",
  "audio/mp4": ".m4a", "audio/aac": ".aac",
  "application/pdf": ".pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
  "text/plain": ".txt", "text/csv": ".csv",
};

// --- Detect system paths dynamically (never hardcode!) ---
const os = require("os");
const HOME_DIR = os.homedir();
const PLATFORM = process.platform; // "linux", "win32", "darwin"

// On WSL, real Windows home is under /mnt/c/Users/<name>
// Detect by checking if /mnt/c/Users exists
let WIN_HOME = null;
const wslUserPath = `/mnt/c/Users/${pathModule.basename(HOME_DIR)}`;
if (existsSync(wslUserPath)) {
  WIN_HOME = wslUserPath;
}
const USER_HOME = WIN_HOME || HOME_DIR;
const DOWNLOADS = pathModule.join(USER_HOME, "Downloads");
const DOCUMENTS = pathModule.join(USER_HOME, "Documents");
const DESKTOP = pathModule.join(USER_HOME, "Desktop");
const PICTURES = pathModule.join(USER_HOME, "Pictures");
const DEFAULT_SAVE_FOLDER = pathModule.join(DOWNLOADS, "Pinpoint");

console.log(`[Pinpoint] User home: ${USER_HOME}`);

// --- LLM setup (Gemini default, Ollama optional) ---
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || ""; // e.g. "qwen3.5:9b" — set to use local LLM instead of Gemini
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const USE_OLLAMA = !!OLLAMA_MODEL;

const ai = USE_OLLAMA ? null : new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const LLM_TAG = USE_OLLAMA ? "Ollama" : "Gemini";
if (USE_OLLAMA) console.log(`[Pinpoint] Using Ollama: ${OLLAMA_MODEL} at ${OLLAMA_URL}`);
else console.log(`[Pinpoint] Using Gemini: ${GEMINI_MODEL}`);

// --- Ollama adapter: translates Gemini format ↔ Ollama format ---
// So the rest of the code stays identical regardless of which LLM is used.
function geminiToolsToOllama(geminiTools) {
  // Gemini: [{ functionDeclarations: [{ name, description, parameters: { type: "OBJECT", properties, required } }] }]
  // Ollama: [{ type: "function", function: { name, description, parameters: { type: "object", ... } } }]
  if (!geminiTools?.[0]?.functionDeclarations) return [];
  return geminiTools[0].functionDeclarations.map(fd => ({
    type: "function",
    function: {
      name: fd.name,
      description: fd.description,
      parameters: lowerTypes(fd.parameters),
    },
  }));
}

function lowerTypes(schema) {
  if (!schema) return schema;
  const out = { ...schema };
  if (out.type) out.type = out.type.toLowerCase();
  if (out.properties) {
    out.properties = {};
    for (const [k, v] of Object.entries(schema.properties)) {
      out.properties[k] = lowerTypes(v);
    }
  }
  if (out.items) out.items = lowerTypes(out.items);
  return out;
}

function geminiContentsToOllama(contents, systemInstruction) {
  // Convert Gemini contents array to Ollama messages array
  const messages = [];
  if (systemInstruction) messages.push({ role: "system", content: systemInstruction });

  for (const entry of contents) {
    if (!entry?.parts) continue;
    const role = entry.role === "model" ? "assistant" : "user";

    // Check for function calls (model response with tool calls)
    const funcCalls = entry.parts.filter(p => p.functionCall);
    if (funcCalls.length > 0) {
      // Text part if any
      const textParts = entry.parts.filter(p => p.text).map(p => p.text).join("\n");
      messages.push({
        role: "assistant",
        content: textParts || "",
        tool_calls: funcCalls.map(p => ({
          id: `call_${p.functionCall.name}_${Date.now()}`,
          type: "function",
          function: { name: p.functionCall.name, arguments: p.functionCall.args || {} },
        })),
      });
      continue;
    }

    // Check for function responses (tool results)
    const funcResponses = entry.parts.filter(p => p.functionResponse);
    if (funcResponses.length > 0) {
      for (const p of funcResponses) {
        messages.push({
          role: "tool",
          content: JSON.stringify(p.functionResponse.response?.result ?? ""),
        });
      }
      // Also include any text nudges (round-based efficiency hints)
      const textNudges = entry.parts.filter(p => p.text);
      for (const p of textNudges) {
        messages.push({ role: "system", content: p.text });
      }
      continue;
    }

    // Regular text + images (Ollama uses "images" array with base64 data)
    const text = entry.parts.filter(p => p.text).map(p => p.text).join("\n");
    const images = entry.parts.filter(p => p.inlineData).map(p => p.inlineData.data);
    if (text || images.length > 0) {
      const msg = { role, content: text || "" };
      if (images.length > 0) msg.images = images;
      messages.push(msg);
    }
  }
  return messages;
}

async function ollamaGenerate(contents, config, toolsDefs) {
  // Build Ollama chat request
  const messages = geminiContentsToOllama(contents, config?.systemInstruction);
  const body = {
    model: OLLAMA_MODEL,
    messages,
    stream: false,
    think: false, // Disable thinking — too slow for tool calling
  };
  if (toolsDefs) body.tools = geminiToolsToOllama(toolsDefs);

  const resp = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`Ollama error: ${resp.status} ${await resp.text()}`);
  const data = await resp.json();
  const msg = data.message || {};

  // Translate Ollama response → Gemini response shape
  const functionCalls = (msg.tool_calls || []).map(tc => ({
    name: tc.function.name,
    args: typeof tc.function.arguments === "string" ? JSON.parse(tc.function.arguments) : tc.function.arguments,
  }));

  return {
    text: msg.content || "",
    functionCalls: functionCalls.length > 0 ? functionCalls : null,
    candidates: [{
      content: {
        role: "model",
        parts: [
          ...(msg.content ? [{ text: msg.content }] : []),
          ...functionCalls.map(fc => ({ functionCall: { name: fc.name, args: fc.args } })),
        ],
      },
      finishReason: functionCalls.length > 0 ? "TOOL_CALLS" : "STOP",
    }],
    usageMetadata: {
      promptTokenCount: data.prompt_eval_count || 0,
      candidatesTokenCount: data.eval_count || 0,
    },
  };
}

// Unified LLM call — routes to Gemini or Ollama
async function llmGenerate({ model, contents, config, tools: toolsDefs }) {
  if (USE_OLLAMA) {
    return ollamaGenerate(contents, config, toolsDefs);
  }
  return ai.models.generateContent({ model, contents, config: { ...config, tools: toolsDefs } });
}

// --- Load skills from skills/*.md at startup (hierarchical: general + task-specific) ---
const SKILLS_DIR = pathModule.join(__dirname, "..", "skills");

// General skills: always injected (core rules, batch awareness, common mistakes)
const GENERAL_SKILL_FILES = ["batch-awareness.md", "common-mistakes.md", "core-rules.md"];

// Task-specific skills: injected based on user intent detection
const SKILL_CATEGORIES = {
  image: ["face-analysis.md", "image-analysis.md", "image-tools.md", "visual-search.md"],
  search: ["search.md"],
  data: ["data-analysis.md"],
  files: ["file-tools.md", "smart-ops.md", "archive-tools.md"],
  write: ["write-create.md", "pdf-tools.md"],
  media: ["video-search.md"],
  web: ["web-search.md", "download.md"],
  memory: ["memory.md"],
  code: ["python.md"],
};

// Intent detection keywords → categories
const INTENT_KEYWORDS = {
  image: /photo|image|picture|jpg|png|face|person|selfie|caption|detect|object|bounding|visual|heic|camera|screenshot/i,
  search: /find|search|where|which|document|file.*contain|look.*for|indexed/i,
  data: /excel|csv|spreadsheet|column|row|data|analyze|chart|graph|pandas|filter|sort/i,
  files: /move|copy|rename|delete|duplicate|folder|list|organize|clean.*up|batch|zip|unzip|compress|extract|archive/i,
  write: /write|create|pdf|merge|split|combine|generate/i,
  media: /video|mp4|clip|frame|scene/i,
  web: /download|url|web|search.*online|internet|website/i,
  memory: /remember|forget|memory|preference/i,
  code: /python|code|script|run|execute|program/i,
};

const _skillCache = {};  // filename → content
function _loadSkill(filename) {
  if (!_skillCache[filename]) {
    try {
      _skillCache[filename] = readFileSync(pathModule.join(SKILLS_DIR, filename), "utf-8");
    } catch (e) {
      _skillCache[filename] = "";
    }
  }
  return _skillCache[filename];
}

// Preload all skills at startup
try {
  const { readdirSync } = require("fs");
  const allFiles = readdirSync(SKILLS_DIR).filter(f => f.endsWith(".md")).sort();
  for (const file of allFiles) _loadSkill(file);
  console.log(`[Pinpoint] Loaded ${allFiles.length} skills: ${allFiles.map(f => f.replace(".md", "")).join(", ")}`);
} catch (err) {
  console.log("[Pinpoint] No skills loaded:", err.message);
}

// Build general skills content (always included)
const generalSkillsContent = GENERAL_SKILL_FILES.map(f => _loadSkill(f)).filter(Boolean).join("\n\n");

// Detect user intent → return relevant skill categories
function detectIntentCategories(message) {
  const cats = new Set();
  for (const [cat, regex] of Object.entries(INTENT_KEYWORDS)) {
    if (regex.test(message)) cats.add(cat);
  }
  // Always include search (core functionality)
  if (cats.size === 0) cats.add("search");
  return cats;
}

// Build task-specific skills for a message
function getTaskSkills(message) {
  const cats = detectIntentCategories(message);
  const files = new Set();
  for (const cat of cats) {
    for (const f of (SKILL_CATEGORIES[cat] || [])) files.add(f);
  }
  // Don't duplicate general skills
  for (const f of GENERAL_SKILL_FILES) files.delete(f);
  return [...files].map(f => _loadSkill(f)).filter(Boolean).join("\n\n");
}

const SYSTEM_PROMPT_BASE = `You are Pinpoint, a local file assistant with full power over the user's files.
You search, read, analyze, organize, OCR, caption, and manage files on their computer.
All files stay local — nothing leaves the machine.

## How to Work
- Before each tool call, briefly think about what you need and which tool will get it.
- You may call tools across multiple rounds until you have the answer. When you have enough info, stop and answer.
- After receiving tool results, analyze what you learned before deciding on the next step.
- Don't call the same tool with identical arguments twice — you'll get the same result.
- Don't ask for the same info multiple ways — once is enough.
- Prefer batch tools (folder param, batch_move) over individual calls in a loop.
- If user sends a file, image, or link with NO instruction — ask what they want. Don't auto-analyze.
- If an image is already sent inline, you can SEE it — don't call read_file or other tools to look at it again.

## Context Priority
When multiple sources of info conflict, trust in this order:
1. Current user message (highest)
2. Recent conversation turns
3. Active tool results
4. Persistent memories
5. Older conversation history

${generalSkillsContent}

## System Paths
- Home: ${USER_HOME}
- Downloads: ${DOWNLOADS}
- Documents: ${DOCUMENTS}
- Desktop: ${DESKTOP}
- Pictures: ${PICTURES}
Use these full ABSOLUTE paths when the user says "Downloads" or "my Documents".

## Result References
When a tool returns many items (files, images, faces), the result is stored server-side and you receive a reference like @ref:1 with a preview.
To use these results in another tool (batch_move, compress_files, etc.), pass the @ref:N as the value — it will be resolved automatically.
Example: list_files returns @ref:1 (500 files) → batch_move({ sources: "@ref:1", destination: "/path" }) moves all 500.
`;

const USER_TZ = process.env.TZ || Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Kolkata";

function getSystemPrompt(userMessage = "") {
  const tz = USER_TZ;
  // Inject task-specific skills based on user message intent
  const taskSkills = userMessage ? getTaskSkills(userMessage) : "";
  let prompt = SYSTEM_PROMPT_BASE;
  if (taskSkills) prompt += `\n${taskSkills}\n`;
  prompt += `\nCurrent date and time: ${new Date().toLocaleString("en-IN", { timeZone: tz, day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: true, timeZoneName: "short" })}`;
  if (memoryEnabled && memoryContext) {
    prompt += `\n\n## Saved memories\n${memoryContext}`;
  } else if (memoryEnabled) {
    prompt += `\n\n## Saved memories\nNo memories saved yet.`;
  } else if (!memoryEnabled) {
    prompt += `\n\n## Memory\nMemory is currently OFF. If user asks to remember something, tell them to enable it with /memory on.`;
  }
  return prompt;
}

// --- Tool grouping: map each tool to intent categories (mirrors SKILL_CATEGORIES) ---
// Core tools are always included. Category tools added based on user message intent.
const CORE_TOOLS = new Set([
  "search_documents", "search_facts", "read_document", "read_file",
  "list_files", "send_file", "get_status", "calculate",
]);
const TOOL_GROUPS = {
  search: ["search_history", "grep_files", "index_file"],
  image: [
    "detect_faces", "crop_face", "find_person", "find_person_by_face",
    "count_faces", "compare_faces", "remember_face", "forget_face",
    "search_images_visual", "ocr", "detect_objects",
  ],
  data: ["analyze_data", "read_excel", "generate_chart", "extract_tables"],
  files: [
    "file_info", "move_file", "copy_file", "batch_move",
    "create_folder", "delete_file", "find_duplicates", "batch_rename",
    "compress_files", "extract_archive",
  ],
  write: [
    "write_file", "generate_excel", "merge_pdf", "split_pdf",
    "pdf_to_images", "images_to_pdf", "resize_image", "convert_image", "crop_image",
  ],
  media: ["search_video", "extract_frame"],
  web: ["web_search", "download_url"],
  memory: ["memory_save", "memory_search", "memory_delete", "memory_forget"],
  code: ["run_python"],
  archive: ["compress_files", "extract_archive"],
  automation: ["set_reminder", "list_reminders", "cancel_reminder", "watch_folder", "unwatch_folder", "list_watched"],
};

// Build filtered tools array based on user message intent
function getToolsForIntent(message) {
  const cats = detectIntentCategories(message);
  const allowedNames = new Set(CORE_TOOLS);
  for (const cat of cats) {
    for (const name of (TOOL_GROUPS[cat] || [])) allowedNames.add(name);
  }
  // automation tools always available (reminders, watch)
  for (const name of TOOL_GROUPS.automation) allowedNames.add(name);

  const filtered = tools[0].functionDeclarations.filter(fd => allowedNames.has(fd.name));
  console.log(`[Pinpoint] Tools: ${filtered.length}/${tools[0].functionDeclarations.length} (intents: ${[...cats].join(",")})`);
  return [{ functionDeclarations: filtered }];
}

// --- Tool declarations for Gemini ---
const tools = [{
  functionDeclarations: [
    {
      name: "search_documents",
      description: "Search indexed documents by keywords. Returns the exact matching section/paragraph (not just filenames). PREFERRED way to answer questions about document content — always try this FIRST before read_document or read_file. If the file is not indexed yet, use index_file first, then search. Can filter by file type and folder.",
      parameters: {
        type: "OBJECT",
        properties: {
          query: {
            type: "STRING",
            description: "Search keywords extracted from the user's message.",
          },
          file_type: {
            type: "STRING",
            description: "Filter by type: pdf, docx, xlsx, pptx, txt, csv, image, epub. Optional.",
          },
          folder: {
            type: "STRING",
            description: "Only search within this folder path. Optional.",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "read_document",
      description: "Read the full text of a document by its ID. Use ONLY when search_documents snippet isn't detailed enough and you need broader context — like summarizing an entire document, comparing two full documents, or translating. For specific questions (what does clause 7 say, what's the depreciation amount), search_documents already returns the exact section.",
      parameters: {
        type: "OBJECT",
        properties: {
          document_id: {
            type: "INTEGER",
            description: "The document ID from search results.",
          },
        },
        required: ["document_id"],
      },
    },
    {
      name: "read_excel",
      description: "Read specific cells or ranges from an Excel (.xlsx) file. Use when the user asks about specific cells, rows, columns, or ranges. For general search, use search_documents instead.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Absolute path to the .xlsx file.",
          },
          sheet_name: {
            type: "STRING",
            description: "Sheet name. Optional — defaults to first sheet.",
          },
          cell_range: {
            type: "STRING",
            description: "Excel range like 'A1:D10', 'B5', 'A:A'. Optional — defaults to first 20 rows.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "calculate",
      description: "Evaluate a mathematical expression. Supports +, -, *, /, **, %, parentheses, and functions like round(), abs(), min(), max(), sum(), sqrt(). Use for any arithmetic: sums, averages, percentages, conversions.",
      parameters: {
        type: "OBJECT",
        properties: {
          expression: {
            type: "STRING",
            description: "Math expression like '45230 * 0.18' or '(12000 + 8500 + 23000) / 3'.",
          },
        },
        required: ["expression"],
      },
    },
    {
      name: "list_files",
      description: "List files and folders in a directory. WORKFLOW: 1) Use sort_by='size' to find large files. 2) Use name_contains to search by filename. 3) Use recursive=true to search in subfolders. 4) Use filter_ext or filter_type to narrow by type. The response includes a 'largest' field when sorted by size. Do NOT call repeatedly with different params — if you can't find a file in the first result, try name_contains or recursive.",
      parameters: {
        type: "OBJECT",
        properties: {
          folder: {
            type: "STRING",
            description: "Folder path to list.",
          },
          sort_by: {
            type: "STRING",
            description: "Sort order: 'name' (default), 'date' (newest first), 'size' (largest first).",
          },
          filter_ext: {
            type: "STRING",
            description: "Filter by single extension like '.pdf', '.xlsx'. Optional.",
          },
          filter_type: {
            type: "STRING",
            description: "Filter by category: 'image', 'document', 'spreadsheet', 'presentation', 'video', 'audio', 'archive'. Optional.",
          },
          name_contains: {
            type: "STRING",
            description: "Search by filename containing this text (case-insensitive). E.g. 'invoice' finds 'Invoice_2024.pdf'.",
          },
          recursive: {
            type: "BOOLEAN",
            description: "Search subdirectories recursively. Default false. Use when file might be in a subfolder.",
          },
        },
        required: ["folder"],
      },
    },
    {
      name: "grep_files",
      description: "Search INSIDE files for text content. Finds files containing a pattern and shows matching lines. Use when you need to find which file contains specific text (a name, phone number, keyword). Works on any text file — no indexing needed.",
      parameters: {
        type: "OBJECT",
        properties: {
          pattern: {
            type: "STRING",
            description: "Text pattern to search for inside files (case-insensitive).",
          },
          folder: {
            type: "STRING",
            description: "Folder to search in.",
          },
          file_filter: {
            type: "STRING",
            description: "Filter by file pattern, e.g. '*.txt', '*.csv', '*.log'. Optional.",
          },
        },
        required: ["pattern", "folder"],
      },
    },
    {
      name: "file_info",
      description: "Get detailed information about a file or folder: size, creation date, modification date, file type, and whether it's indexed in Pinpoint's database.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "File or folder path.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "move_file",
      description: "Move, copy, or rename a single file. For moving multiple files, use batch_move instead.",
      parameters: {
        type: "OBJECT",
        properties: {
          source: {
            type: "STRING",
            description: "Source file path.",
          },
          destination: {
            type: "STRING",
            description: "Destination path (file path or folder).",
          },
          copy: {
            type: "BOOLEAN",
            description: "If true, copy instead of move. Default: false.",
          },
        },
        required: ["source", "destination"],
      },
    },
    {
      name: "copy_file",
      description: "Copy a file or folder to a new location.",
      parameters: {
        type: "OBJECT",
        properties: {
          source: { type: "STRING", description: "Source file or folder path." },
          destination: { type: "STRING", description: "Destination path." },
        },
        required: ["source", "destination"],
      },
    },
    {
      name: "batch_move",
      description: "Move or copy multiple files to a destination folder in one call. Much faster than calling move_file repeatedly. Creates destination folder if needed.",
      parameters: {
        type: "OBJECT",
        properties: {
          sources: {
            type: "ARRAY",
            items: { type: "STRING" },
            description: "List of source file paths to move/copy.",
          },
          destination: {
            type: "STRING",
            description: "Destination folder path. All files will be moved/copied here.",
          },
          is_copy: {
            type: "BOOLEAN",
            description: "If true, copy files instead of moving. Default: false (move).",
          },
        },
        required: ["sources", "destination"],
      },
    },
    {
      name: "create_folder",
      description: "Create a new folder (directory). Creates parent folders too if they don't exist.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Folder path to create.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "delete_file",
      description: "Delete a file. SAFETY: ALWAYS ask the user for explicit confirmation before deleting. Never delete without user approval. Cannot delete folders.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "File path to delete.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "send_file",
      description: "Send a file to the user on WhatsApp. ONLY use this when the user explicitly asks to receive/send/share a file. Never send files automatically. Max 16MB for images, 100MB for documents. If too large, use resize_image or compress_files first.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Absolute path of the file to send.",
          },
          caption: {
            type: "STRING",
            description: "Short caption for the file. Optional.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "get_status",
      description: "Get indexing statistics: total files indexed, count by file type, database size.",
      parameters: {
        type: "OBJECT",
        properties: {},
      },
    },
    {
      name: "read_file",
      description: "Read a file from disk. For images: you SEE the image visually. If the user sent a photo, it is already visible to you — do NOT call read_file on it again. For documents (PDF, DOCX, TXT): prefer index_file + search_documents for searching. For Excel: use analyze_data instead.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Absolute path to the file to read.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "search_history",
      description: "Search past conversation messages from previous sessions. Use when the user refers to something discussed earlier that's not in the current conversation, like 'that file from yesterday' or 'what did I search for last time?'.",
      parameters: {
        type: "OBJECT",
        properties: {
          query: {
            type: "STRING",
            description: "Keywords to search for in past conversations.",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "detect_faces",
      description: "Detect and analyze faces in an image or all images in a folder. Returns face count, bounding boxes, confidence, age, gender, head pose. Pass folder for batch processing (one call instead of many).",
      parameters: {
        type: "OBJECT",
        properties: {
          image_path: {
            type: "STRING",
            description: "Absolute path to a single image file.",
          },
          folder: {
            type: "STRING",
            description: "Absolute path to folder — processes ALL images in it.",
          },
        },
      },
    },
    {
      name: "crop_face",
      description: "Crop a specific face from an image and save it as a separate file. Use this when detect_faces found multiple faces and you need to show them to the user so they can pick which person to search for. Returns the path to the cropped face image which you can send via send_file.",
      parameters: {
        type: "OBJECT",
        properties: {
          image_path: {
            type: "STRING",
            description: "Absolute path to the original image.",
          },
          face_idx: {
            type: "INTEGER",
            description: "Index of the face to crop (from detect_faces result).",
          },
        },
        required: ["image_path", "face_idx"],
      },
    },
    {
      name: "find_person",
      description: "Find all photos of a specific person in a folder. The reference image should contain exactly ONE face. If the reference has multiple faces, first use detect_faces + crop_face to let the user pick, then use find_person_by_face instead. Scans all images in the folder and returns matching photos sorted by similarity. First scan may take a while (caches results for instant repeat searches).",
      parameters: {
        type: "OBJECT",
        properties: {
          reference_image: {
            type: "STRING",
            description: "Path to the reference image containing the person's face.",
          },
          folder: {
            type: "STRING",
            description: "Absolute path to the folder to search in.",
          },
        },
        required: ["reference_image", "folder"],
      },
    },
    {
      name: "find_person_by_face",
      description: "Find all photos of a specific person using a face index from a multi-face reference image. Use this when the reference image has multiple faces and the user has picked which face to search for (via detect_faces + crop_face). The face_idx comes from detect_faces.",
      parameters: {
        type: "OBJECT",
        properties: {
          reference_image: {
            type: "STRING",
            description: "Path to the reference image.",
          },
          face_idx: {
            type: "INTEGER",
            description: "Index of the chosen face (from detect_faces).",
          },
          folder: {
            type: "STRING",
            description: "Absolute path to the folder to search in.",
          },
        },
        required: ["reference_image", "face_idx", "folder"],
      },
    },
    {
      name: "count_faces",
      description: "Count faces in an image, a list of images, or all images in a folder. Returns face count, age/gender breakdown. Use paths array to batch multiple specific images in ONE call instead of calling per-image.",
      parameters: {
        type: "OBJECT",
        properties: {
          image_path: {
            type: "STRING",
            description: "Absolute path to a single image file.",
          },
          paths: {
            type: "ARRAY",
            items: { type: "STRING" },
            description: "Array of image paths to count faces in batch. Use this instead of calling count_faces per-image.",
          },
          folder: {
            type: "STRING",
            description: "Absolute path to folder — counts faces in ALL images in folder.",
          },
        },
      },
    },
    {
      name: "compare_faces",
      description: "Compare two specific faces from two images to check if they are the same person. Returns similarity score (0-1) and confidence level. Use when user asks 'is this the same person?' or wants to verify identity across photos.",
      parameters: {
        type: "OBJECT",
        properties: {
          image_path_1: {
            type: "STRING",
            description: "Path to the first image.",
          },
          face_idx_1: {
            type: "INTEGER",
            description: "Face index in first image (default 0 for first/only face).",
          },
          image_path_2: {
            type: "STRING",
            description: "Path to the second image.",
          },
          face_idx_2: {
            type: "INTEGER",
            description: "Face index in second image (default 0 for first/only face).",
          },
        },
        required: ["image_path_1", "image_path_2"],
      },
    },
    {
      name: "remember_face",
      description: "Save a face for future recognition. After this, detect_faces will auto-identify this person in any photo. One person can have multiple saved faces (different angles improve accuracy). Use detect_faces first to get face_idx.",
      parameters: {
        type: "OBJECT",
        properties: {
          image_path: {
            type: "STRING",
            description: "Absolute path to the image containing the face.",
          },
          face_idx: {
            type: "INTEGER",
            description: "Index of the face to save (from detect_faces result). Default 0 for single-face images.",
          },
          name: {
            type: "STRING",
            description: "Name to associate with this face (e.g. 'Sharika', 'Dad').",
          },
        },
        required: ["image_path", "name"],
      },
    },
    {
      name: "forget_face",
      description: "Delete all saved face data for a person. After this, they will no longer be auto-recognized by detect_faces.",
      parameters: {
        type: "OBJECT",
        properties: {
          name: {
            type: "STRING",
            description: "Name of the person to forget (case-insensitive).",
          },
        },
        required: ["name"],
      },
    },
    {
      name: "ocr",
      description: "Extract text from an image or scanned PDF using OCR (Tesseract). Use this when you need the text as a string for processing. For just SEEING an image, use read_file instead (sends image visually). Pass folder for batch processing. After OCR, use index_file to make the text searchable.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Absolute path to a single image or PDF file.",
          },
          folder: {
            type: "STRING",
            description: "Absolute path to folder — OCR ALL images and PDFs in it.",
          },
        },
      },
    },
    {
      name: "detect_objects",
      description: "Detect non-human objects in an image and get bounding boxes (x_min, y_min, x_max, y_max). Use for: cars, dogs, plates, signs, logos, etc. Do NOT use for people/faces — use detect_faces + crop_face instead (InsightFace).",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Absolute path to the image file.",
          },
          object: {
            type: "STRING",
            description: "Object to detect (e.g. 'car', 'dog', 'text', 'plate').",
          },
        },
        required: ["path", "object"],
      },
    },
    {
      name: "analyze_data",
      description: "Run pandas data analysis on CSV or Excel files. WORKFLOW: 1) FIRST call with operation='columns' to see all sheets, column names, types, and sample values. 2) Use operation='search' with query to find values across ALL sheets — auto-normalizes phone/ID formats (strips dashes, parens). 3) Use filter/groupby/sort when you know the exact column name. Operations: columns (schema+sheets), search (grep-like across all cells), describe, head, filter, value_counts, groupby, corr, sort, unique, shape, eval. For phone/ID lookups ALWAYS use search — it normalizes automatically. File is cached after first load (instant repeat calls).",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Absolute path to CSV or Excel file.",
          },
          operation: {
            type: "STRING",
            description: "Operation: columns (FIRST — see sheets+types), search (find values across all sheets), describe, head, filter, value_counts, groupby, corr, sort, unique, shape, eval.",
          },
          columns: {
            type: "STRING",
            description: "Column name(s). For groupby use 'group_col:agg_col'. For sort prefix with '-' for descending.",
          },
          query: {
            type: "STRING",
            description: "For search: value to find (e.g. '9208896630' — auto-normalizes phone formats). For filter: pandas query like 'amount > 1000'. For eval: expression like 'df.groupby(\"Cat\")[[\"Price\"]].sum()'.",
          },
          sheet: {
            type: "STRING",
            description: "Sheet name for Excel files. Omit to use first sheet (or search ALL sheets with operation='search').",
          },
        },
        required: ["path", "operation"],
      },
    },
    {
      name: "index_file",
      description: "Index a single file into the search database. Extracts text, chunks it into sections, and makes it searchable. Use this BEFORE search_documents when the file hasn't been indexed yet. After indexing, use search_documents to find specific sections within the file.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: {
            type: "STRING",
            description: "Absolute path to the file to index.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "write_file",
      description: "Create or write a text file. Can append to existing files. Use for creating notes, reports, summaries, text exports.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Absolute path for the file." },
          content: { type: "STRING", description: "Text content to write." },
          append: { type: "BOOLEAN", description: "If true, append to existing file. Default: false." },
        },
        required: ["path", "content"],
      },
    },
    {
      name: "generate_excel",
      description: "Create an Excel file from data. Use for expense reports, data exports, aggregated results.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Output path for .xlsx file." },
          data: { type: "ARRAY", description: "List of row objects (e.g. [{\"name\":\"A\",\"amount\":100}]).", items: { type: "OBJECT" } },
          sheet_name: { type: "STRING", description: "Sheet name. Default: Sheet1." },
        },
        required: ["path", "data"],
      },
    },
    {
      name: "generate_chart",
      description: "Create a chart image (bar, line, pie, scatter, hist) from data using matplotlib. Returns image path — send via send_file.",
      parameters: {
        type: "OBJECT",
        properties: {
          data: { type: "OBJECT", description: "Chart data: {\"labels\": [...], \"values\": [...]}." },
          chart_type: { type: "STRING", description: "Chart type: bar, line, pie, scatter, hist." },
          title: { type: "STRING", description: "Chart title." },
          xlabel: { type: "STRING", description: "X-axis label." },
          ylabel: { type: "STRING", description: "Y-axis label." },
          output_path: { type: "STRING", description: "Output image path. Optional." },
        },
        required: ["data", "chart_type"],
      },
    },
    {
      name: "merge_pdf",
      description: "Combine multiple PDFs into one file. Use for merging invoices, reports, certificates.",
      parameters: {
        type: "OBJECT",
        properties: {
          paths: { type: "ARRAY", description: "List of PDF file paths to merge.", items: { type: "STRING" } },
          output_path: { type: "STRING", description: "Output path for merged PDF." },
        },
        required: ["paths", "output_path"],
      },
    },
    {
      name: "split_pdf",
      description: "Extract specific pages from a PDF. Page format: '1-5', '3,7,10', '1-3,5,8-10'.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Source PDF path." },
          pages: { type: "STRING", description: "Pages to extract: '1-5', '3,7', '1-3,5,8-10'." },
          output_path: { type: "STRING", description: "Output path for extracted PDF." },
        },
        required: ["path", "pages", "output_path"],
      },
    },
    {
      name: "pdf_to_images",
      description: "Render PDF pages as PNG images. Returns list of image paths. Use to send PDF pages via WhatsApp.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "PDF file path." },
          pages: { type: "STRING", description: "Pages to render: '1-5', '3,7'. Omit for all pages." },
          dpi: { type: "INTEGER", description: "Resolution. Default 150." },
          output_folder: { type: "STRING", description: "Folder for output images. Default: same folder as PDF." },
        },
        required: ["path"],
      },
    },
    {
      name: "images_to_pdf",
      description: "Combine multiple images into a single PDF. Supports jpg, png, webp, bmp.",
      parameters: {
        type: "OBJECT",
        properties: {
          paths: { type: "ARRAY", items: { type: "STRING" }, description: "Array of image paths to combine." },
          output_path: { type: "STRING", description: "Output PDF path." },
        },
        required: ["paths", "output_path"],
      },
    },
    {
      name: "resize_image",
      description: "Resize or compress an image. Set width OR height to keep aspect ratio, or both for exact size.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Image path." },
          width: { type: "INTEGER", description: "Target width in pixels." },
          height: { type: "INTEGER", description: "Target height in pixels." },
          quality: { type: "INTEGER", description: "JPEG quality 1-100. Default 85." },
          output_path: { type: "STRING", description: "Output path. If omitted, overwrites original." },
        },
        required: ["path"],
      },
    },
    {
      name: "convert_image",
      description: "Convert image format. Supports: jpg, png, webp, bmp. Handles HEIC (iPhone photos) to JPG.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Source image path." },
          format: { type: "STRING", description: "Target format: jpg, png, webp, bmp." },
          output_path: { type: "STRING", description: "Output path. Optional — defaults to same name with new extension." },
        },
        required: ["path", "format"],
      },
    },
    {
      name: "crop_image",
      description: "Crop an image to specified rectangle. Coordinates in pixels. Use file_info or read_file first to know image dimensions before cropping.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Image path." },
          x: { type: "INTEGER", description: "Left edge (pixels from left)." },
          y: { type: "INTEGER", description: "Top edge (pixels from top)." },
          width: { type: "INTEGER", description: "Crop width in pixels." },
          height: { type: "INTEGER", description: "Crop height in pixels." },
          output_path: { type: "STRING", description: "Output path. Optional." },
        },
        required: ["path", "x", "y", "width", "height"],
      },
    },
    {
      name: "compress_files",
      description: "Zip files or folders into a .zip archive. Can include multiple files and entire folders.",
      parameters: {
        type: "OBJECT",
        properties: {
          paths: { type: "ARRAY", description: "List of file/folder paths to zip.", items: { type: "STRING" } },
          output_path: { type: "STRING", description: "Output .zip file path." },
        },
        required: ["paths", "output_path"],
      },
    },
    {
      name: "extract_archive",
      description: "Extract a zip archive. If no output_path, extracts to folder with same name.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Path to .zip file." },
          output_path: { type: "STRING", description: "Folder to extract into. Optional." },
        },
        required: ["path"],
      },
    },
    {
      name: "download_url",
      description: "Download a file from a URL. Saves to Downloads/Pinpoint/ by default. Use when user shares a link.",
      parameters: {
        type: "OBJECT",
        properties: {
          url: { type: "STRING", description: "URL to download." },
          save_path: { type: "STRING", description: "Where to save. Optional." },
        },
        required: ["url"],
      },
    },
    {
      name: "find_duplicates",
      description: "Find duplicate files in a folder by content hash. Returns groups of identical files for cleanup.",
      parameters: {
        type: "OBJECT",
        properties: {
          folder: { type: "STRING", description: "Folder path to scan." },
        },
        required: ["folder"],
      },
    },
    {
      name: "batch_rename",
      description: "Rename files matching a regex pattern. ALWAYS call with dry_run=true first to preview changes, show the user, and only execute with dry_run=false after confirmation.",
      parameters: {
        type: "OBJECT",
        properties: {
          folder: { type: "STRING", description: "Folder containing files to rename." },
          pattern: { type: "STRING", description: "Regex pattern to match in filenames." },
          replace: { type: "STRING", description: "Replacement string." },
          dry_run: { type: "BOOLEAN", description: "Preview only (default true). Set false to execute after user confirms." },
        },
        required: ["folder", "pattern", "replace"],
      },
    },
    {
      name: "run_python",
      description: "Execute Python code. Use for any custom operation: image manipulation, data processing, file operations, calculations, generating files. Pre-loaded: PIL, pandas, numpy, matplotlib, os, json. Working dir: /tmp/pinpoint_python/. Print results to stdout.",
      parameters: {
        type: "OBJECT",
        properties: {
          code: { type: "STRING", description: "Python code to execute. Use print() for output. Save files to WORK_DIR." },
          timeout: { type: "INTEGER", description: "Max execution time in seconds. Default 30, max 120." },
        },
        required: ["code"],
      },
    },
    {
      name: "memory_save",
      description: "Save a fact to persistent memory. Persists across sessions and restarts. Smart dedup: skips duplicates, merges related facts, handles contradictions. Only works when memory is enabled.",
      parameters: {
        type: "OBJECT",
        properties: {
          fact: { type: "STRING", description: "The fact to remember. Keep it short and factual." },
          category: { type: "STRING", description: "Category: people, places, preferences, professional, health, plans, general. Default: general." },
        },
        required: ["fact"],
      },
    },
    {
      name: "memory_search",
      description: "Search persistent memories by keyword. Use to recall personal facts about the user before saying 'I don't know'. Only works when memory is enabled.",
      parameters: {
        type: "OBJECT",
        properties: {
          query: { type: "STRING", description: "Search keywords. E.g. 'mom', 'trip', 'preference'." },
        },
        required: ["query"],
      },
    },
    {
      name: "memory_delete",
      description: "Delete a memory by ID. Use when user asks to forget something and you have the ID.",
      parameters: {
        type: "OBJECT",
        properties: {
          id: { type: "INTEGER", description: "Memory ID to delete (from memory_search results)." },
        },
        required: ["id"],
      },
    },
    {
      name: "memory_forget",
      description: "Forget a memory by description — no ID needed. Searches memories for best match and deletes it. Use when user says 'forget that I like dark mode' or 'remove the thing about Mumbai'. Preferred over memory_delete when you don't have the ID.",
      parameters: {
        type: "OBJECT",
        properties: {
          description: { type: "STRING", description: "Natural language description of what to forget. E.g. 'dark mode preference', 'living in Mumbai'." },
        },
        required: ["description"],
      },
    },
    {
      name: "search_facts",
      description: "Search extracted facts from indexed documents. Facts are key details (names, dates, amounts, topics) auto-extracted at index time. Use for quick factual lookups like 'who is the electrician?' or 'what was the invoice amount?'. Falls back to search_documents for full-text search if facts don't have enough detail.",
      parameters: {
        type: "OBJECT",
        properties: {
          query: { type: "STRING", description: "Search keywords for facts. E.g. 'electrician', 'invoice amount', 'meeting date'." },
        },
        required: ["query"],
      },
    },
    // web_search (Jina) commented out — replaced by web_read (Segment 18U)
    // {
    //   name: "web_search",
    //   description: "Search the web for real-time information. Use when user asks about current events, weather, sports, news, or anything not in local files. Returns top 5 results with titles, URLs, and content.",
    //   parameters: {
    //     type: "OBJECT",
    //     properties: {
    //       query: { type: "STRING", description: "Search query." },
    //     },
    //     required: ["query"],
    //   },
    // },
    {
      name: "web_search",
      description: "Search the web for real-world information. Use for news, weather, sports, people, products, prices, current events, comparisons — anything NOT in local files. Returns search results with titles, snippets, and URLs. The results are reliable and current — answer directly from them. Do NOT fall back to search_documents or search_facts for web queries. If you need more detail on a specific result, call again with that result's full URL.",
      parameters: {
        type: "OBJECT",
        properties: {
          query: { type: "STRING", description: "Search query. Be specific." },
          url: { type: "STRING", description: "Optional: a specific URL to read instead of searching. Use to read full content of a search result." },
          start: { type: "INTEGER", description: "Character offset for long pages. Use the end value from previous response to continue reading." },
        },
        required: ["query"],
      },
    },
    {
      name: "search_images_visual",
      description: "Search images in a folder by text description using AI vision embeddings (SigLIP2). Returns ranked list of images matching the query. First call embeds the folder (takes time), subsequent queries on same folder are instant (cached). Pass queries as array for batch search (multiple queries in one call). Use for: finding specific photos in a large folder, visual search, filtering images by content.",
      parameters: {
        type: "OBJECT",
        properties: {
          folder: { type: "STRING", description: "Absolute path to folder containing images." },
          query: { type: "STRING", description: "Single text query. E.g. 'bride cutting cake'." },
          queries: {
            type: "ARRAY",
            items: { type: "STRING" },
            description: "Multiple queries for batch search. E.g. ['dancing', 'flowers', 'group photo']. Use instead of query for multiple searches at once.",
          },
          limit: { type: "INTEGER", description: "Max results per query. Default 10." },
        },
        required: ["folder"],
      },
    },
    {
      name: "search_video",
      description: "Search inside a video by text description using SigLIP2. Returns timestamps of matching frames. First call extracts+embeds frames (slow), repeat queries are instant (cached). After finding timestamps, use extract_frame to get the image — it will be auto-sent to the user.",
      parameters: {
        type: "OBJECT",
        properties: {
          video_path: { type: "STRING", description: "Absolute path to video file." },
          query: { type: "STRING", description: "Text description of what to find. E.g. 'person dancing', 'sunset scene'." },
          fps: { type: "NUMBER", description: "Frames per second to extract. Default 1. Use 0.5 for long videos, 2 for short clips." },
          limit: { type: "INTEGER", description: "Max results. Default 5." },
        },
        required: ["video_path", "query"],
      },
    },
    {
      name: "extract_frame",
      description: "Extract a single frame from a video at a specific timestamp. Returns the frame as an image file. Use after search_video to get the actual frame image for sending.",
      parameters: {
        type: "OBJECT",
        properties: {
          video_path: { type: "STRING", description: "Absolute path to video file." },
          seconds: { type: "NUMBER", description: "Timestamp in seconds to extract frame from." },
        },
        required: ["video_path", "seconds"],
      },
    },
    {
      name: "set_reminder",
      description: "Set a reminder that will be sent to the user at a specific time. Supports one-time or recurring. Use when user says 'remind me to X at/by Y time' or 'remind me every Monday'. Persists across restarts.",
      parameters: {
        type: "OBJECT",
        properties: {
          message: { type: "STRING", description: "The reminder message. E.g. 'Buy tablets', 'Call dentist'." },
          time: { type: "STRING", description: "When to remind. ISO format preferred: '2026-02-27T17:00:00'. Also accepts: '17:00' (today), '5pm' (today), 'in 2 hours', 'tomorrow 9am'." },
          repeat: { type: "STRING", description: "Optional. Repeat schedule: 'daily', 'weekly', 'monthly', 'weekdays'. Omit for one-time reminder." },
        },
        required: ["message", "time"],
      },
    },
    {
      name: "list_reminders",
      description: "List all pending reminders. Use when user asks 'what reminders do I have?' or 'show my reminders'.",
      parameters: {
        type: "OBJECT",
        properties: {},
      },
    },
    {
      name: "cancel_reminder",
      description: "Cancel a pending reminder by its ID. For recurring reminders, this stops all future occurrences.",
      parameters: {
        type: "OBJECT",
        properties: {
          id: { type: "INTEGER", description: "Reminder ID (from list_reminders)." },
        },
        required: ["id"],
      },
    },
    {
      name: "extract_tables",
      description: "Extract structured tables from a PDF. Returns headers + rows for each table found. Works on native PDFs (not scanned). Use for invoices, reports, financial statements, any PDF with tabular data.",
      parameters: {
        type: "OBJECT",
        properties: {
          path: { type: "STRING", description: "Absolute path to PDF file." },
          pages: { type: "STRING", description: "Page range: '1-5', '3', or 'all'. Default: all." },
        },
        required: ["path"],
      },
    },
    {
      name: "watch_folder",
      description: "Start auto-indexing a folder. New or modified files will be automatically indexed for search. Persists across restarts.",
      parameters: {
        type: "OBJECT",
        properties: {
          folder: { type: "STRING", description: "Absolute path to folder to watch." },
        },
        required: ["folder"],
      },
    },
    {
      name: "unwatch_folder",
      description: "Stop auto-indexing a folder.",
      parameters: {
        type: "OBJECT",
        properties: {
          folder: { type: "STRING", description: "Absolute path to folder to stop watching." },
        },
        required: ["folder"],
      },
    },
    {
      name: "list_watched",
      description: "List all folders currently being watched for auto-indexing.",
      parameters: {
        type: "OBJECT",
        properties: {},
      },
    },
  ],
}];

// --- Reminders system (persistent via API) ---
const reminders = []; // { id, chatJid, message, triggerAt (ms), repeat?, createdAt }

function parseReminderTime(timeStr) {
  const now = new Date();
  const lower = timeStr.toLowerCase().trim();

  // "in X hours/minutes"
  const inMatch = lower.match(/^in\s+(\d+)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)$/);
  if (inMatch) {
    const n = parseInt(inMatch[1]);
    const unit = inMatch[2];
    if (unit.startsWith("h")) return new Date(now.getTime() + n * 3600000);
    if (unit.startsWith("m")) return new Date(now.getTime() + n * 60000);
    if (unit.startsWith("s")) return new Date(now.getTime() + n * 1000);
  }

  // ISO format or full date string
  const parsed = new Date(timeStr);
  if (!isNaN(parsed.getTime())) return parsed;

  // "5pm", "17:00", "5:30pm" — treat as today
  const timeMatch = lower.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/);
  if (timeMatch) {
    let h = parseInt(timeMatch[1]);
    const m = parseInt(timeMatch[2] || "0");
    const ampm = timeMatch[3];
    if (ampm === "pm" && h < 12) h += 12;
    if (ampm === "am" && h === 12) h = 0;
    const target = new Date(now);
    target.setHours(h, m, 0, 0);
    if (target <= now) target.setDate(target.getDate() + 1);
    return target;
  }

  // "tomorrow 9am"
  const tmrMatch = lower.match(/^tomorrow\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/);
  if (tmrMatch) {
    let h = parseInt(tmrMatch[1]);
    const m = parseInt(tmrMatch[2] || "0");
    const ampm = tmrMatch[3];
    if (ampm === "pm" && h < 12) h += 12;
    if (ampm === "am" && h === 12) h = 0;
    const target = new Date(now);
    target.setDate(target.getDate() + 1);
    target.setHours(h, m, 0, 0);
    return target;
  }

  return null;
}

function getNextOccurrence(triggerAt, repeat) {
  const d = new Date(triggerAt);
  const now = Date.now();
  switch (repeat) {
    case "daily":
      while (d.getTime() <= now) d.setDate(d.getDate() + 1);
      break;
    case "weekly":
      while (d.getTime() <= now) d.setDate(d.getDate() + 7);
      break;
    case "monthly":
      while (d.getTime() <= now) d.setMonth(d.getMonth() + 1);
      break;
    case "weekdays":
      do { d.setDate(d.getDate() + 1); } while (d.getDay() === 0 || d.getDay() === 6 || d.getTime() <= now);
      break;
    default:
      return null;
  }
  return d;
}

async function loadReminders() {
  try {
    const res = await apiGet("/reminders");
    reminders.length = 0; // clear in-memory
    for (const r of (res.reminders || [])) {
      reminders.push({
        id: r.id, chatJid: r.chat_jid, message: r.message,
        triggerAt: new Date(r.trigger_at).getTime(),
        repeat: r.repeat || null, createdAt: new Date(r.created_at).getTime(),
      });
    }
    if (reminders.length > 0) console.log(`[Reminder] Loaded ${reminders.length} reminders from DB`);
  } catch (e) {
    console.error(`[Reminder] Failed to load: ${e.message}`);
  }
}

// --- Reconnect policy (matches OpenClaw) ---
const RECONNECT = { initialMs: 2000, maxMs: 30000, factor: 1.8, jitter: 0.25, maxAttempts: 12 };
let reconnectAttempt = 0;

// --- TempStore: large tool results stored server-side, Gemini gets ref ID ---
const tempStore = new Map();
let refCounter = 0;
const REF_EXPIRE_MS = 30 * 60 * 1000; // 30 min safety net
const REF_THRESHOLD = 2000; // Only store if JSON > 2000 chars
const HISTORY_CAP = 500; // Cap old tool results in conversation history

// Tools whose results should be stored as refs when large
const REF_TOOLS = new Set([
  "list_files", "search_images_visual", "find_person", "find_person_by_face",
  "detect_faces", "count_faces", "ocr", "find_duplicates",
  "pdf_to_images", "search_documents",
]);

// Preview limits per tool (how many items to show Gemini)
const REF_PREVIEW = {
  list_files: 20, search_images_visual: 10, find_person: 5, find_person_by_face: 5,
  detect_faces: 10, count_faces: 10, ocr: 3, find_duplicates: 5,
  pdf_to_images: 10, search_documents: 5,
};

function storeRef(data) {
  const id = ++refCounter;
  const key = `@ref:${id}`;
  tempStore.set(key, { data, createdAt: Date.now() });
  return key;
}

function resolveRef(key) {
  const entry = tempStore.get(key);
  if (!entry) return null;
  tempStore.delete(key); // Delete after consumption
  return entry.data;
}

// Clean expired refs (called periodically)
function cleanExpiredRefs() {
  const now = Date.now();
  for (const [key, entry] of tempStore) {
    if (now - entry.createdAt > REF_EXPIRE_MS) tempStore.delete(key);
  }
}
setInterval(cleanExpiredRefs, 5 * 60 * 1000); // Check every 5 min

// Create a preview summary for Gemini (array of items → first N + count + ref)
function makeRefPreview(toolName, result, refKey) {
  const limit = REF_PREVIEW[toolName] || 10;

  // Handle array results (list_files, find_person, pdf_to_images, etc.)
  if (Array.isArray(result)) {
    const total = result.length;
    const preview = result.slice(0, limit);
    return { _ref: refKey, total, showing: preview.length, preview, note: `${total} items stored. Use ${refKey} in subsequent tool calls to reference all items.` };
  }

  // Handle object with results/matches/files array
  for (const arrayKey of ["entries", "results", "matches", "files", "images", "groups", "pages", "faces"]) {
    if (result[arrayKey] && Array.isArray(result[arrayKey]) && result[arrayKey].length > 0) {
      const arr = result[arrayKey];
      const total = arr.length;
      const preview = arr.slice(0, limit);
      const rest = { ...result, [arrayKey]: preview };
      return { _ref: refKey, total_items: total, showing: preview.length, ...rest, note: `${total} ${arrayKey} stored. Use ${refKey} to reference all.` };
    }
  }

  // Fallback: just mark it as ref'd
  return { _ref: refKey, note: `Large result stored. Use ${refKey} to reference it.`, summary: JSON.stringify(result).slice(0, 300) + "..." };
}

// Resolve @ref:N in tool args before execution
function resolveRefsInArgs(args) {
  if (!args) return args;
  const resolved = { ...args };
  for (const [key, value] of Object.entries(resolved)) {
    if (typeof value === "string" && value.startsWith("@ref:")) {
      const data = resolveRef(value);
      if (data) {
        // If the stored data is an array, use it directly
        if (Array.isArray(data)) {
          resolved[key] = data;
        } else {
          // If it's an object with a results/matches/files array, extract the array
          for (const arrayKey of ["entries", "results", "matches", "files", "images", "groups", "pages"]) {
            if (data[arrayKey] && Array.isArray(data[arrayKey])) {
              // For batch_move sources, extract just the paths
              if (key === "sources" || key === "paths") {
                resolved[key] = data[arrayKey].map(item => item.path || item.file || item);
              } else {
                resolved[key] = data[arrayKey];
              }
              break;
            }
          }
          // If no array found, use the whole object
          if (typeof resolved[key] === "string") resolved[key] = data;
        }
        console.log(`[TempStore] Resolved ${value} → ${Array.isArray(resolved[key]) ? resolved[key].length + " items" : "object"}`);
      } else {
        console.warn(`[TempStore] ${value} not found (expired or already consumed)`);
      }
    }
  }
  return resolved;
}

// --- Echo prevention (3 layers) ---
const processedIds = new Set();
const MAX_PROCESSED_IDS = 5000;
const sentTexts = new Set();
const MAX_SENT_TEXTS = 100;

function hashText(text) {
  return createHash("sha256").update(text).digest("hex").slice(0, 16);
}
function rememberSent(text) {
  const h = hashText(text);
  sentTexts.add(h);
  if (sentTexts.size > MAX_SENT_TEXTS) sentTexts.delete(sentTexts.values().next().value);
}
function wasSentByUs(text) {
  return sentTexts.has(hashText(text));
}
function markProcessed(id) {
  processedIds.add(id);
  if (processedIds.size > MAX_PROCESSED_IDS) processedIds.delete(processedIds.values().next().value);
}

// --- Markdown → WhatsApp formatting ---
// Gemini uses markdown (**bold**, ```code```). WhatsApp uses *bold*, ```code```.
function markdownToWhatsApp(text) {
  // Protect code blocks first (don't modify inside them)
  const codeBlocks = [];
  text = text.replace(/```[\s\S]*?```/g, (m) => {
    codeBlocks.push(m);
    return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
  });
  // Protect inline code
  const inlineCode = [];
  text = text.replace(/`[^`]+`/g, (m) => {
    inlineCode.push(m);
    return `__INLINE_CODE_${inlineCode.length - 1}__`;
  });
  // **bold** → *bold* (WhatsApp bold)
  text = text.replace(/\*\*(.+?)\*\*/g, "*$1*");
  // ### Header → *Header* (WhatsApp has no headers, bold them)
  text = text.replace(/^#{1,6}\s+(.+)$/gm, "*$1*");
  // [text](url) → text (url) — WhatsApp doesn't support links
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)");
  // Restore code blocks and inline code
  text = text.replace(/__INLINE_CODE_(\d+)__/g, (_, i) => inlineCode[i]);
  text = text.replace(/__CODE_BLOCK_(\d+)__/g, (_, i) => codeBlocks[i]);
  return text;
}

// --- Message debouncer (combine rapid messages per chat) ---
const pendingMessages = new Map(); // chatJid → { texts: [], timer, resolvers[] }

function debounceMessage(chatJid, text, sock) {
  return new Promise((resolve) => {
    const existing = pendingMessages.get(chatJid);
    if (existing) {
      existing.texts.push(text);
      existing.resolvers.push(resolve);
      clearTimeout(existing.timer);
    } else {
      pendingMessages.set(chatJid, { texts: [text], sock, resolvers: [resolve] });
    }
    const entry = pendingMessages.get(chatJid);
    entry.timer = setTimeout(() => {
      const combined = entry.texts.join("\n");
      pendingMessages.delete(chatJid);
      // First resolver gets the combined text (processes it), rest get null (skip)
      entry.resolvers[0](combined);
      for (let i = 1; i < entry.resolvers.length; i++) entry.resolvers[i](null);
    }, DEBOUNCE_MS);
  });
}

// --- API helpers ---

async function apiGet(path) {
  const resp = await fetch(`${API_URL}${path}`);
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiPost(path, body) {
  const resp = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiDelete(path) {
  const resp = await fetch(`${API_URL}${path}`, { method: "DELETE" });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiPut(path, body) {
  const resp = await fetch(`${API_URL}${path}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiPing() {
  try { return (await fetch(`${API_URL}/ping`)).ok; } catch { return false; }
}

// --- Execute a tool call from Gemini ---
// Pre-validation: catch bad args before hitting the API (saves a round-trip)
function preValidate(name, args) {

  // Tools that need a valid file path
  const fileTools = ["read_file", "read_excel", "move_file", "copy_file", "delete_file",
    "ocr", "caption_image", "detect_faces", "crop_face", "find_person", "find_person_by_face",
    "resize_image", "convert_image", "crop_image", "merge_pdf", "split_pdf",
    "pdf_to_images", "index_file", "compare_faces", "remember_face",
    "query_image", "detect_objects", "point_object"];
  const pathKey = name === "move_file" || name === "copy_file" ? "source"
    : name === "find_person" || name === "find_person_by_face" ? "reference_image"
    : name === "compare_faces" ? "image_path_1"
    : name === "crop_face" || name === "remember_face" ? "image_path"
    : "path";

  if (fileTools.includes(name) && args[pathKey]) {
    try {
      if (!existsSync(args[pathKey])) return `File not found: ${args[pathKey]}. Check the path and try again.`;
    } catch (_) {}
  }

  // Tools that need a valid folder
  const folderTools = ["list_files", "grep_files", "search_images_visual", "find_person",
    "find_person_by_face", "create_folder"];
  const folderKey = "folder";
  if (folderTools.includes(name) && args[folderKey] && name !== "create_folder") {
    try {
      if (!existsSync(args[folderKey])) return `Folder not found: ${args[folderKey]}. Check the path and try again.`;
      if (!statSync(args[folderKey]).isDirectory()) return `Not a folder: ${args[folderKey]}`;
    } catch (_) {}
  }

  // Video tools need valid video path
  if ((name === "search_video" || name === "extract_frame") && args.video_path) {
    try {
      if (!existsSync(args.video_path)) return `Video not found: ${args.video_path}. Check the path and try again.`;
    } catch (_) {}
  }

  // Empty query check
  if (name === "search_documents" && (!args.query || !args.query.trim())) {
    return "Search query cannot be empty.";
  }

  return null; // All good
}

async function executeTool(functionCall, sock, chatJid) {
  const { name } = functionCall;
  const args = resolveRefsInArgs(functionCall.args);
  console.log(`[${LLM_TAG}] Tool call: ${name}(${JSON.stringify(args).slice(0, 200)})`);

  // Pre-validate args before hitting the API
  const validationError = preValidate(name, args);
  if (validationError) {
    console.log(`[Validate] ${name}: ${validationError}`);
    return { error: validationError };
  }

  try {
    switch (name) {
      case "search_documents": {
        let url = `/search?q=${encodeURIComponent(args.query || "")}&limit=${MAX_RESULTS}`;
        if (args.file_type) url += `&file_type=${encodeURIComponent(args.file_type)}`;
        if (args.folder) url += `&folder=${encodeURIComponent(args.folder)}`;
        return await apiGet(url);
      }
      case "read_document":
        return await apiGet(`/document/${args.document_id}`);
      case "read_excel":
        return await apiPost("/read_excel", {
          path: args.path,
          sheet_name: args.sheet_name || null,
          cell_range: args.cell_range || null,
        });
      case "calculate":
        return await apiPost("/calculate", { expression: args.expression });
      case "list_files": {
        let url = `/list_files?folder=${encodeURIComponent(args.folder)}`;
        if (args.sort_by) url += `&sort_by=${encodeURIComponent(args.sort_by)}`;
        if (args.filter_ext) url += `&filter_ext=${encodeURIComponent(args.filter_ext)}`;
        if (args.filter_type) url += `&filter_type=${encodeURIComponent(args.filter_type)}`;
        if (args.name_contains) url += `&name_contains=${encodeURIComponent(args.name_contains)}`;
        if (args.recursive) url += `&recursive=true`;
        return await apiGet(url);
      }
      case "grep_files":
        return await apiPost("/grep", {
          pattern: args.pattern,
          folder: args.folder,
          file_filter: args.file_filter,
        });
      case "file_info":
        return await apiGet(`/file_info?path=${encodeURIComponent(args.path)}`);
      case "batch_move":
        return await apiPost("/batch_move", {
          sources: args.sources || [],
          destination: args.destination,
          is_copy: args.is_copy || false,
        });
      case "move_file":
        return await apiPost("/move_file", {
          source: args.source,
          destination: args.destination,
          is_copy: args.copy || false,
        });
      case "copy_file":
        return await apiPost("/move_file", {
          source: args.source,
          destination: args.destination,
          is_copy: true,
        });
      case "create_folder":
        return await apiPost("/create_folder", { path: args.path });
      case "delete_file":
        return await apiPost("/delete_file", { path: args.path });
      case "send_file": {
        const filePath = args.path;
        const caption = args.caption || `${PREFIX} ${pathModule.basename(filePath)}`;
        const sent = await sendFile(sock, chatJid, filePath, `${PREFIX} ${caption}`);
        if (sent) {
          console.log(`[Pinpoint] Sent: ${pathModule.basename(filePath)}`);
          return { success: true, file: pathModule.basename(filePath) };
        }
        return { error: "File not found or too large to send" };
      }
      case "read_file":
        return await apiPost("/read_file", { path: args.path });
      case "get_status":
        return await apiGet("/status");
      case "search_history":
        return await apiGet(`/conversation/search?q=${encodeURIComponent(args.query || "")}&limit=10`);
      case "detect_faces":
        return await apiPost("/detect-faces", { image_path: args.image_path, folder: args.folder });
      case "crop_face":
        return await apiPost("/crop-face", { image_path: args.image_path, face_idx: args.face_idx });
      case "find_person":
        return await apiPost("/find-person", { reference_image: args.reference_image, folder: args.folder });
      case "find_person_by_face":
        return await apiPost("/find-person-by-face", {
          reference_image: args.reference_image,
          face_idx: args.face_idx,
          folder: args.folder,
        });
      case "count_faces":
        return await apiPost("/count-faces", { image_path: args.image_path, paths: args.paths, folder: args.folder });
      case "compare_faces":
        return await apiPost("/compare-faces", {
          image_path_1: args.image_path_1,
          face_idx_1: args.face_idx_1 || 0,
          image_path_2: args.image_path_2,
          face_idx_2: args.face_idx_2 || 0,
        });
      case "remember_face":
        return await apiPost("/remember-face", {
          image_path: args.image_path,
          face_idx: args.face_idx || 0,
          name: args.name,
        });
      case "forget_face":
        return await apiPost("/forget-face", { name: args.name });
      case "ocr":
        return await apiPost("/ocr", { path: args.path, folder: args.folder });
      case "detect_objects":
        return await apiPost("/detect-objects", { path: args.path, object: args.object });
      case "analyze_data":
        return await apiPost("/analyze-data", {
          path: args.path,
          operation: args.operation || "describe",
          columns: args.columns || null,
          query: args.query || null,
          sheet: args.sheet || null,
        });
      case "index_file":
        return await apiPost("/index-file", { path: args.path });
      case "write_file":
        return await apiPost("/write-file", { path: args.path, content: args.content, append: args.append || false });
      case "generate_excel":
        return await apiPost("/generate-excel", { path: args.path, data: args.data, sheet_name: args.sheet_name || "Sheet1" });
      case "generate_chart":
        return await apiPost("/generate-chart", {
          data: args.data, chart_type: args.chart_type, title: args.title || "",
          xlabel: args.xlabel || "", ylabel: args.ylabel || "", output_path: args.output_path || null,
        });
      case "merge_pdf":
        return await apiPost("/merge-pdf", { paths: args.paths, output_path: args.output_path });
      case "split_pdf":
        return await apiPost("/split-pdf", { path: args.path, pages: args.pages, output_path: args.output_path });
      case "pdf_to_images":
        return await apiPost("/pdf-to-images", { path: args.path, pages: args.pages || null, dpi: args.dpi || 150, output_folder: args.output_folder || null });
      case "images_to_pdf":
        return await apiPost("/images-to-pdf", { paths: args.paths, output_path: args.output_path });
      case "resize_image":
        return await apiPost("/resize-image", {
          path: args.path, width: args.width || null, height: args.height || null,
          quality: args.quality || 85, output_path: args.output_path || null,
        });
      case "convert_image":
        return await apiPost("/convert-image", { path: args.path, format: args.format, output_path: args.output_path || null });
      case "crop_image":
        return await apiPost("/crop-image", { path: args.path, x: args.x, y: args.y, width: args.width, height: args.height, output_path: args.output_path || null });
      case "compress_files":
        return await apiPost("/compress-files", { paths: args.paths, output_path: args.output_path });
      case "extract_archive":
        return await apiPost("/extract-archive", { path: args.path, output_path: args.output_path || null });
      case "download_url":
        return await apiPost("/download-url", { url: args.url, save_path: args.save_path || null });
      case "find_duplicates":
        return await apiPost("/find-duplicates", { folder: args.folder });
      case "batch_rename":
        return await apiPost("/batch-rename", { folder: args.folder, pattern: args.pattern, replace: args.replace, dry_run: args.dry_run !== false });
      case "run_python":
        return await apiPost("/run-python", { code: args.code, timeout: args.timeout || 30 });
      case "search_images_visual": {
        const queries = args.queries || (args.query ? [args.query] : []);
        if (queries.length <= 1) {
          return await apiPost("/search-images-visual", { folder: args.folder, query: queries[0] || "", limit: args.limit || 10 });
        }
        // Batch: first call caches embeddings, then run remaining in parallel
        const firstResult = await apiPost("/search-images-visual", { folder: args.folder, query: queries[0], limit: args.limit || 10 });
        const restPromises = queries.slice(1).map(q =>
          apiPost("/search-images-visual", { folder: args.folder, query: q, limit: args.limit || 10 })
        );
        const restResults = await Promise.all(restPromises);
        const results = { [queries[0]]: firstResult };
        queries.slice(1).forEach((q, i) => { results[q] = restResults[i]; });
        return results;
      }
      case "search_video":
        return await apiPost("/search-video", { video_path: args.video_path, query: args.query, fps: args.fps || 1.0, limit: args.limit || 5 });
      case "extract_frame":
        return await apiPost("/extract-frame", { video_path: args.video_path, seconds: args.seconds });
      case "web_search": {
        // If url provided, read that URL directly; otherwise search Brave
        const q = encodeURIComponent(args.query);
        const webUrl = args.url || `https://search.brave.com/search?q=${q}`;
        const result = await apiGet(`/web-read?url=${encodeURIComponent(webUrl)}${args.start ? `&start=${args.start}` : ""}`);
        return result;
      }
      case "memory_save": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        const res = await apiPost("/memory", { fact: args.fact, category: args.category || "general" });
        // Refresh memory context after saving
        try { const ctx = await apiGet("/memory/context"); memoryContext = ctx.text || ""; } catch (_) {}
        return res;
      }
      case "memory_search": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        return await apiGet(`/memory/search?q=${encodeURIComponent(args.query)}&limit=10`);
      }
      case "memory_delete": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        const res = await apiDelete(`/memory/${args.id}`);
        // Refresh memory context after deleting
        try { const ctx = await apiGet("/memory/context"); memoryContext = ctx.text || ""; } catch (_) {}
        return res;
      }
      case "memory_forget": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        const res = await apiPost("/memory/forget", { description: args.description });
        // Refresh memory context after forgetting
        if (res.success) {
          try { const ctx = await apiGet("/memory/context"); memoryContext = ctx.text || ""; } catch (_) {}
        }
        return res;
      }
      case "search_facts":
        return await apiGet(`/search-facts?q=${encodeURIComponent(args.query)}&limit=10`);
      case "set_reminder": {
        const triggerAt = parseReminderTime(args.time);
        if (!triggerAt) return { error: `Could not parse time: "${args.time}". Use format like "5pm", "in 2 hours", or "2026-02-27T17:00:00".` };
        if (triggerAt.getTime() <= Date.now()) return { error: "That time is in the past." };
        const repeat = args.repeat || null;
        if (repeat && !["daily", "weekly", "monthly", "weekdays"].includes(repeat)) {
          return { error: `Invalid repeat: "${repeat}". Use: daily, weekly, monthly, or weekdays.` };
        }
        const tz = USER_TZ;
        // Persist to DB
        const saved = await apiPost("/reminders", {
          chat_jid: chatJid, message: args.message,
          trigger_at: triggerAt.toISOString(), repeat,
        });
        const id = saved.id;
        reminders.push({ id, chatJid, message: args.message, triggerAt: triggerAt.getTime(), repeat, createdAt: Date.now() });
        const result = { success: true, id, message: args.message, trigger_at: triggerAt.toLocaleString("en-IN", { timeZone: tz, hour: "2-digit", minute: "2-digit", hour12: true, day: "numeric", month: "short" }) };
        if (repeat) result.repeat = repeat;
        return result;
      }
      case "list_reminders": {
        const tz = USER_TZ;
        const pending = reminders.filter(r => r.triggerAt > Date.now());
        if (pending.length === 0) return { count: 0, reminders: [], note: "No pending reminders." };
        return {
          count: pending.length,
          reminders: pending.map(r => ({
            id: r.id, message: r.message,
            repeat: r.repeat || null,
            trigger_at: new Date(r.triggerAt).toLocaleString("en-IN", { timeZone: tz, hour: "2-digit", minute: "2-digit", hour12: true, day: "numeric", month: "short" }),
          })),
        };
      }
      case "cancel_reminder": {
        const idx = reminders.findIndex(r => r.id === args.id);
        if (idx === -1) return { error: `Reminder #${args.id} not found.` };
        const removed = reminders.splice(idx, 1)[0];
        // Delete from DB
        try { await apiDelete(`/reminders/${removed.id}`); } catch (_) {}
        return { success: true, cancelled: removed.message };
      }
      case "extract_tables": {
        const params = new URLSearchParams({ path: args.path });
        if (args.pages) params.set("pages", args.pages);
        return await apiPost(`/extract-tables?${params.toString()}`, {});
      }
      case "watch_folder": {
        return await apiPost(`/watch?folder=${encodeURIComponent(args.folder)}`, {});
      }
      case "unwatch_folder": {
        return await apiPost(`/unwatch?folder=${encodeURIComponent(args.folder)}`, {});
      }
      case "list_watched": {
        return await apiGet("/watched");
      }
      default:
        return { error: `Unknown tool: ${name}` };
    }
  } catch (err) {
    return { error: err.message };
  }
}

// --- Conversation memory helpers ---

async function loadHistory(sessionId) {
  try {
    const data = await apiGet(`/conversation/history?session_id=${encodeURIComponent(sessionId)}&limit=${MAX_HISTORY_MESSAGES}`);
    return data;
  } catch {
    return { messages: [], updated_at: null, message_count: 0 };
  }
}

async function saveMessage(sessionId, role, content) {
  try {
    await apiPost("/conversation", { session_id: sessionId, role, content });
  } catch (err) {
    console.error("[Memory] Save failed:", err.message);
  }
}

const TEMP_MEDIA_DIR = pathModule.join(require("os").tmpdir(), "pinpoint_tmp");

function cleanupTempMedia() {
  try {
    if (!existsSync(TEMP_MEDIA_DIR)) return;
    const files = readdirSync(TEMP_MEDIA_DIR);
    for (const f of files) {
      try { unlinkSync(pathModule.join(TEMP_MEDIA_DIR, f)); } catch (_) {}
    }
    if (files.length > 0) console.log(`[Pinpoint] Cleaned up ${files.length} temp media file(s)`);
  } catch (_) {}
}

async function resetSession(sessionId) {
  try {
    const data = await apiPost("/conversation/reset", { session_id: sessionId });
    cleanupTempMedia();
    lastImage.delete(sessionId);
    return data.deleted_count || 0;
  } catch {
    return 0;
  }
}

function isSessionIdle(updatedAt) {
  if (!updatedAt) return false;
  const lastActivity = new Date(updatedAt + "Z").getTime(); // ISO 8601 UTC
  return (Date.now() - lastActivity) > IDLE_TIMEOUT_MS;
}

// --- Context compaction: summarize old messages instead of dropping them ---
const COMPACT_THRESHOLD = 30; // Compact when total messages exceed this
const COMPACT_KEEP = 10; // Keep this many recent messages after compacting
const CONTENTS_COMPACT_THRESHOLD = 20; // Compact in-memory contents when entries exceed this (≈10 tool rounds)

async function compactHistory(sessionId) {
  try {
    // Load ALL messages (up to 100)
    const data = await apiGet(`/conversation/history?session_id=${encodeURIComponent(sessionId)}&limit=100`);
    const msgs = data.messages || [];
    if (msgs.length < COMPACT_THRESHOLD) return; // Not enough to compact

    // Split: old messages to summarize, recent to keep
    const toSummarize = msgs.slice(0, msgs.length - COMPACT_KEEP);
    const toKeep = msgs.slice(msgs.length - COMPACT_KEEP);

    // Build a structured summary via Gemini (inspired by Gemini CLI)
    let summaryInput = `Summarize this conversation into a structured snapshot. Be concise.

Format:
GOAL: What the user is trying to do (1 line)
KNOWLEDGE: Key facts discovered (bullet points)
FILES: Files created, modified, or moved (bullet points, skip if none)
STATE: What's done and what's remaining (1-2 lines)

Conversation:
`;
    for (const m of toSummarize) {
      summaryInput += `${m.role}: ${m.content.slice(0, 300)}\n`;
    }

    const response = await llmGenerate({
      model: GEMINI_MODEL,
      contents: [{ role: "user", parts: [{ text: summaryInput }] }],
    });
    trackTokens(sessionId, response);
    const summary = response.text || "Previous conversation context unavailable.";

    // Reset session and re-save: summary + kept messages
    await resetSession(sessionId);
    await saveMessage(sessionId, "user", `[Previous conversation context]\n${summary}`);
    for (const m of toKeep) {
      await saveMessage(sessionId, m.role, m.content);
    }
    console.log(`[Memory] Compacted ${toSummarize.length} msgs → summary + ${toKeep.length} recent`);
  } catch (err) {
    console.error("[Memory] Compaction failed:", err.message);
  }
}

// --- Token-based in-memory compaction ---
// When prompt tokens exceed threshold mid-conversation, summarize old turns in the contents array
// This prevents token burn from tool-heavy conversations that hit 300K+ tokens in just a few messages
async function compactContents(contents, chatJid) {
  // Keep last 6 entries (3 user-model pairs) + current user message
  const keepCount = Math.min(7, Math.ceil(contents.length / 2));
  if (contents.length <= keepCount + 2) return false; // Not enough to compact

  const toSummarize = contents.slice(0, contents.length - keepCount);
  const toKeep = contents.slice(contents.length - keepCount);

  // Build a text summary of old turns (strip tool results to save tokens on the summary call)
  let summaryParts = [];
  for (const entry of toSummarize) {
    if (!entry?.parts) continue;
    for (const part of entry.parts) {
      if (part.text) {
        summaryParts.push(`${entry.role}: ${part.text.slice(0, 300)}`);
      } else if (part.functionCall) {
        summaryParts.push(`tool: ${part.functionCall.name}(${JSON.stringify(part.functionCall.args || {}).slice(0, 100)})`);
      } else if (part.functionResponse) {
        const res = JSON.stringify(part.functionResponse.response?.result || "").slice(0, 200);
        summaryParts.push(`result: ${part.functionResponse.name} → ${res}`);
      }
    }
  }

  const summaryInput = `Summarize this conversation into a structured snapshot. Be concise (under 500 chars).

Format:
GOAL: What the user is trying to do (1 line)
KNOWLEDGE: Key facts, file paths, results discovered (bullet points)
STATE: What's done and what's pending (1-2 lines)

Conversation:
${summaryParts.join("\n")}`;

  try {
    const response = await llmGenerate({
      model: GEMINI_MODEL,
      contents: [{ role: "user", parts: [{ text: summaryInput }] }],
    });
    trackTokens(chatJid, response);
    const summary = response.text || "Previous context unavailable.";

    // Replace contents array in-place: summary + kept entries
    contents.length = 0;
    contents.push({ role: "user", parts: [{ text: `[Previous conversation context]\n${summary}` }] });
    contents.push({ role: "model", parts: [{ text: "Understood. I have the context. Continuing." }] });
    for (const entry of toKeep) contents.push(entry);

    console.log(`[Memory] Token compaction: ${toSummarize.length} entries → summary + ${toKeep.length} recent`);
    // Also compact DB history in background
    compactHistory(chatJid).catch(() => {});
    return true;
  } catch (err) {
    console.error("[Memory] Token compaction failed:", err.message);
    return false;
  }
}

// --- Run Gemini agent loop (may do multiple tool calls) ---
async function runGemini(userMessage, sock, chatJid, opts = {}) {
  // opts.inlineImage: { mimeType, data (base64) } — image already visible to Gemini
  // Refresh memory context if enabled — pass query for relevant retrieval (Supermemory pattern)
  if (memoryEnabled) {
    try {
      const qParam = encodeURIComponent(userMessage.slice(0, 200));
      const ctx = await apiGet(`/memory/context?q=${qParam}`);
      memoryContext = ctx.text || "";
    } catch (_) {}
  }

  // Load conversation history for context
  const history = await loadHistory(chatJid);

  // Auto-reset if idle for 60+ minutes
  if (history.updated_at && isSessionIdle(history.updated_at)) {
    const deleted = await resetSession(chatJid);
    delete sessionCosts[chatJid];
    if (deleted > 0) console.log(`[Memory] Auto-reset session (idle ${IDLE_TIMEOUT_MS / 60000} min), cleared ${deleted} messages`);
    history.messages = [];
  }

  // Compact if history is getting long (runs in background, doesn't block this request)
  if (history.message_count > COMPACT_THRESHOLD) {
    compactHistory(chatJid).catch(() => {});
  }

  // Build contents: history + current message
  const contents = [];
  for (const msg of history.messages) {
    contents.push({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }],
    });
  }
  // Current message — include inline image if provided (Gemini sees it immediately)
  const userParts = [{ text: userMessage }];
  if (opts.inlineImage) {
    userParts.push({ inlineData: { mimeType: opts.inlineImage.mimeType, data: opts.inlineImage.data } });
  }
  contents.push({ role: "user", parts: userParts });

  const config = {
    systemInstruction: getSystemPrompt(userMessage),
  };
  let activeTools = null;
  if (!opts.noTools) activeTools = getToolsForIntent(userMessage);

  const toolCache = new Map(); // Dedup: cache tool results within this turn
  const toolLog = []; // Track tool calls for conversation context
  let inlineImageCount = 0; // Track images sent as visual data
  let notifiedUser = false; // Track if we sent "working on it" message
  const toolStartTime = Date.now(); // For elapsed time logging

  // Loop detection: track consecutive identical tool calls (from Gemini CLI + OpenCode)
  const LOOP_THRESHOLD = 3; // Same exact call N times → stop (OpenCode uses 3)
  const MAX_ROUNDS = 25; // Max rounds per prompt
  let lastCallHash = null;
  let lastCallCount = 0;
  // Semantic loop: disabled — caused false positives on multi-step workflows (analyze_data columns→shape→sort)
  // const toolResourceCounts = {}; // "tool:resource" → count
  // const SEMANTIC_LOOP_THRESHOLD = 4; // Same tool+resource 4 times total → stop
  let didTokenCompact = false; // Only compact once per runGemini call

  for (let round = 0; round < MAX_ROUNDS; round++) {
    // Check if user sent "stop" / "cancel" (lock was released)
    if (!activeRequests.has(chatJid)) {
      console.log(`[${LLM_TAG}] Stopped by user after ${round} rounds`);
      return { text: "Request stopped.", toolLog };
    }

    // Layer 2: Cap old tool results — 20/80 split (from Gemini CLI)
    // Keep first 20% + last 80% of cap. End has errors/results/counts.
    if (round > 0) {
      const currentBoundary = contents.length - 2;
      for (let i = 0; i < currentBoundary; i++) {
        const entry = contents[i];
        if (!entry?.parts) continue;
        for (const part of entry.parts) {
          if (part.functionResponse?.response?.result) {
            const resultStr = JSON.stringify(part.functionResponse.response.result);
            if (resultStr.length > HISTORY_CAP) {
              const headLen = Math.round(HISTORY_CAP * 0.2); // first 20%
              const tailLen = HISTORY_CAP - headLen; // last 80%
              part.functionResponse.response.result = {
                _truncated: true,
                head: resultStr.slice(0, headLen),
                tail: resultStr.slice(-tailLen),
                _note: `[truncated, was ${resultStr.length} chars]`,
              };
            }
          }
        }
      }
    }

    const response = await llmGenerate({
      model: GEMINI_MODEL,
      contents,
      config,
      tools: activeTools,
    });

    // Track token usage
    const roundTokens = trackTokens(chatJid, response);

    // Mid-call compaction: if contents array is getting large (many tool rounds), summarize old turns
    // Prevents token burn from long tool chains (e.g. 24 rounds searching for a file)
    if (!didTokenCompact && contents.length > CONTENTS_COMPACT_THRESHOLD) {
      console.log(`[Memory] Contents ${contents.length} entries > ${CONTENTS_COMPACT_THRESHOLD} threshold — compacting...`);
      didTokenCompact = await compactContents(contents, chatJid);
    }

    // Check for function calls
    if (response.functionCalls && response.functionCalls.length > 0) {
      // Notify user on first tool call so they know we're working
      if (!notifiedUser) {
        notifiedUser = true;
        try {
          await sock.sendMessage(chatJid, { text: "⏳ Working on it..." });
          rememberSent("⏳ Working on it...");
        } catch (e) { /* ignore send errors */ }
      }

      // Add model's response to conversation
      contents.push(response.candidates[0].content);

      // Execute each tool call
      const elapsed = Math.round((Date.now() - toolStartTime) / 1000);
      const tokenInfo = roundTokens ? `, ${formatTokens(roundTokens.input)} in / ${formatTokens(roundTokens.output)} out` : "";
      console.log(`[${LLM_TAG}] Round ${round + 1}, ${response.functionCalls.length} tool(s), ${elapsed}s elapsed${tokenInfo}`);
      const functionResponses = [];
      const imageParts = []; // For read_file images — Gemini sees them visually
      for (const fc of response.functionCalls) {
        // Loop detection: same tool+args called N times consecutively → stop
        const callHash = fc.name + ":" + JSON.stringify(fc.args || {});
        if (callHash === lastCallHash) {
          lastCallCount++;
          if (lastCallCount >= LOOP_THRESHOLD) {
            console.warn(`[${LLM_TAG}] Loop detected: ${fc.name} called ${lastCallCount}x with same args`);
            functionResponses.push({
              functionResponse: {
                name: fc.name,
                response: { result: { error: `Loop detected: you've called ${fc.name} ${lastCallCount} times with identical args. Stop retrying and answer with what you have — if the data wasn't found, tell the user.` } },
              },
            });
            continue;
          }
        } else {
          lastCallHash = callHash;
          lastCallCount = 1;
        }

        // Semantic loop: disabled — exact-match detector (threshold 3) + tool descriptions are sufficient
        // const resource = fc.args?.folder || fc.args?.path || fc.args?.image_path || "";
        // if (resource) {
        //   const resourceKey = `${fc.name}:${resource}`;
        //   toolResourceCounts[resourceKey] = (toolResourceCounts[resourceKey] || 0) + 1;
        //   if (toolResourceCounts[resourceKey] >= SEMANTIC_LOOP_THRESHOLD) {
        //     console.warn(`[${LLM_TAG}] Semantic loop: ${fc.name} on ${resource} called ${toolResourceCounts[resourceKey]}x`);
        //     functionResponses.push({ functionResponse: { name: fc.name,
        //       response: { result: { error: `Called ${fc.name} on same path too many times.` } } } });
        //     continue;
        //   }
        // }

        // Dedup: skip if exact same tool+args already called this turn
        let result;
        if (toolCache.has(callHash)) {
          result = toolCache.get(callHash);
          console.log(`[${LLM_TAG}] Dedup skip: ${fc.name} (cached)`);
        } else {
          result = await executeTool(fc, sock, chatJid);
          toolCache.set(callHash, result);
        }

        // Tail tool calls: auto-send generated files to user without LLM round-trip
        // (extract_frame, crop_face, generate_chart → auto send_file)
        if (!result?.error) {
          const autoSendPath = (fc.name === "extract_frame" && result.path)
            || (fc.name === "crop_face" && result.path)
            || (fc.name === "generate_chart" && result.path);
          if (autoSendPath) {
            const sent = await sendFile(sock, chatJid, autoSendPath, `${PREFIX} ${pathModule.basename(autoSendPath)}`);
            if (sent) {
              result._auto_sent = true;
              console.log(`[${LLM_TAG}] Tail call: auto-sent ${pathModule.basename(autoSendPath)}`);
            }
          }
        }

        // Layer 1: TempStore — store large results as refs, give Gemini preview
        let geminiResult = result; // What Gemini sees (may be preview)
        if (REF_TOOLS.has(fc.name) && !result?.error) {
          const resultJson = JSON.stringify(result);
          if (resultJson.length > REF_THRESHOLD) {
            // Multi-query results: only search_images_visual with explicit queries:[] returns { "query1": result, "query2": result }
            // Split into separate refs per query so Gemini can reference each individually
            const isMultiQuery = fc.name === "search_images_visual"
              && fc.args?.queries && fc.args.queries.length > 1
              && typeof result === "object" && !Array.isArray(result);
            if (isMultiQuery) {
              const multiPreview = {};
              for (const [queryStr, queryResult] of Object.entries(result)) {
                const subJson = JSON.stringify(queryResult);
                if (subJson.length > REF_THRESHOLD) {
                  const subRefKey = storeRef(queryResult);
                  multiPreview[queryStr] = makeRefPreview(fc.name, queryResult, subRefKey);
                  console.log(`[TempStore] ${fc.name} "${queryStr.slice(0, 40)}..." → ${subRefKey}`);
                } else {
                  multiPreview[queryStr] = queryResult;
                }
              }
              geminiResult = multiPreview;
            } else {
              const refKey = storeRef(result);
              geminiResult = makeRefPreview(fc.name, result, refKey);
              console.log(`[TempStore] ${fc.name} → ${refKey} (${resultJson.length} chars → preview)`);
            }
          }
        }

        // Track tool calls for conversation memory context
        const argsShort = JSON.stringify(fc.args || {}).slice(0, 100);
        toolLog.push(`${fc.name}(${argsShort})`);

        // If read_file returned an image, include as inlineData so Gemini SEES it (capped)
        if (fc.name === "read_file" && result.type === "image" && result.data && inlineImageCount < MAX_INLINE_IMAGES) {
          inlineImageCount++;
          functionResponses.push({
            functionResponse: {
              name: fc.name,
              response: { result: { type: "image", path: result.path, size: result.size, note: "Image included as visual data — analyze it directly." } },
            },
          });
          imageParts.push({
            inlineData: {
              mimeType: result.mime_type,
              data: result.data,
            },
          });
        } else if (fc.name === "read_file" && result.type === "image" && inlineImageCount >= MAX_INLINE_IMAGES) {
          functionResponses.push({
            functionResponse: {
              name: fc.name,
              response: { result: { type: "image", path: result.path, size: result.size, note: `Image limit reached (${MAX_INLINE_IMAGES}). Use detect_faces or run_python for batch image analysis.` } },
            },
          });
        } else {
          functionResponses.push({
            functionResponse: {
              name: fc.name,
              response: { result: geminiResult },
            },
          });
        }
      }

      // Send tool results back to Gemini (with images if any)
      const parts = [...functionResponses, ...imageParts];

      // Round-based efficiency nudges (from SkillRL/ReCall research)
      if (round === 7) {
        parts.push({ text: "[System: You've used 8 tool rounds. Be efficient — only call more tools if truly needed, otherwise answer now.]" });
      } else if (round === 15) {
        parts.push({ text: "[System: 16 rounds used. Give your best answer with the information you have. Do not call more tools unless absolutely critical.]" });
      }

      contents.push({ role: "user", parts });
    } else {
      // No more tool calls — return the text response
      const text = response.text;
      if (!text) {
        const finishReason = response.candidates?.[0]?.finishReason;
        const safetyRatings = response.candidates?.[0]?.safetyRatings;
        console.error(`[${LLM_TAG}] Empty response. finishReason=${finishReason}, safety=${JSON.stringify(safetyRatings || [])}`);
        // MALFORMED_FUNCTION_CALL in no-tools mode: Gemini tried to call a tool
        // Only re-enable tools if this was a text message (not a bare image/file)
        if (finishReason === "MALFORMED_FUNCTION_CALL" && opts.noTools && round === 0 && !opts.inlineImage) {
          console.log(`[${LLM_TAG}] MALFORMED_FUNCTION_CALL in no-tools mode — retrying with tools`);
          opts.noTools = false;
          activeTools = getToolsForIntent(userMessage);
          continue; // retry this round
        }
      }
      return {
        text: text || "Sorry, I couldn't process that. Please try rephrasing your question.",
        toolLog,
      };
    }
  }

  // Max rounds reached — return what we have
  console.warn(`[${LLM_TAG}] Max rounds (${MAX_ROUNDS}) reached`);
  return { text: `I've completed ${MAX_ROUNDS} rounds of work. Here's what I did so far — let me know if you need me to continue.`, toolLog };
}

// --- Detect non-search messages (greetings, thanks, etc.) ---
const NON_SEARCH = /^(hi|hello|hey|thanks|thank you|ok|okay|bye|good morning|good night|gm|gn|yo|sup)[\s!?.]*$/i;

function isGreeting(text) {
  return NON_SEARCH.test(text.trim());
}

// --- Fallback: direct FTS search (when Gemini is unavailable) ---
async function fallbackSearch(query) {
  // Don't search for greetings
  if (isGreeting(query)) {
    return { text: "Hello! Smart search is temporarily unavailable (rate limit). Try a search query like \"find reliance invoice\".", files: [] };
  }

  const data = await apiGet(`/search?q=${encodeURIComponent(query)}&limit=${MAX_RESULTS}`);
  if (!data.results || data.results.length === 0) {
    return { text: `No results found for "${query}"`, files: [] };
  }

  let msg = `Found ${data.results.length} result(s) for "${query}":\n`;
  for (let i = 0; i < data.results.length; i++) {
    const r = data.results[i];
    const score = Math.round(r.score * 100);
    const snippet = r.snippet.length > 150 ? r.snippet.slice(0, 150) + "…" : r.snippet;
    msg += `\n*${i + 1}. ${r.title}* (${r.file_type.toUpperCase()}, ${score}%)\n${snippet}\n`;
  }

  const files = data.results.slice(0, MAX_FILES_TO_SEND).map(r => ({
    path: r.path, title: r.title, score: r.score, file_type: r.file_type,
  }));

  return { text: msg, files };
}

// --- Text chunking ---
function chunkText(text, limit = TEXT_CHUNK_LIMIT) {
  if (text.length <= limit) return [text];
  const chunks = [];
  let remaining = text;
  while (remaining.length > 0) {
    if (remaining.length <= limit) { chunks.push(remaining); break; }
    let breakAt = remaining.lastIndexOf("\n", limit);
    if (breakAt < limit * 0.5) breakAt = remaining.lastIndexOf(" ", limit);
    if (breakAt < limit * 0.5) breakAt = limit;
    chunks.push(remaining.slice(0, breakAt));
    remaining = remaining.slice(breakAt).trimStart();
  }
  return chunks;
}

// --- Send a file via WhatsApp ---
async function sendFile(sock, chatJid, filePath, caption) {
  if (!existsSync(filePath)) return false;

  const ext = pathModule.extname(filePath).toLowerCase();
  const fileName = pathModule.basename(filePath);
  const fileSize = statSync(filePath).size;
  const isImage = IMAGE_EXTENSIONS.has(ext);

  if (isImage && fileSize > MAX_IMAGE_SIZE) return false;
  if (!isImage && fileSize > MAX_DOC_SIZE) return false;

  const buffer = readFileSync(filePath);

  let sentMsg;
  if (isImage) {
    sentMsg = await sock.sendMessage(chatJid, {
      image: buffer,
      caption,
      mimetype: ext === ".png" ? "image/png" : "image/jpeg",
    });
  } else {
    const mimeMap = {
      ".pdf": "application/pdf",
      ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
      ".txt": "text/plain",
      ".csv": "text/csv",
      ".epub": "application/epub+zip",
    };
    sentMsg = await sock.sendMessage(chatJid, {
      document: buffer,
      mimetype: mimeMap[ext] || "application/octet-stream",
      fileName,
      caption,
    });
  }
  // Mark sent media as processed so echo doesn't trigger handleMedia (self-chat echo prevention)
  if (sentMsg?.key?.id) markProcessed(sentMsg.key.id);
  return true;
}

// --- Main bot ---
async function startBot() {
  // Check API key
  if (!process.env.GEMINI_API_KEY) {
    console.log("[Pinpoint] WARNING: GEMINI_API_KEY not set in .env — using fallback keyword search");
  } else {
    const toolCount = tools[0].functionDeclarations.length;
    console.log(`[Pinpoint] Gemini AI enabled (${GEMINI_MODEL}) — ${toolCount} tools + skills system`);
  }

  const apiOk = await apiPing();
  if (!apiOk) {
    console.log("[Pinpoint] WARNING: Python API not reachable at " + API_URL);
  } else {
    console.log("[Pinpoint] Python API connected at " + API_URL);
    // Load memory state from settings
    try {
      const setting = await apiGet("/setting?key=memory_enabled");
      // Default ON: only disable if explicitly set to "false"
      memoryEnabled = setting.value !== "false";
      if (memoryEnabled) {
        const ctx = await apiGet("/memory/context");
        memoryContext = ctx.text || "";
        console.log(`[Pinpoint] Memory ON (${ctx.count || 0} memories loaded)`);
      } else {
        console.log("[Pinpoint] Memory OFF (enable with /memory on)");
      }
    } catch (_) {
      console.log("[Pinpoint] Memory ON (default)");
    }

    // Load allowed users list
    await loadAllowedUsers();
    if (allowedUsers.size > 0) {
      console.log(`[Pinpoint] Allowed users: ${[...allowedUsers].join(", ")}`);
    }
  }

  // Restore creds from backup if main file is corrupted (OpenClaw pattern)
  const credsPath = pathModule.join(AUTH_DIR, "creds.json");
  const credsBackup = credsPath + ".bak";
  if (!existsSync(credsPath) && existsSync(credsBackup)) {
    try {
      const { copyFileSync } = require("fs");
      copyFileSync(credsBackup, credsPath);
      console.log("[Pinpoint] Restored creds from backup");
    } catch (_) {}
  }

  const { state, saveCreds: _saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  // Queued credential saves — prevent concurrent writes corrupting creds.json (OpenClaw pattern)
  let credsSaveQueue = Promise.resolve();
  const saveCreds = () => {
    credsSaveQueue = credsSaveQueue.then(async () => {
      try {
        // Backup before saving — only if current creds are valid JSON
        if (existsSync(credsPath)) {
          try {
            const raw = readFileSync(credsPath, "utf-8");
            JSON.parse(raw); // validate before backup
            const { copyFileSync } = require("fs");
            copyFileSync(credsPath, credsBackup);
          } catch (_) {} // keep existing backup if invalid
        }
      } catch (_) {}
      return _saveCreds();
    }).catch((err) => {
      console.error("[Pinpoint] Creds save error:", err.message);
    });
  };

  const sock = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    version, logger,
    printQRInTerminal: false,
    browser: ["Pinpoint", "CLI", "1.0"],
    syncFullHistory: false,
    markOnlineOnConnect: false,
  });

  // Handle WebSocket errors to prevent unhandled exceptions (OpenClaw pattern)
  if (sock.ws && typeof sock.ws.on === "function") {
    sock.ws.on("error", (err) => {
      console.error("[Pinpoint] WebSocket error:", err.message);
    });
  }

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      console.log("\n[Pinpoint] Scan this QR code with WhatsApp:\n");
      qrcode.generate(qr, { small: true });
    }
    if (connection === "open") {
      reconnectAttempt = 0;
      console.log(`[Pinpoint] Connected! JID: ${sock.user?.id}`);
      console.log("[Pinpoint] Send yourself a message to search.\n");

      // Load reminders from DB on connect
      loadReminders().catch(e => console.error("[Reminder] Load failed:", e.message));

      // Start reminder checker (every 30 seconds)
      if (!global._reminderInterval) {
        global._reminderInterval = setInterval(async () => {
          const now = Date.now();
          const due = reminders.filter(r => r.triggerAt <= now);
          for (const r of due) {
            try {
              const label = r.repeat ? `⏰ *Reminder (${r.repeat}):* ${r.message}` : `⏰ *Reminder:* ${r.message}`;
              await sock.sendMessage(r.chatJid, { text: label });
              console.log(`[Reminder] Sent: "${r.message}" to ${r.chatJid}${r.repeat ? ` (${r.repeat})` : ""}`);
            } catch (e) {
              console.error(`[Reminder] Failed to send: ${e.message}`);
            }
          }
          // Handle sent reminders: reschedule recurring, remove one-time
          for (const r of due) {
            const idx = reminders.indexOf(r);
            if (idx === -1) continue;
            if (r.repeat) {
              // Reschedule to next occurrence
              const next = getNextOccurrence(r.triggerAt, r.repeat);
              if (next) {
                r.triggerAt = next.getTime();
                try { await apiPut(`/reminders/${r.id}?trigger_at=${encodeURIComponent(next.toISOString())}`, {}); } catch (_) {}
                console.log(`[Reminder] Rescheduled "${r.message}" → ${next.toISOString()}`);
              } else {
                reminders.splice(idx, 1);
                try { await apiDelete(`/reminders/${r.id}`); } catch (_) {}
              }
            } else {
              reminders.splice(idx, 1);
              try { await apiDelete(`/reminders/${r.id}`); } catch (_) {}
            }
          }
        }, 30000);
      }
    }
    if (connection === "close") {
      const code = lastDisconnect?.error?.output?.statusCode;
      const shouldReconnect = code !== DisconnectReason.loggedOut;
      console.log(`[Pinpoint] Disconnected (${code}). ${shouldReconnect ? "Reconnecting..." : "Logged out."}`);
      // Only clear auth on explicit logout (401) — never on 515
      // 515 means stream error, just reconnect with existing creds
      if (code === DisconnectReason.loggedOut) {
        console.log("[Pinpoint] Logged out — clearing auth for fresh login...");
        try {
          const files = readdirSync(AUTH_DIR);
          for (const f of files) unlinkSync(pathModule.join(AUTH_DIR, f));
        } catch (_) {}
      }
      if (shouldReconnect && reconnectAttempt < RECONNECT.maxAttempts) {
        const baseDelay = Math.min(RECONNECT.initialMs * Math.pow(RECONNECT.factor, reconnectAttempt), RECONNECT.maxMs);
        const jitter = baseDelay * RECONNECT.jitter * (Math.random() * 2 - 1); // ±25%
        const delay = Math.max(1000, Math.round(baseDelay + jitter));
        reconnectAttempt++;
        console.log(`[Pinpoint] Reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${reconnectAttempt}/${RECONNECT.maxAttempts})`);
        setTimeout(startBot, delay);
      }
    }
  });

  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify" && type !== "append") return;
    for (const msg of messages) {
      try { await handleMessage(sock, msg); }
      catch (err) { console.error("[Pinpoint] Error:", err.message); }
    }
  });

  return sock;
}

// --- Handle received media files (save to computer) ---

function generateFilename(mediaType, mimetype) {
  const now = new Date();
  const ts = now.getFullYear().toString()
    + String(now.getMonth() + 1).padStart(2, "0")
    + String(now.getDate()).padStart(2, "0")
    + "_" + String(now.getHours()).padStart(2, "0")
    + String(now.getMinutes()).padStart(2, "0")
    + String(now.getSeconds()).padStart(2, "0");
  const prefixes = { imageMessage: "IMG", videoMessage: "VID", audioMessage: "AUD" };
  const prefix = prefixes[mediaType] || "FILE";
  const ext = MIME_TO_EXT[mimetype] || ".bin";
  return `${prefix}_${ts}${ext}`;
}

function parseSaveFolder(caption) {
  if (!caption) return null;
  // Match patterns like "save to Desktop/work" or "save in Documents"
  const match = caption.match(/save\s+(?:to|in)\s+(.+)/i);
  if (!match) return null;
  let folder = match[1].trim();
  // Resolve common folder names
  const lower = folder.toLowerCase();
  if (lower.startsWith("downloads")) folder = folder.replace(/^downloads/i, DOWNLOADS);
  else if (lower.startsWith("desktop")) folder = folder.replace(/^desktop/i, DESKTOP);
  else if (lower.startsWith("documents")) folder = folder.replace(/^documents/i, DOCUMENTS);
  else if (lower.startsWith("pictures")) folder = folder.replace(/^pictures/i, PICTURES);
  else if (!pathModule.isAbsolute(folder)) folder = pathModule.join(DEFAULT_SAVE_FOLDER, folder);
  return folder;
}

function uniquePath(filePath) {
  if (!existsSync(filePath)) return filePath;
  const dir = pathModule.dirname(filePath);
  const ext = pathModule.extname(filePath);
  const base = pathModule.basename(filePath, ext);
  let i = 1;
  while (existsSync(pathModule.join(dir, `${base} (${i})${ext}`))) i++;
  return pathModule.join(dir, `${base} (${i})${ext}`);
}

async function handleMedia(sock, msg, chatJid) {
  const msgType = getContentType(msg.message);
  if (!msgType) return false;

  // Only handle media types
  const mediaTypes = new Set(["imageMessage", "documentMessage", "videoMessage", "audioMessage"]);
  if (!mediaTypes.has(msgType)) return false;

  const media = msg.message[msgType];
  if (!media) return false;

  // Get metadata
  const caption = media.caption || "";
  const mimetype = media.mimetype || "application/octet-stream";
  const fileSize = Number(media.fileLength || 0);

  // Determine filename
  let fileName;
  if (msgType === "documentMessage" && media.fileName) {
    fileName = media.fileName;
  } else {
    fileName = generateFilename(msgType, mimetype);
  }

  // Determine save folder
  // If user said "save to X" → save permanently there
  // If caption has a question (processing request) → save to temp, cleaned up on session reset
  // If no caption + image → save to temp (will be auto-captioned)
  // If no caption + non-image → save permanently to default
  const customFolder = parseSaveFolder(caption);
  const cleanCaption_ = caption.replace(/save\s+(?:to|in)\s+.+/i, "").trim();
  const hasCaptionText = cleanCaption_.length > 2;
  const isImageMsg = msgType === "imageMessage";
  const isProcessingOnly = !customFolder && (hasCaptionText || isImageMsg);
  const saveFolder = customFolder || (isProcessingOnly ? TEMP_MEDIA_DIR : DEFAULT_SAVE_FOLDER);

  // Create folder if needed
  mkdirSync(saveFolder, { recursive: true });

  // Download the file
  let buffer;
  try {
    buffer = await downloadMediaMessage(msg, "buffer", {}, {
      logger,
      reuploadRequest: sock.updateMediaMessage,
    });
  } catch (err) {
    console.error("[Pinpoint] Media download failed:", err.message);
    const reply = `${PREFIX} Failed to download file: ${err.message}`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    return true;
  }

  // Save to disk (unique name to avoid overwrite)
  const savePath = uniquePath(pathModule.join(saveFolder, fileName));
  writeFileSync(savePath, buffer);

  const sizeStr = _humanSize(buffer.length);
  const relFolder = saveFolder.replace(USER_HOME, "~");
  console.log(`[Pinpoint] Saved file: ${fileName} (${sizeStr}) → ${saveFolder}${isProcessingOnly ? " (temp)" : ""}`);

  // Only show "Saved" confirmation for permanent saves (not processing-only temp files)
  if (!isProcessingOnly) {
    const reply = `${PREFIX} Saved *${pathModule.basename(savePath)}* to ${relFolder} (${sizeStr})`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);

    // Save file receipt to conversation memory so Gemini knows about it in future messages
    // (e.g. user sends photo now, then says "find this person" later)
    await saveMessage(chatJid, "user", `[Sent a file: ${pathModule.basename(savePath)} saved at ${savePath}]`);
    await saveMessage(chatJid, "assistant", `Saved ${pathModule.basename(savePath)} to ${relFolder} (${sizeStr})`);
  }

  // Build the message for Gemini
  const cleanCaption = caption.replace(/save\s+(?:to|in)\s+.+/i, "").trim();
  const isImage = msgType === "imageMessage";
  const hasCaption = cleanCaption && cleanCaption.length > 2;

  // For images: send inline to Gemini (it sees the image immediately, no read_file needed)
  // For non-images with caption: send file path + caption to Gemini
  const shouldProcess = hasCaption || isImage;

  if (shouldProcess && !activeRequests.has(chatJid)) {
    const userMsg = hasCaption
      ? `[File: ${pathModule.basename(savePath)} at ${savePath}]\n${cleanCaption}`
      : `[Photo: ${pathModule.basename(savePath)} at ${savePath}]\nUser sent this with no instruction. Just ask what they want to do with it.`;
    const myRequestId = ++requestCounter;
    activeRequests.set(chatJid, { msg: hasCaption ? cleanCaption : "[photo]", startTime: Date.now(), id: myRequestId });
    try {
      await sock.sendPresenceUpdate("composing", chatJid);

      // If image, read as base64 and send inline — Gemini sees it directly
      let inlineImage = null;
      if (isImage) {
        try {
          const imgData = readFileSync(savePath).toString("base64");
          inlineImage = { mimeType: mimetype, data: imgData };
          // Save for follow-up messages (e.g. user sends photo, then asks "who is this?")
          lastImage.set(chatJid, { mimeType: mimetype, data: imgData, path: savePath, ts: Date.now() });
          console.log(`[Pinpoint] Sending image inline to ${LLM_TAG} (${sizeStr})`);
        } catch (e) {
          console.error("[Pinpoint] Failed to read image for inline:", e.message);
        }
      }

      // No caption + image → disable tools (Gemini just describes, no fishing expedition)
      const noTools = isImage && !hasCaption;
      const result = await runGemini(userMsg, sock, chatJid, { inlineImage, noTools });
      const current = activeRequests.get(chatJid);
      if (!current || current.id !== myRequestId) {
        console.log(`[Pinpoint] Discarded media result (stopped)`);
      } else {
        console.log(`[${LLM_TAG}] Response: ${result.text.length} chars`);
        const replyText = `${PREFIX} ${markdownToWhatsApp(result.text)}`;
        const chunks = chunkText(replyText);
        for (const chunk of chunks) {
          await sock.sendMessage(chatJid, { text: chunk });
          rememberSent(chunk);
        }
        let assistantSave = result.text;
        if (result.toolLog && result.toolLog.length > 0) {
          assistantSave = `[Actions: ${result.toolLog.join(", ")}]\n${result.text}`;
        }
        await saveMessage(chatJid, "user", hasCaption ? cleanCaption : `[Sent photo: ${pathModule.basename(savePath)}]`);
        await saveMessage(chatJid, "assistant", assistantSave);
      }
    } catch (err) {
      console.error(`[${LLM_TAG}] Error on media processing:`, err.message);
    } finally {
      const current = activeRequests.get(chatJid);
      if (current && current.id === myRequestId) {
        activeRequests.delete(chatJid);
      }
    }
  } else if (shouldProcess) {
    console.log(`[Pinpoint] Skipped media processing — Gemini already running`);
    // Save file receipt to conversation memory so Gemini knows about it later
    await saveMessage(chatJid, "user", `[Sent file: ${pathModule.basename(savePath)} saved at ${savePath}]`);
    // Don't set lastImage here — skipped images are usually bot echoes (send_file → WhatsApp echo)
  }

  await sock.sendPresenceUpdate("paused", chatJid);
  return true; // handled
}

function _humanSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

async function handleMessage(sock, msg) {
  const key = msg.key;
  const chatJid = key.remoteJid;
  if (chatJid === "status@broadcast") return;

  const msgId = key.id;
  if (processedIds.has(msgId)) return;
  markProcessed(msgId);

  // Access control: self-chat always allowed, others need /allow permission
  const myNumber = sock.user?.id?.split(":")[0]?.split("@")[0];
  const myLid = sock.user?.lid?.split(":")[0]?.split("@")[0];
  const chatNumber = chatJid?.split("@")[0];
  const isGroup = chatJid?.endsWith("@g.us");
  const isSelfChat = !isGroup && (
    (myNumber && chatNumber && myNumber === chatNumber)
    || (myLid && chatNumber && myLid === chatNumber)
  );

  // Debug: log JID matching (remove after confirming)
  if (!isGroup && !isSelfChat) {
    console.log(`[Debug] chatJid=${chatJid} myNumber=${myNumber} myLid=${myLid} fromMe=${key.fromMe}`);
  }

  if (!isSelfChat && !isAllowedUser(chatJid)) {
    if (!isGroup) console.log(`[Pinpoint] Ignored message from: ${chatJid} (not allowed). To allow: /allow ${chatJid.split("@")[0]}`);
    return;
  }

  // Allowed users: "pinpoint" starts session, "bye/stop pinpoint" ends it, 60min idle timeout
  const isAllowed = !isSelfChat && isAllowedUser(chatJid);
  if (isAllowed) {
    const peekText = (msg.message?.conversation || msg.message?.extendedTextMessage?.text
      || msg.message?.imageMessage?.caption || msg.message?.documentMessage?.caption || "").toLowerCase();
    const hasSession = allowedSessions.has(chatJid) && (Date.now() - allowedSessions.get(chatJid)) < IDLE_TIMEOUT_MS;

    // End session: "bye pinpoint" / "by pinpoint" / "stop pinpoint" / "exit pinpoint" / "pinpoint bye/stop"
    const isEndCmd = /\b(bye|by|stop|exit|quit|close)\b.*\bpinpoint\b|\bpinpoint\b.*\b(bye|by|stop|exit|quit|close)\b/.test(peekText);
    if (hasSession && isEndCmd) {
      allowedSessions.delete(chatJid);
      const endMsg = `${PREFIX} Session ended. Say "pinpoint" anytime to start again.`;
      await sock.sendMessage(chatJid, { text: endMsg });
      rememberSent(endMsg);
      return;
    }

    // No active session — need "pinpoint" to start
    if (!hasSession) {
      if (!peekText.includes("pinpoint")) return;
      // Starting new session — greet and return (don't send "pinpoint" to Gemini)
      allowedSessions.set(chatJid, Date.now());
      const greetMsg = `${PREFIX} Hi! I'm Pinpoint. How can I help? (Say "bye pinpoint" when done)`;
      await sock.sendMessage(chatJid, { text: greetMsg });
      rememberSent(greetMsg);
      return;
    }

    // Refresh session timer
    allowedSessions.set(chatJid, Date.now());
  }

  // Send read receipt (blue ticks) so user knows bot received it
  try { await sock.readMessages([key]); } catch (_) {}

  // Check for media messages FIRST (before text check)
  const msgType = getContentType(msg.message);
  if (msgType && msgType !== "conversation" && msgType !== "extendedTextMessage") {
    const handled = await handleMedia(sock, msg, chatJid);
    if (handled) return;
  }

  const rawText = msg.message?.conversation || msg.message?.extendedTextMessage?.text || "";
  if (!rawText.trim()) return;
  if (wasSentByUs(rawText.trim())) return;
  if (rawText.trim().startsWith(PREFIX)) return;

  // Debounce: combine rapid messages within 1.5s (null = merged into another message)
  const userMsg = await debounceMessage(chatJid, rawText.trim(), sock);
  if (!userMsg) return; // This message was merged into another — skip
  console.log(`[Pinpoint] Message: "${userMsg}"`);

  // Handle stop/cancel — abort in-progress request
  const cmdLower = userMsg.toLowerCase();
  if ((cmdLower === "stop" || cmdLower === "cancel" || cmdLower === "/stop") && activeRequests.has(chatJid)) {
    const active = activeRequests.get(chatJid);
    activeRequests.delete(chatJid); // Release lock immediately — next message can proceed
    const reply = `${PREFIX} Stopped.`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    console.log(`[Pinpoint] Aborted request: "${active.msg}"`);
    return;
  }

  // --- Admin commands (self-chat only) ---
  if (isSelfChat && cmdLower.startsWith("/allow ")) {
    const number = userMsg.slice(7).trim().replace(/[^0-9]/g, "");
    if (!number || number.length < 10) {
      await sock.sendMessage(chatJid, { text: `${PREFIX} Invalid number. Use: /allow 919876543210 (include country code)` });
      return;
    }
    allowedUsers.add(number);
    await saveAllowedUsers();
    const reply = `${PREFIX} Allowed ${number}. They can message "pinpoint" to use it.`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    console.log(`[Pinpoint] Allowed user: ${number}`);
    return;
  }
  if (isSelfChat && cmdLower.startsWith("/revoke ")) {
    const number = userMsg.slice(8).trim().replace(/[^0-9]/g, "");
    // Remove all matching entries (phone number + any resolved LIDs)
    const removed = [];
    for (const id of [...allowedUsers]) {
      if (id === number || id.startsWith(number)) {
        allowedUsers.delete(id);
        removed.push(id);
      }
    }
    if (removed.length > 0) {
      await saveAllowedUsers();
      const reply = `${PREFIX} Revoked access for ${removed.join(", ")}.`;
      await sock.sendMessage(chatJid, { text: reply });
      rememberSent(reply);
      console.log(`[Pinpoint] Revoked user: ${removed.join(", ")}`);
    } else {
      await sock.sendMessage(chatJid, { text: `${PREFIX} ${number} is not in the allowed list.` });
    }
    return;
  }
  if (isSelfChat && cmdLower === "/allowed") {
    const list = allowedUsers.size > 0 ? [...allowedUsers].join(", ") : "None";
    const reply = `${PREFIX} Allowed users: ${list}`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    return;
  }

  // Handle slash commands (before Gemini)
  if (cmdLower === "/new" || cmdLower === "/reset") {
    const deleted = await resetSession(chatJid);
    delete sessionCosts[chatJid];
    const reply = `${PREFIX} Conversation reset. Fresh start! (cleared ${deleted} messages)`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    console.log(`[Pinpoint] Session reset: ${deleted} messages cleared`);
    return;
  }
  if (cmdLower === "/history") {
    const hist = await loadHistory(chatJid);
    if (hist.messages.length === 0) {
      const reply = `${PREFIX} No conversation history.`;
      await sock.sendMessage(chatJid, { text: reply });
      rememberSent(reply);
    } else {
      let reply = `${PREFIX} Last ${hist.messages.length} messages (${hist.message_count} total):\n`;
      for (const m of hist.messages.slice(-10)) {
        const role = m.role === "user" ? "You" : "Bot";
        const snippet = m.content.length > 80 ? m.content.slice(0, 80) + "…" : m.content;
        reply += `\n*${role}:* ${snippet}`;
      }
      const chunks = chunkText(reply);
      for (const chunk of chunks) {
        await sock.sendMessage(chatJid, { text: chunk });
        rememberSent(chunk);
      }
    }
    return;
  }
  if (cmdLower === "/memory on" || cmdLower === "/memory off") {
    const on = cmdLower === "/memory on";
    memoryEnabled = on;
    try { await apiPost(`/setting?key=memory_enabled&value=${on}`, {}); } catch (_) {}
    if (on) {
      try { const ctx = await apiGet("/memory/context"); memoryContext = ctx.text || ""; } catch (_) {}
    }
    const reply = `${PREFIX} Memory ${on ? "enabled" : "disabled"}.${on ? " I'll remember personal facts you share." : " I won't save or recall memories."}`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    console.log(`[Pinpoint] Memory ${on ? "ON" : "OFF"}`);
    return;
  }
  if (cmdLower === "/memory" || cmdLower === "/memory status") {
    let reply = `${PREFIX} Memory is ${memoryEnabled ? "ON" : "OFF"}.`;
    if (memoryEnabled) {
      try {
        const list = await apiGet("/memory/list?limit=50");
        reply += ` ${list.count} memories saved.`;
        if (list.count > 0) {
          reply += "\n";
          for (const m of list.memories.slice(0, 10)) {
            reply += `\n[${m.category}] ${m.fact}`;
          }
          if (list.count > 10) reply += `\n\n…and ${list.count - 10} more.`;
        }
      } catch (_) {}
    } else {
      reply += " Use /memory on to enable.";
    }
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    return;
  }

  if (cmdLower === "/cost") {
    const reply = `${PREFIX} ${getCostSummary(chatJid)}`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    return;
  }

  if (cmdLower === "/help") {
    const reply = `${PREFIX} *Commands:*
/new or /reset — Fresh conversation
/history — Recent messages
/memory on — Enable persistent memory
/memory off — Disable memory
/memory — Show saved memories
/cost — Token usage & estimated cost
/allow 91XXXXXXXXXX — Give someone Pinpoint access
/revoke 91XXXXXXXXXX — Remove their access
/allowed — List allowed users
/help — This list
stop — Cancel current request`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    return;
  }
  // Catch unknown / commands — don't send to Gemini
  if (cmdLower.startsWith("/")) {
    const reply = `${PREFIX} Unknown command: ${userMsg}\nType /help for available commands.`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    return;
  }

  // Processing lock: if already running Gemini for this chat, save to memory and skip
  if (activeRequests.has(chatJid)) {
    const active = activeRequests.get(chatJid);
    const elapsed = Math.round((Date.now() - active.startTime) / 1000);
    const snippet = active.msg.length > 60 ? active.msg.slice(0, 60) + "…" : active.msg;
    const reply = `${PREFIX} Still working on your request (${elapsed}s)… "${snippet}"`;
    await sock.sendMessage(chatJid, { text: reply });
    rememberSent(reply);
    // Save blocked message to conversation memory so Gemini sees it in the next call
    await saveMessage(chatJid, "user", userMsg);
    console.log(`[Pinpoint] Blocked concurrent request (active: "${snippet}", ${elapsed}s) — saved to memory`);
    return;
  }

  const myRequestId = ++requestCounter;
  activeRequests.set(chatJid, { msg: userMsg, startTime: Date.now(), id: myRequestId });

  try {
    await sock.sendPresenceUpdate("composing", chatJid);

    // Run Gemini (or fallback to direct search)
    let result;
    try {
      if (process.env.GEMINI_API_KEY) {
        // Simple messages (greetings, thanks, short replies) don't need tools
        const isSimple = /^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|bye|good morning|good night|gm|gn|hm+|lol|haha|cool|nice|great|awesome|sure|nope|yep|yea|ya|fine|alright|amazing|wow|perfect|got it|oh|damn|omg|hehe|bruh|exactly|right|correct|wrong|good|bad|done|hmm|ah|whoa|dope|sick|sweet|beautiful|wonderful|brilliant|excellent|fantastic|superb|impressive|neat|solid|lit|fire|legit|bet|word|ooh|aah|yay|woah|geez|omw|ty|thx|np|gg|kk|ikr|imo|fyi|asap|🔥|💯|👏|😍|🤩|🥳|💪|🎉|✅|🙌|👌|😭|🤣|😎|💀|🫡|👍|🙏|❤️|😂|😊)\s*[.!?]*$/i.test(userMsg.trim());
        const noTools = isSimple;
        // Re-inject last image only if recent (< 2 min TTL) — prevents stale image re-injection
        const prevImg = lastImage.get(chatJid);
        const imgAge = prevImg ? Date.now() - (prevImg.ts || 0) : Infinity;
        const inlineImage = (prevImg && imgAge < 120000) ? prevImg : null;
        // Prepend image path so Gemini knows where it is (for tools like crop_image, detect_objects)
        let geminiMsg = userMsg;
        if (inlineImage && prevImg.path) {
          geminiMsg = `[Image: ${pathModule.basename(prevImg.path)} at ${prevImg.path}]\n${userMsg}`;
        }
        result = await runGemini(geminiMsg, sock, chatJid, { noTools, inlineImage });
        // Clear lastImage after use (one-shot re-injection)
        if (inlineImage) lastImage.delete(chatJid);
        console.log(`[${LLM_TAG}] Response: ${result.text.length} chars`);
      } else {
        result = await fallbackSearch(userMsg);
      }
    } catch (err) {
      console.error(`[${LLM_TAG}] Error:`, err.message);
      // Fallback to direct keyword search
      try {
        result = await fallbackSearch(userMsg);
        console.log("[Pinpoint] Fell back to keyword search");
      } catch (err2) {
        const errMsg = `${PREFIX} Search failed. Is the Python API running?`;
        await sock.sendMessage(chatJid, { text: errMsg });
        rememberSent(errMsg);
        await sock.sendPresenceUpdate("paused", chatJid);
        return;
      }
    }

    // If this request was stopped (lock released by "stop"), discard result silently
    const current = activeRequests.get(chatJid);
    if (!current || current.id !== myRequestId) {
      console.log(`[Pinpoint] Discarded result for stopped request: "${userMsg.slice(0, 40)}"`);
      return;
    }

    // Send text reply (markdown → WhatsApp format, chunked)
    const replyText = `${PREFIX} ${markdownToWhatsApp(result.text)}`;
    const chunks = chunkText(replyText);
    for (const chunk of chunks) {
      await sock.sendMessage(chatJid, { text: chunk });
      rememberSent(chunk);
    }

    // Save conversation to memory with tool context for continuity
    await saveMessage(chatJid, "user", userMsg);
    // Enrich assistant message with tool context so Gemini knows what it did last turn
    let assistantSave = result.text;
    if (result.toolLog && result.toolLog.length > 0) {
      const toolSummary = result.toolLog.join(", ");
      assistantSave = `[Actions: ${toolSummary}]\n${result.text}`;
    }
    await saveMessage(chatJid, "assistant", assistantSave);

    await sock.sendPresenceUpdate("paused", chatJid);
    const sc = sessionCosts[chatJid];
    const tokenSummary = sc ? `, ${formatTokens(sc.input + sc.output)} tokens` : "";
    console.log(`[Pinpoint] Done (${replyText.length} chars${tokenSummary}, session: ${chatJid.slice(0, 12)}…)`);
  } finally {
    // Only release lock if we still own it (stop handler may have already released)
    const current = activeRequests.get(chatJid);
    if (current && current.id === myRequestId) {
      activeRequests.delete(chatJid);
    }
  }
}

// --- Start ---
console.log("=== Pinpoint WhatsApp Bot ===\n");
startBot().catch((err) => {
  console.error("[Pinpoint] Fatal error:", err);
  process.exit(1);
});
