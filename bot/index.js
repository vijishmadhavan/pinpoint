/**
 * Pinpoint — WhatsApp Bot with Gemini AI (60+ tools + skills system + file receive)
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

// --- Extracted modules ---
const {
  TOOL_DECLARATIONS,
  getToolsForIntent,
  clearIntentCache,
  hasActiveIntent,
  buildToolRoutes,
  preValidate,
  summarizeToolResult,
} = require("./src/tools");
const { getSystemPrompt, USER_HOME, DOWNLOADS, DOCUMENTS, DESKTOP, PICTURES } = require("./src/skills");
const llm = require("./src/llm");

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
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-flash-lite-preview";
// No tool-calling timeout — batch mode handles long operations in single calls (not token-burning loops)
const IDLE_TIMEOUT_MS = 60 * 60 * 1000; // 60 minutes — auto-reset conversation
const MAX_HISTORY_MESSAGES = 50; // last 50 messages passed to Gemini (Claude Code sends ALL — we cap for token budget)
const MAX_INLINE_IMAGES = 5; // Max images sent as visual data per turn

const DEBOUNCE_MS = 1500; // Combine rapid messages within 1.5s

// Log to file so we can check what happened (tee stdout → file)
const logFile = pathModule.join(__dirname, "..", "pinpoint.log");
const logStream = require("fs").createWriteStream(logFile, { flags: "a" });
const origLog = console.log,
  origWarn = console.warn,
  origErr = console.error;
function _ts() {
  return new Date().toISOString().slice(11, 19);
}
console.log = (...a) => {
  const s = a.map(String).join(" ");
  origLog(s);
  logStream.write(`${_ts()} ${s}\n`);
};
console.warn = (...a) => {
  const s = a.map(String).join(" ");
  origWarn(s);
  logStream.write(`${_ts()} WARN ${s}\n`);
};
console.error = (...a) => {
  const s = a.map(String).join(" ");
  origErr(s);
  logStream.write(`${_ts()} ERR ${s}\n`);
};

// Processing lock: prevent concurrent Gemini calls per chat (causes context loss)
const activeRequests = new Map(); // chatJid → { msg, startTime, id }
const lastImage = new Map(); // chatJid → { mimeType, data (base64), path, ts } — for follow-up re-injection (2 min TTL)
let requestCounter = 0;
let currentSock = null; // Module-level sock reference for reminders (survives reconnects)

// Persistent memory: on by default (all local, single user)
let memoryEnabled = true;
let memoryContext = ""; // Loaded from API, injected into system prompt

// Allowed users: phone numbers/LIDs that can use Pinpoint (managed via /allow, /revoke)
const allowedUsers = new Set();
const allowedSessions = new Map(); // chatJid → last activity timestamp (active session)

// Cost tracking: per-session token usage (OpenCode-inspired)
const sessionCosts = {}; // chatJid → { input, output, rounds, started }
const TOKEN_COST_INPUT = 0.25 / 1_000_000; // gemini-3.1-flash-lite-preview $/token (input)
const TOKEN_COST_OUTPUT = 1.5 / 1_000_000; // gemini-3.1-flash-lite-preview $/token (output, includes thinking)

// --- Action Ledger: structural truth enforcement (OpenClaw-inspired) ---
// Tracks every mutating tool call + real outcome. Injected into every LLM call.
// The LLM sees "## Actions Taken" with exact counts — cannot invent outcomes.
const actionLedger = {}; // chatJid → [{ tool, summary, outcome, ts }]
const MUTATING_TOOLS = new Set([
  "batch_move",
  "move_file",
  "copy_file",
  "delete_file",
  "write_file",
  "batch_rename",
  "create_folder",
  "generate_excel",
  "merge_pdf",
  "split_pdf",
  "resize_image",
  "convert_image",
  "crop_image",
  "compress_files",
  "extract_archive",
  "images_to_pdf",
  "pdf_to_images",
  "download_url",
  "compress_pdf",
  "add_page_numbers",
  "pdf_to_word",
  "organize_pdf",
  "pdf_to_excel",
  "cull_photos",
  "group_photos",
  "gmail_send",
  "calendar_create",
  "drive_upload",
]);

function recordAction(chatJid, toolName, args, result) {
  if (!actionLedger[chatJid]) actionLedger[chatJid] = [];
  const entry = { tool: toolName, ts: Date.now() };

  // Build a truthful one-line summary from the ACTUAL result
  if (result?.error) {
    entry.outcome = "FAILED";
    entry.summary = `${toolName} → ERROR: ${String(result.error).slice(0, 100)}`;
  } else if (toolName === "batch_move") {
    const moved = result?.moved_count ?? 0;
    const skipped = result?.skipped_count ?? 0;
    const errors = result?.error_count ?? 0;
    const action = result?.action || "moved";
    const dest = args?.destination || "?";
    entry.outcome = moved > 0 ? "OK" : "NOTHING_DONE";
    entry.summary = `batch_move → ${moved} ${action}, ${skipped} skipped, ${errors} errors → ${dest}`;
  } else if (toolName === "move_file" || toolName === "copy_file") {
    const action = result?.action || toolName.replace("_file", "d");
    const src = args?.source ? pathModule.basename(args.source) : "?";
    const dest = args?.destination || "?";
    entry.outcome = result?.success ? "OK" : "FAILED";
    entry.summary = `${toolName} → ${action} ${src} → ${dest}`;
  } else if (toolName === "delete_file") {
    entry.outcome = result?.success ? "OK" : "FAILED";
    entry.summary = `delete_file → ${result?.success ? "deleted" : "FAILED"} ${args?.path ? pathModule.basename(args.path) : "?"}`;
  } else if (toolName === "write_file") {
    entry.outcome = result?.success ? "OK" : "FAILED";
    entry.summary = `write_file → ${result?.success ? "created" : "FAILED"} ${result?.path ? pathModule.basename(result.path) : args?.path ? pathModule.basename(args.path) : "?"}`;
  } else if (toolName === "create_folder") {
    entry.outcome = "OK";
    entry.summary = `create_folder → ${result?.already_existed ? "already existed" : "created"} ${result?.path || args?.path || "?"}`;
  } else if (toolName === "batch_rename") {
    const renamed = result?.renamed_count ?? result?.renamed ?? 0;
    entry.outcome = renamed > 0 ? "OK" : "NOTHING_DONE";
    entry.summary = `batch_rename → ${renamed} renamed, ${result?.error_count ?? 0} errors`;
  } else if (toolName === "generate_excel") {
    entry.outcome = result?.success ? "OK" : "FAILED";
    entry.summary = `generate_excel → ${result?.path ? pathModule.basename(result.path) : "?"}`;
  } else if (toolName === "compress_files") {
    entry.outcome = result?.success ? "OK" : "FAILED";
    entry.summary = `compress_files → ${result?.path ? pathModule.basename(result.path) : "?"} (${result?.file_count ?? "?"} files)`;
  } else if (toolName === "cull_photos") {
    entry.outcome = result?.started ? "STARTED" : "FAILED";
    entry.summary = `cull_photos → ${result?.started ? `started culling ${result.total_images} photos (keep ${result.keep_pct}%)` : "FAILED"}`;
  } else if (toolName === "group_photos") {
    entry.outcome = result?.started ? "STARTED" : "FAILED";
    entry.summary = `group_photos → ${result?.started ? `started grouping ${result.total_images} photos into ${result.categories?.length ?? "?"} categories` : "FAILED"}`;
  } else {
    // Generic mutating tool
    entry.outcome = result?.success !== false ? "OK" : "FAILED";
    entry.summary = `${toolName} → ${result?.success !== false ? "done" : "FAILED"}`;
    if (result?.path) entry.summary += ` → ${pathModule.basename(result.path)}`;
  }

  actionLedger[chatJid].push(entry);
  // Cap at 50 entries per session (oldest dropped)
  if (actionLedger[chatJid].length > 50) actionLedger[chatJid].shift();
}

function getActionLedgerText(chatJid) {
  const entries = actionLedger[chatJid];
  if (!entries || entries.length === 0) return "";
  return (
    "\n\n## Actions Taken This Session\nThese are the ACTUAL outcomes of every action you performed. Report ONLY these results.\n" +
    entries.map((e) => `- ${e.summary}`).join("\n")
  );
}

function isAllowedUser(jid) {
  const number = jid?.split("@")[0];
  return allowedUsers.has(number);
}

async function loadAllowedUsers() {
  try {
    const setting = await apiGet("/setting?key=allowed_users");
    if (setting.value) {
      const numbers = setting.value.split(",").filter((n) => n.trim());
      numbers.forEach((n) => allowedUsers.add(n.trim()));
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
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp",
  "image/bmp": ".bmp",
  "image/gif": ".gif",
  "image/tiff": ".tiff",
  "video/mp4": ".mp4",
  "video/mkv": ".mkv",
  "video/avi": ".avi",
  "video/quicktime": ".mov",
  "video/webm": ".webm",
  "audio/mpeg": ".mp3",
  "audio/ogg": ".ogg",
  "audio/wav": ".wav",
  "audio/mp4": ".m4a",
  "audio/aac": ".aac",
  "application/pdf": ".pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
  "text/plain": ".txt",
  "text/csv": ".csv",
};

// --- System paths (from skills module) ---
const os = require("os");
const DEFAULT_SAVE_FOLDER = pathModule.join(DOWNLOADS, "Pinpoint");

console.log(`[Pinpoint] User home: ${USER_HOME}`);

// --- LLM setup (Gemini default, Ollama optional) ---
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || ""; // e.g. "qwen3.5:9b" — set to use local LLM instead of Gemini
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const OLLAMA_THINK = process.env.OLLAMA_THINK === "true"; // Enable thinking for Ollama (slower but smarter tool picks)
const USE_OLLAMA = !!OLLAMA_MODEL;

const ai = USE_OLLAMA ? null : new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const LLM_TAG = USE_OLLAMA ? "Ollama" : "Gemini";
if (USE_OLLAMA) console.log(`[Pinpoint] Using Ollama: ${OLLAMA_MODEL} at ${OLLAMA_URL}`);
else console.log(`[Pinpoint] Using Gemini: ${GEMINI_MODEL}`);

// Initialize LLM module with runtime config
llm.init({
  ai,
  OLLAMA_MODEL,
  OLLAMA_URL,
  OLLAMA_THINK,
  USE_OLLAMA,
  GEMINI_MODEL,
  sessionCosts,
  TOKEN_COST_INPUT,
  TOKEN_COST_OUTPUT,
});

// Build TOOL_ROUTES from extracted tools module
const TOOL_ROUTES = buildToolRoutes(MAX_RESULTS);

// (Tool declarations moved to src/tools.js — TOOL_DECLARATIONS array)
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
      do {
        d.setDate(d.getDate() + 1);
      } while (d.getDay() === 0 || d.getDay() === 6 || d.getTime() <= now);
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
    for (const r of res.reminders || []) {
      reminders.push({
        id: r.id,
        chatJid: r.chat_jid,
        message: r.message,
        triggerAt: new Date(r.trigger_at).getTime(),
        repeat: r.repeat || null,
        createdAt: new Date(r.created_at).getTime(),
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

// Tools whose results should be stored as refs when large
const REF_TOOLS = new Set([
  "list_files",
  "search_images_visual",
  "find_person",
  "find_person_by_face",
  "detect_faces",
  "count_faces",
  "ocr",
  "find_duplicates",
  "pdf_to_images",
  "search_documents",
]);

// Preview limits per tool (how many items to show Gemini)
const REF_PREVIEW = {
  list_files: 20,
  search_images_visual: 10,
  find_person: 5,
  find_person_by_face: 5,
  detect_faces: 10,
  count_faces: 10,
  ocr: 3,
  find_duplicates: 5,
  pdf_to_images: 10,
  search_documents: 5,
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
  // Don't delete — refs may be used multiple times (e.g. retry, follow-up messages)
  // Expiry timer handles cleanup
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

// GC for stale per-chat state (prevents unbounded growth over days/weeks)
setInterval(() => {
  const now = Date.now();
  const staleMs = 2 * 60 * 60 * 1000; // 2 hours
  for (const [jid, ts] of allowedSessions) {
    if (now - ts > staleMs) {
      allowedSessions.delete(jid);
      lastImage.delete(jid);
      activeRequests.delete(jid);
      delete sessionCosts[jid];
      delete actionLedger[jid];
      clearIntentCache(jid);
    }
  }
}, 15 * 60 * 1000); // Check every 15 min

// Create a preview summary for Gemini (array of items → first N + count + ref)
function makeRefPreview(toolName, result, refKey) {
  const limit = REF_PREVIEW[toolName] || 10;

  // Handle array results (list_files, find_person, pdf_to_images, etc.)
  if (Array.isArray(result)) {
    const total = result.length;
    const preview = result.slice(0, limit);
    return {
      _ref: refKey,
      total,
      showing: preview.length,
      preview,
      note: `${total} items stored. Use ${refKey} in subsequent tool calls to reference all items.`,
    };
  }

  // Handle multi-query search_images_visual: { "query1": { results: [...] }, "query2": { results: [...] } }
  // Show category summary + grouping file path so Gemini can read it and batch_move
  if (toolName === "search_images_visual") {
    const queryKeys = Object.keys(result).filter((k) => !k.startsWith("_") && result[k]?.results);
    if (queryKeys.length > 1) {
      const summary = {};
      let totalFiles = 0;
      for (const q of queryKeys) {
        const files = result[q].results || [];
        summary[q] = { count: files.length, top3: files.slice(0, 3).map((r) => r.filename || r.path) };
        totalFiles += files.length;
      }
      const groupFile = result._grouping_file || null;
      return {
        _ref: refKey,
        categories: summary,
        total_files: totalFiles,
        grouping_file: groupFile,
        note: groupFile
          ? `Grouped ${totalFiles} files into ${queryKeys.length} categories. Full mapping saved at ${groupFile}. Read that file to get all paths, then batch_move per category.`
          : `Grouped ${totalFiles} files into ${queryKeys.length} categories.`,
      };
    }
  }

  // Handle object with results/matches/files array
  for (const arrayKey of ["entries", "results", "matches", "files", "images", "groups", "pages", "faces"]) {
    if (result[arrayKey] && Array.isArray(result[arrayKey]) && result[arrayKey].length > 0) {
      const arr = result[arrayKey];
      const total = arr.length;
      const preview = arr.slice(0, limit);
      const rest = { ...result, [arrayKey]: preview };
      return {
        _ref: refKey,
        total_items: total,
        showing: preview.length,
        ...rest,
        note: `${total} ${arrayKey} stored. Use ${refKey} to reference all.`,
      };
    }
  }

  // Fallback: just mark it as ref'd
  return {
    _ref: refKey,
    note: `Large result stored. Use ${refKey} to reference it.`,
    summary: JSON.stringify(result).slice(0, 300) + "...",
  };
}

// Resolve a single @ref:N value to its stored data, extracting paths for sources/paths keys
function _resolveOneRef(refKey, argName) {
  const data = resolveRef(refKey);
  if (!data) {
    console.warn(`[TempStore] ${refKey} not found (expired)`);
    return null;
  }
  // If the stored data is already an array, use it directly
  if (Array.isArray(data)) {
    console.log(`[TempStore] Resolved ${refKey} → ${data.length} items (array)`);
    return data;
  }
  // If it's an object with a results/matches/files array, extract paths
  for (const arrayKey of ["entries", "results", "matches", "files", "images", "groups", "pages"]) {
    if (data[arrayKey] && Array.isArray(data[arrayKey])) {
      let items;
      if (argName === "sources" || argName === "paths") {
        items = data[arrayKey].map((item) => item.path || item.file || item);
      } else {
        items = data[arrayKey];
      }
      console.log(`[TempStore] Resolved ${refKey} → ${items.length} items (from .${arrayKey})`);
      return items;
    }
  }
  // No array found, return whole object
  console.log(`[TempStore] Resolved ${refKey} → object`);
  return data;
}

// Resolve @ref:N in tool args before execution
function resolveRefsInArgs(args) {
  if (!args) return args;
  const resolved = { ...args };
  for (const [key, value] of Object.entries(resolved)) {
    // Case 1: direct string ref — sources: "@ref:1"
    if (typeof value === "string" && value.startsWith("@ref:")) {
      const result = _resolveOneRef(value, key);
      if (result) resolved[key] = result;
    }
    // Case 2: array containing refs — sources: ["@ref:1"] or sources: ["@ref:1", "@ref:2"]
    // This is the common case: Gemini puts refs inside arrays for batch_move sources
    else if (Array.isArray(value)) {
      const expanded = [];
      for (const item of value) {
        if (typeof item === "string" && item.startsWith("@ref:")) {
          const result = _resolveOneRef(item, key);
          if (result) {
            // Flatten: if resolved to array, spread it into the parent array
            if (Array.isArray(result)) expanded.push(...result);
            else expanded.push(result);
          }
        } else {
          expanded.push(item);
        }
      }
      if (expanded.length !== value.length || expanded.some((v, i) => v !== value[i])) {
        resolved[key] = expanded;
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

const API_SECRET = process.env.API_SECRET || "";
const _apiHeaders = API_SECRET ? { "X-API-Secret": API_SECRET } : {};

async function apiGet(path) {
  const resp = await fetch(`${API_URL}${path}`, { headers: _apiHeaders });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiPost(path, body) {
  const resp = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ..._apiHeaders },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiDelete(path) {
  const resp = await fetch(`${API_URL}${path}`, { method: "DELETE", headers: _apiHeaders });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiPut(path, body) {
  const resp = await fetch(`${API_URL}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ..._apiHeaders },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function apiPing() {
  try {
    return (await fetch(`${API_URL}/ping`)).ok;
  } catch {
    return false;
  }
}

// --- Execute a tool call from Gemini ---
// (preValidate + TOOL_ROUTES moved to src/tools.js)
// TOOL_ROUTES was built above via buildToolRoutes(MAX_RESULTS)

// Legacy enc used in executeTool switch cases
const enc = encodeURIComponent;

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
    // --- Routing table dispatch (covers ~45 tools) ---
    const route = TOOL_ROUTES[name];
    if (route) {
      const path = typeof route.p === "function" ? route.p(args) : route.p;
      if (route.m === "GET") return await apiGet(path);
      return await apiPost(path, route.b ? route.b(args) : args);
    }

    // --- Custom handlers (tools with side effects or complex logic) ---
    switch (name) {
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
      case "search_images_visual": {
        if (!args.folder) return { error: "folder is required. Pass the absolute path to the image folder." };
        const queries = args.queries || (args.query ? [args.query] : []);
        if (queries.length <= 1) {
          return await apiPost("/search-images-visual", {
            folder: args.folder,
            query: queries[0] || "",
            limit: args.limit || 10,
          });
        }
        // Multi-query = grouping mode: classify ALL images, no limit
        const groupLimit = 9999;
        const firstResult = await apiPost("/search-images-visual", {
          folder: args.folder,
          query: queries[0],
          limit: groupLimit,
        });
        if (firstResult.status === "embedding") return firstResult;
        const restPromises = queries
          .slice(1)
          .map((q) => apiPost("/search-images-visual", { folder: args.folder, query: q, limit: groupLimit }));
        const restResults = await Promise.all(restPromises);
        const allResults = { [queries[0]]: firstResult };
        queries.slice(1).forEach((q, i) => {
          allResults[q] = restResults[i];
        });

        // Deduplicate: each image → best matching category only
        const bestCategory = {};
        for (const [q, r] of Object.entries(allResults)) {
          if (!r?.results) continue;
          for (const item of r.results) {
            const p = item.path || item.filename;
            const score = item.match_pct || 0;
            if (!bestCategory[p] || score > bestCategory[p].score) {
              bestCategory[p] = { query: q, score };
            }
          }
        }
        const catMap = {};
        for (const [p, { query }] of Object.entries(bestCategory)) {
          if (!catMap[query]) catMap[query] = [];
          catMap[query].push(p);
        }
        const totalMapped = Object.values(catMap).reduce((sum, arr) => sum + arr.length, 0);

        if (totalMapped > 0) {
          const tmpDir = pathModule.join(os.tmpdir(), "pinpoint");
          if (!existsSync(tmpDir)) mkdirSync(tmpDir, { recursive: true });
          const mapFile = pathModule.join(tmpDir, `visual_group_${Date.now()}.json`);
          writeFileSync(mapFile, JSON.stringify(catMap, null, 2));

          const summary = {};
          for (const [q, files] of Object.entries(catMap)) {
            const refKey = storeRef(files);
            summary[q] = {
              count: files.length,
              sources_ref: refKey,
              top3: files.slice(0, 3).map((f) => pathModule.basename(f)),
            };
          }
          const catList = Object.entries(summary)
            .map(([q, s]) => `${q}: ${s.count} files (${s.sources_ref})`)
            .join(", ");
          console.log(
            `[Visual] Classified ${totalMapped} images into ${Object.keys(catMap).length} categories → ${mapFile}`,
          );
          return {
            total_classified: totalMapped,
            categories: summary,
            _grouping_file: mapFile,
            _hint: `Classified ALL ${totalMapped} images into ${Object.keys(catMap).length} categories: ${catList}. To move: call batch_move({ sources: "<sources_ref>", destination: "<folder>" }) for each category. The sources_ref values are ready to use.`,
          };
        }
        return allResults;
      }
      case "web_search": {
        const q = enc(args.query);
        const webUrl = args.url || `https://search.brave.com/search?q=${q}`;
        return await apiGet(`/web-read?url=${enc(webUrl)}${args.start ? `&start=${args.start}` : ""}`);
      }
      case "memory_save": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        const res = await apiPost("/memory", { fact: args.fact, category: args.category || "general" });
        try {
          const ctx = await apiGet("/memory/context");
          memoryContext = ctx.text || "";
        } catch (_) {}
        return res;
      }
      case "memory_search": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        return await apiGet(`/memory/search?q=${enc(args.query)}&limit=10`);
      }
      case "memory_delete": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        const res = await apiDelete(`/memory/${args.id}`);
        try {
          const ctx = await apiGet("/memory/context");
          memoryContext = ctx.text || "";
        } catch (_) {}
        return res;
      }
      case "memory_forget": {
        if (!memoryEnabled) return { error: "Memory is disabled. User can enable it with /memory on." };
        const res = await apiPost("/memory/forget", { description: args.description });
        if (res.success) {
          try {
            const ctx = await apiGet("/memory/context");
            memoryContext = ctx.text || "";
          } catch (_) {}
        }
        return res;
      }
      case "set_reminder": {
        const triggerAt = parseReminderTime(args.time);
        if (!triggerAt)
          return {
            error: `Could not parse time: "${args.time}". Use format like "5pm", "in 2 hours", or "2026-02-27T17:00:00".`,
          };
        if (triggerAt.getTime() <= Date.now()) return { error: "That time is in the past." };
        const repeat = args.repeat || null;
        if (repeat && !["daily", "weekly", "monthly", "weekdays"].includes(repeat)) {
          return { error: `Invalid repeat: "${repeat}". Use: daily, weekly, monthly, or weekdays.` };
        }
        const tz = USER_TZ;
        const saved = await apiPost("/reminders", {
          chat_jid: chatJid,
          message: args.message,
          trigger_at: triggerAt.toISOString(),
          repeat,
        });
        reminders.push({
          id: saved.id,
          chatJid,
          message: args.message,
          triggerAt: triggerAt.getTime(),
          repeat,
          createdAt: Date.now(),
        });
        const result = {
          success: true,
          id: saved.id,
          message: args.message,
          trigger_at: triggerAt.toLocaleString("en-IN", {
            timeZone: tz,
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
            day: "numeric",
            month: "short",
          }),
        };
        if (repeat) result.repeat = repeat;
        return result;
      }
      case "list_reminders": {
        const tz = USER_TZ;
        const pending = reminders.filter((r) => r.triggerAt > Date.now());
        if (pending.length === 0) return { count: 0, reminders: [], note: "No pending reminders." };
        return {
          count: pending.length,
          reminders: pending.map((r) => ({
            id: r.id,
            message: r.message,
            repeat: r.repeat || null,
            trigger_at: new Date(r.triggerAt).toLocaleString("en-IN", {
              timeZone: tz,
              hour: "2-digit",
              minute: "2-digit",
              hour12: true,
              day: "numeric",
              month: "short",
            }),
          })),
        };
      }
      case "cancel_reminder": {
        const idx = reminders.findIndex((r) => r.id === args.id);
        if (idx === -1) return { error: `Reminder #${args.id} not found.` };
        const removed = reminders.splice(idx, 1)[0];
        try {
          await apiDelete(`/reminders/${removed.id}`);
        } catch (_) {}
        return { success: true, cancelled: removed.message };
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
    const data = await apiGet(
      `/conversation/history?session_id=${encodeURIComponent(sessionId)}&limit=${MAX_HISTORY_MESSAGES}`,
    );
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
      try {
        unlinkSync(pathModule.join(TEMP_MEDIA_DIR, f));
      } catch (_) {}
    }
    if (files.length > 0) console.log(`[Pinpoint] Cleaned up ${files.length} temp media file(s)`);
  } catch (_) {}
}

async function resetSession(sessionId) {
  try {
    const data = await apiPost("/conversation/reset", { session_id: sessionId });
    cleanupTempMedia();
    lastImage.delete(sessionId);
    activeRequests.delete(sessionId);
    delete sessionCosts[sessionId];
    delete actionLedger[sessionId];
    clearIntentCache(sessionId);
    return data.deleted_count || 0;
  } catch {
    return 0;
  }
}

function isSessionIdle(updatedAt) {
  if (!updatedAt) return false;
  const lastActivity = new Date(updatedAt + "Z").getTime(); // ISO 8601 UTC
  return Date.now() - lastActivity > IDLE_TIMEOUT_MS;
}

// --- Context compaction: summarize old messages instead of dropping them ---
const COMPACT_THRESHOLD = 40; // Compact when total DB messages exceed this
const COMPACT_KEEP = 15; // Keep this many recent messages after compacting
const CONTENTS_COMPACT_THRESHOLD = 24; // Compact in-memory contents when entries exceed this (≈12 tool rounds, matches MAX_ROUNDS)

async function compactHistory(sessionId) {
  try {
    // Load ALL messages (up to 100)
    const data = await apiGet(`/conversation/history?session_id=${encodeURIComponent(sessionId)}&limit=100`);
    const msgs = data.messages || [];
    if (msgs.length < COMPACT_THRESHOLD) return; // Not enough to compact

    // Split: old messages to summarize, recent to keep
    const toSummarize = msgs.slice(0, msgs.length - COMPACT_KEEP);
    const toKeep = msgs.slice(msgs.length - COMPACT_KEEP);

    // Build a structured summary via Gemini (adapted from Claude Code compaction prompt)
    let summaryInput = `Summarize this conversation into a structured snapshot that preserves all context needed to continue the task.

Format:
USER REQUEST: What the user explicitly asked for (1-2 lines, include exact quotes if important)
KEY FACTS: File paths, search results, data discovered (bullet points)
ACTIONS TAKEN: What tools were called and what they did (bullet points)
CURRENT TASK: What is being worked on RIGHT NOW (1 line — critical for follow-up messages like "okay go ahead")
PENDING: What still needs to be done (bullet points)

Conversation:
`;
    for (const m of toSummarize) {
      summaryInput += `${m.role}: ${m.content.slice(0, 300)}\n`;
    }

    const response = await llm.llmGenerate({
      model: GEMINI_MODEL,
      contents: [{ role: "user", parts: [{ text: summaryInput }] }],
    });
    llm.trackTokens(sessionId, response);
    const summary = response.text || "Previous conversation context unavailable.";

    // DB-only reset: clear messages but DON'T clear in-memory state (actionLedger, sessionCosts, etc.)
    // because runGemini() may still be active for this chat
    try { await apiPost("/conversation/reset", { session_id: sessionId }); } catch {}
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

  // Build a text summary of old turns — strip tool results to bare minimum (saves tokens on summary call)
  // Claude Code pattern: clear old tool results BEFORE summarization
  let summaryParts = [];
  for (const entry of toSummarize) {
    if (!entry?.parts) continue;
    for (const part of entry.parts) {
      if (part.text) {
        summaryParts.push(`${entry.role}: ${part.text.slice(0, 300)}`);
      } else if (part.functionCall) {
        summaryParts.push(
          `tool: ${part.functionCall.name}(${JSON.stringify(part.functionCall.args || {}).slice(0, 100)})`,
        );
      } else if (part.functionResponse) {
        const r = part.functionResponse.response?.result;
        const toolName = part.functionResponse.name;
        let status;
        if (r?.error) {
          status = `error: ${String(r.error).slice(0, 60)}`;
        } else if (MUTATING_TOOLS.has(toolName)) {
          // Preserve action outcomes through compaction (OpenClaw: tool failures survive compaction)
          status = summarizeToolResult(toolName, null, r) || "ok";
        } else {
          status = "ok";
        }
        summaryParts.push(`result: ${toolName} → ${status}`);
      }
    }
  }

  // Inject action ledger into compaction input (OpenClaw: action outcomes survive compaction)
  const ledgerForCompaction = chatJid ? getActionLedgerText(chatJid) : "";
  const summaryInput = `Summarize this tool-calling session. Preserve all context needed to continue the current task.

Format:
USER REQUEST: What the user asked for (1 line, quote key phrases)
RESULTS: Key findings from tools — file paths, counts, search results (bullet points)
ACTIONS TAKEN: What mutating actions were performed and their EXACT outcomes (moved_count, errors, etc.)
CURRENT TASK: What is being worked on RIGHT NOW (1 line — critical)
REMAINING: What still needs to be done to answer the user (1 line)

Conversation:
${summaryParts.join("\n")}${ledgerForCompaction}`;

  try {
    const response = await llm.llmGenerate({
      model: GEMINI_MODEL,
      contents: [{ role: "user", parts: [{ text: summaryInput }] }],
    });
    llm.trackTokens(chatJid, response);
    const summary = response.text || "Previous context unavailable.";

    // Replace contents array in-place: summary + kept entries
    contents.length = 0;
    contents.push({
      role: "user",
      parts: [
        {
          text: `[Previous conversation context]\n${summary}\n\nContinue from where you left off. Do not ask the user to repeat — use the context above.`,
        },
      ],
    });
    contents.push({ role: "model", parts: [{ text: "I have the full context. Continuing with the current task." }] });
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

// (summarizeToolResult moved to src/tools.js)

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
    clearIntentCache(chatJid);
    delete actionLedger[chatJid];
    if (deleted > 0)
      console.log(`[Memory] Auto-reset session (idle ${IDLE_TIMEOUT_MS / 60000} min), cleared ${deleted} messages`);
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
    systemInstruction: getSystemPrompt(userMessage, chatJid, {
      memoryEnabled,
      memoryContext,
      actionLedgerText: getActionLedgerText(chatJid),
    }),
    thinkingConfig: { thinkingLevel: "low" },
    mediaResolution: "MEDIA_RESOLUTION_LOW",
  };
  let activeTools = null;
  if (!opts.noTools) {
    // getToolsForIntent handles follow-up intent carry-forward via lastIntentCats
    activeTools = getToolsForIntent(userMessage, chatJid);
  }

  const toolCache = new Map(); // Dedup: cache tool results within this turn
  const toolLog = []; // Track tool calls for conversation context
  let inlineImageCount = 0; // Track images sent as visual data
  let notifiedUser = false; // Track if we sent "working on it" message
  const toolStartTime = Date.now(); // For elapsed time logging

  // Snapshot token counts at start of this message (for per-message budget)
  const sc0 = sessionCosts[chatJid];
  const msgStartTokens = { input: sc0?.input || 0, output: sc0?.output || 0 };

  // Loop detection: track consecutive identical tool calls (from Gemini CLI + OpenCode)
  const LOOP_THRESHOLD = 3; // Same exact call N times → stop (OpenCode uses 3)
  const MAX_ROUNDS = 12; // Max rounds per prompt (Claude Code uses 1-20 depending on task)
  let lastCallHash = null;
  let lastCallCount = 0;
  let didTokenCompact = false; // Only compact once per runGemini call

  for (let round = 0; round < MAX_ROUNDS; round++) {
    // Check if user sent "stop" / "cancel" (lock was released)
    if (!activeRequests.has(chatJid)) {
      console.log(`[${LLM_TAG}] Stopped by user after ${round} rounds`);
      return { text: "Request stopped.", toolLog };
    }

    // Cost-based circuit breaker: per-message budget (not cumulative session)
    const MESSAGE_BUDGET_USD = 0.1;
    const sc = sessionCosts[chatJid];
    if (sc && round > 0) {
      const msgCost =
        (sc.input - msgStartTokens.input) * TOKEN_COST_INPUT + (sc.output - msgStartTokens.output) * TOKEN_COST_OUTPUT;
      if (msgCost >= MESSAGE_BUDGET_USD) {
        console.warn(
          `[${LLM_TAG}] Budget exceeded ($${msgCost.toFixed(4)} >= $${MESSAGE_BUDGET_USD}) after ${round} rounds`,
        );
        return {
          text: `I've used my token budget for this request. Here's what I found so far — let me know if you need more.`,
          toolLog,
        };
      }
    }

    // Microcompact: clear old tool results, keep last N (Claude Code pattern)
    // Never clear: expensive search results + mutating action results (accountability — OpenClaw pattern)
    if (round > 0) {
      const KEEP_LAST = 5;
      const PRESERVE_TOOLS = new Set(["search_images_visual", "search_documents", "detect_faces"]);
      let toolResultCount = 0;
      for (let i = contents.length - 1; i >= 0; i--) {
        const entry = contents[i];
        if (!entry?.parts) continue;
        for (const part of entry.parts) {
          if (part.functionResponse?.response?.result) toolResultCount++;
        }
      }
      if (toolResultCount > KEEP_LAST) {
        let clearCount = toolResultCount - KEEP_LAST;
        for (let i = 0; i < contents.length && clearCount > 0; i++) {
          const entry = contents[i];
          if (!entry?.parts) continue;
          for (const part of entry.parts) {
            if (part.functionResponse?.response?.result && clearCount > 0) {
              const toolName = part.functionResponse?.name;
              // Never clear expensive search/visual results
              if (PRESERVE_TOOLS.has(toolName)) continue;
              // Never clear mutating tool results — accountability (OpenClaw: action outcomes persist)
              if (MUTATING_TOOLS.has(toolName)) continue;
              part.functionResponse.response.result = "[Result cleared]";
              clearCount--;
            }
          }
        }
      }
    }

    const response = await llm.llmGenerate({
      model: GEMINI_MODEL,
      contents,
      config,
      tools: activeTools,
    });

    // Track token usage
    const roundTokens = llm.trackTokens(chatJid, response);

    // Mid-call compaction: if contents array is getting large (many tool rounds), summarize old turns
    // Prevents token burn from long tool chains (e.g. 24 rounds searching for a file)
    if (!didTokenCompact && contents.length > CONTENTS_COMPACT_THRESHOLD) {
      console.log(
        `[Memory] Contents ${contents.length} entries > ${CONTENTS_COMPACT_THRESHOLD} threshold — compacting...`,
      );
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
        } catch (e) {
          /* ignore send errors */
        }
      }

      // Add model's response to conversation
      contents.push(response.candidates[0].content);

      // Execute each tool call
      const elapsed = Math.round((Date.now() - toolStartTime) / 1000);
      const thinkInfo = roundTokens?.thinking ? `, ${llm.formatTokens(roundTokens.thinking)} think` : "";
      const tokenInfo = roundTokens
        ? `, ${llm.formatTokens(roundTokens.input)} in / ${llm.formatTokens(roundTokens.output)} out${thinkInfo}`
        : "";
      console.log(
        `[${LLM_TAG}] Round ${round + 1}, ${response.functionCalls.length} tool(s), ${elapsed}s elapsed${tokenInfo}`,
      );
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
                response: {
                  result: {
                    error: `Loop detected: you've called ${fc.name} ${lastCallCount} times with identical args. Stop retrying and answer with what you have — if the data wasn't found, tell the user.`,
                  },
                },
              },
            });
            continue;
          }
        } else {
          lastCallHash = callHash;
          lastCallCount = 1;
        }

        // Dedup: skip if same tool+args already called, or same-folder list_files
        let result;
        // Smart dedup: list_files on same folder = same result (sort/limit/recursive don't matter)
        const folderKey = fc.name === "list_files" && fc.args?.folder ? `list_files:${fc.args.folder}` : null;
        if (toolCache.has(callHash)) {
          result = toolCache.get(callHash);
          console.log(`[${LLM_TAG}] Dedup skip: ${fc.name} (cached)`);
        } else if (folderKey && toolCache.has(folderKey)) {
          result = toolCache.get(folderKey);
          console.log(`[${LLM_TAG}] Dedup skip: ${fc.name} (same folder already listed)`);
        } else {
          result = await executeTool(fc, sock, chatJid);
          toolCache.set(callHash, result);
          if (folderKey) toolCache.set(folderKey, result); // Smart dedup for same-folder
        }

        // Post-tool error guidance: help Gemini recover instead of retrying blindly
        if (result?.error && !result._hint) {
          const err = String(result.error).toLowerCase();
          if (err.includes("not found") || err.includes("no such") || err.includes("does not exist")) {
            result._hint = "Path not found. Try list_files on the parent folder to check what exists.";
          } else if (err.includes("permission") || err.includes("access denied") || err.includes("read-only")) {
            result._hint = "Permission denied. File may be in use or read-only.";
          } else if (err.includes("no images") || err.includes("no results") || err.includes("no match")) {
            result._hint = "No matches. Try broader search terms or a different folder.";
          }
        }

        // Action Ledger: record every mutating tool's REAL outcome (OpenClaw pattern)
        // This gets injected into every subsequent LLM call as "## Actions Taken"
        if (MUTATING_TOOLS.has(fc.name)) {
          toolCache.clear(); // Invalidate cached results — filesystem may have changed
          recordAction(chatJid, fc.name, fc.args, result);
          // Independent result verification (isToolResultError — OpenClaw 4-layer check)
          // Detect "success but nothing done" — the root cause of Gemini lying
          if (!result?.error) {
            if (fc.name === "batch_move" && (result?.moved_count ?? 0) === 0) {
              result._warning = `⚠️ 0 files were actually ${result?.action || "moved"}. ${result?.skipped_count || 0} skipped, ${result?.error_count || 0} errors. Tell the user NO files were moved.`;
              console.log(
                `[Trust] batch_move: 0 files moved (${result?.skipped_count || 0} skipped) → ${fc.args?.destination}`,
              );
            } else if (fc.name === "batch_rename" && (result?.renamed_count ?? result?.renamed ?? 0) === 0) {
              result._warning = `⚠️ 0 files were actually renamed. Tell the user no files were renamed.`;
            }
          }
        }

        // Tail tool calls: auto-send generated files to user without LLM round-trip
        // (extract_frame, crop_face, generate_chart → auto send_file)
        if (!result?.error) {
          const autoSendPath =
            (fc.name === "extract_frame" && result.path) ||
            (fc.name === "crop_face" && result.path) ||
            (fc.name === "generate_chart" && result.path);
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
            const isMultiQuery =
              fc.name === "search_images_visual" &&
              fc.args?.queries &&
              fc.args.queries.length > 1 &&
              typeof result === "object" &&
              !Array.isArray(result);
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

        // Track tool calls + results for conversation memory context
        const summary = summarizeToolResult(fc.name, fc.args, result);
        toolLog.push(summary || `${fc.name}: done`);

        // If read_file returned an image, include as inlineData so Gemini SEES it (capped)
        if (fc.name === "read_file" && result.type === "image" && result.data && inlineImageCount < MAX_INLINE_IMAGES) {
          inlineImageCount++;
          functionResponses.push({
            functionResponse: {
              name: fc.name,
              response: {
                result: {
                  type: "image",
                  path: result.path,
                  size: result.size,
                  note: "Image included as visual data — analyze it directly.",
                },
              },
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
              response: {
                result: {
                  type: "image",
                  path: result.path,
                  size: result.size,
                  note: `Image limit reached (${MAX_INLINE_IMAGES}). Use detect_faces or run_python for batch image analysis.`,
                },
              },
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

      // Inject tool result summaries (Claude Code pattern: model trusts summaries, doesn't re-verify raw data)
      const summaries = response.functionCalls
        .map((fc) => {
          const r = toolCache.get(fc.name + ":" + JSON.stringify(fc.args || {}));
          return summarizeToolResult(fc.name, fc.args, r);
        })
        .filter(Boolean);
      if (summaries.length > 0) {
        parts.push({ text: `[Tool summaries: ${summaries.join(". ")}]` });
      }

      // Loop hard-break: if loop was detected, force model to answer next round
      if (lastCallCount >= LOOP_THRESHOLD) {
        parts.push({
          text: "[System: LOOP DETECTED. You've called the same tool multiple times with identical args. STOP calling tools. Answer with what you have NOW.]",
        });
        // Send the error results back, then set round to max-1 so next iteration exits
        contents.push({ role: "user", parts });
        round = MAX_ROUNDS - 2; // Next round will be the last — forces text response
        continue;
      }

      // Round-based efficiency nudges (adapted from Claude Code patterns)
      if (round === 3) {
        parts.push({
          text: "[System: Go straight to the point. Try the simplest approach first without going in circles. Do not overdo it. If you have search results, answer now.]",
        });
      } else if (round === 6) {
        parts.push({
          text: "[System: You've used 7 rounds. Do what was asked, nothing more. Answer with what you have NOW. Do not call more tools.]",
        });
      } else if (round === 9) {
        parts.push({ text: "[System: 10 rounds used. STOP calling tools. Give your answer immediately.]" });
      }

      contents.push({ role: "user", parts });
    } else {
      // No more tool calls — return the text response
      const text = response.text;
      if (!text) {
        const finishReason = response.candidates?.[0]?.finishReason;
        const safetyRatings = response.candidates?.[0]?.safetyRatings;
        console.error(
          `[${LLM_TAG}] Empty response. finishReason=${finishReason}, safety=${JSON.stringify(safetyRatings || [])}`,
        );
        // MALFORMED_FUNCTION_CALL in no-tools mode: Gemini tried to call a tool
        // Only re-enable tools if this was a text message (not a bare image/file)
        if (finishReason === "MALFORMED_FUNCTION_CALL" && opts.noTools && round === 0 && !opts.inlineImage) {
          console.log(`[${LLM_TAG}] MALFORMED_FUNCTION_CALL in no-tools mode — retrying with tools`);
          opts.noTools = false;
          activeTools = getToolsForIntent(userMessage, chatJid);
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
  return {
    text: `I've completed ${MAX_ROUNDS} rounds of work. Here's what I did so far — let me know if you need me to continue.`,
    toolLog,
  };
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
    return {
      text: 'Hello! Smart search is temporarily unavailable (rate limit). Try a search query like "find reliance invoice".',
      files: [],
    };
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

  const files = data.results.slice(0, MAX_FILES_TO_SEND).map((r) => ({
    path: r.path,
    title: r.title,
    score: r.score,
    file_type: r.file_type,
  }));

  return { text: msg, files };
}

// --- Text chunking ---
function chunkText(text, limit = TEXT_CHUNK_LIMIT) {
  if (text.length <= limit) return [text];
  const chunks = [];
  let remaining = text;
  while (remaining.length > 0) {
    if (remaining.length <= limit) {
      chunks.push(remaining);
      break;
    }
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
  let fileSize = statSync(filePath).size;
  const isImage = IMAGE_EXTENSIONS.has(ext);

  // Auto-resize large images to fit WhatsApp limit
  if (isImage && fileSize > MAX_IMAGE_SIZE) {
    try {
      const sharp = (await import("sharp")).default;
      const resizedPath = pathModule.join("/tmp", `pinpoint_send_${Date.now()}.jpg`);
      await sharp(filePath)
        .resize({ width: 2048, height: 2048, fit: "inside" })
        .jpeg({ quality: 80 })
        .toFile(resizedPath);
      filePath = resizedPath;
      fileSize = statSync(filePath).size;
      console.log(`[Pinpoint] Auto-resized large image for sending (${(fileSize / 1024 / 1024).toFixed(1)}MB)`);
    } catch (e) {
      console.warn(`[Pinpoint] Auto-resize failed: ${e.message}`);
      return false;
    }
  }
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
    const toolCount = TOOL_DECLARATIONS.length;
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
    credsSaveQueue = credsSaveQueue
      .then(async () => {
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
      })
      .catch((err) => {
        console.error("[Pinpoint] Creds save error:", err.message);
      });
  };

  // Clear stale reminder interval before creating new socket (prevents stale sock reference)
  if (global._reminderInterval) {
    clearInterval(global._reminderInterval);
    global._reminderInterval = null;
  }

  const sock = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    version,
    logger,
    printQRInTerminal: false,
    browser: ["Pinpoint", "CLI", "1.0"],
    syncFullHistory: false,
    markOnlineOnConnect: false,
  });
  currentSock = sock; // Update module-level reference for reminders

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
      loadReminders().catch((e) => console.error("[Reminder] Load failed:", e.message));

      // Start reminder checker (every 30 seconds)
      if (!global._reminderInterval) {
        global._reminderInterval = setInterval(async () => {
          if (!currentSock) return;
          const now = Date.now();
          const due = reminders.filter((r) => r.triggerAt <= now && !r._firing);
          for (const r of due) r._firing = true; // Mark to prevent double-fire
          for (const r of due) {
            try {
              const label = r.repeat ? `⏰ *Reminder (${r.repeat}):* ${r.message}` : `⏰ *Reminder:* ${r.message}`;
              await currentSock.sendMessage(r.chatJid, { text: label });
              console.log(`[Reminder] Sent: "${r.message}" to ${r.chatJid}${r.repeat ? ` (${r.repeat})` : ""}`);
            } catch (e) {
              console.error(`[Reminder] Failed to send: ${e.message}`);
            }
          }
          // Handle sent reminders: reschedule recurring, remove one-time
          for (const r of due) {
            delete r._firing;
            const idx = reminders.indexOf(r);
            if (idx === -1) continue;
            if (r.repeat) {
              // Reschedule to next occurrence
              const next = getNextOccurrence(r.triggerAt, r.repeat);
              if (next) {
                r.triggerAt = next.getTime();
                try {
                  await apiPut(`/reminders/${r.id}?trigger_at=${encodeURIComponent(next.toISOString())}`, {});
                } catch (_) {}
                console.log(`[Reminder] Rescheduled "${r.message}" → ${next.toISOString()}`);
              } else {
                console.log(`[Reminder] Cannot reschedule "${r.message}" (unknown repeat: ${r.repeat}) — removing`);
                reminders.splice(idx, 1);
                try {
                  await apiDelete(`/reminders/${r.id}`);
                } catch (_) {}
              }
            } else {
              reminders.splice(idx, 1);
              try {
                await apiDelete(`/reminders/${r.id}`);
              } catch (_) {}
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
        console.log(
          `[Pinpoint] Reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${reconnectAttempt}/${RECONNECT.maxAttempts})`,
        );
        setTimeout(startBot, delay);
      }
    }
  });

  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify" && type !== "append") return;
    for (const msg of messages) {
      try {
        await handleMessage(sock, msg);
      } catch (err) {
        console.error("[Pinpoint] Error:", err.message);
      }
    }
  });

  return sock;
}

// --- Handle received media files (save to computer) ---

function generateFilename(mediaType, mimetype) {
  const now = new Date();
  const ts =
    now.getFullYear().toString() +
    String(now.getMonth() + 1).padStart(2, "0") +
    String(now.getDate()).padStart(2, "0") +
    "_" +
    String(now.getHours()).padStart(2, "0") +
    String(now.getMinutes()).padStart(2, "0") +
    String(now.getSeconds()).padStart(2, "0");
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
  const isAudioMsg = msgType === "audioMessage";
  const isProcessingOnly = !customFolder && (hasCaptionText || isImageMsg || isAudioMsg);
  const saveFolder = customFolder || (isProcessingOnly ? TEMP_MEDIA_DIR : DEFAULT_SAVE_FOLDER);

  // Create folder if needed
  mkdirSync(saveFolder, { recursive: true });

  // Download the file
  let buffer;
  try {
    buffer = await downloadMediaMessage(
      msg,
      "buffer",
      {},
      {
        logger,
        reuploadRequest: sock.updateMediaMessage,
      },
    );
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

  // For images/audio: send inline to Gemini (it sees/hears it directly)
  // For non-media with caption: send file path + caption to Gemini
  const isAudio = msgType === "audioMessage";
  const shouldProcess = hasCaption || isImage || isAudio;

  if (shouldProcess && !activeRequests.has(chatJid)) {
    let userMsg;
    if (isAudio) {
      userMsg = hasCaption
        ? `[Voice note at ${savePath}]\n${cleanCaption}`
        : `[Voice note at ${savePath}]\nUser sent a voice message. Listen and respond to what they said.`;
    } else if (hasCaption) {
      userMsg = `[File: ${pathModule.basename(savePath)} at ${savePath}]\n${cleanCaption}`;
    } else {
      userMsg = `[Photo: ${pathModule.basename(savePath)} at ${savePath}]\nUser sent this with no instruction. Just ask what they want to do with it.`;
    }
    const myRequestId = ++requestCounter;
    activeRequests.set(chatJid, { msg: hasCaption ? cleanCaption : isAudio ? "[voice]" : "[photo]", startTime: Date.now(), id: myRequestId });
    try {
      await sock.sendPresenceUpdate("composing", chatJid);

      // If image or audio, read as base64 and send inline — Gemini sees/hears it directly
      let inlineImage = null;
      if (isImage || isAudio) {
        try {
          const mediaData = readFileSync(savePath).toString("base64");
          inlineImage = { mimeType: mimetype, data: mediaData };
          if (isImage) {
            lastImage.set(chatJid, { mimeType: mimetype, data: mediaData, path: savePath, ts: Date.now() });
          }
          console.log(`[Pinpoint] Sending ${isAudio ? "audio" : "image"} inline to ${LLM_TAG} (${sizeStr})`);
        } catch (e) {
          console.error(`[Pinpoint] Failed to read ${isAudio ? "audio" : "image"} for inline:`, e.message);
        }
      }

      // No caption + image → disable tools (Gemini just describes, no fishing expedition)
      // Audio always gets tools (user might ask to do something)
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
        await saveMessage(
          chatJid,
          "user",
          hasCaption ? cleanCaption : `[Sent photo: ${pathModule.basename(savePath)}]`,
        );
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
  const isSelfChat =
    !isGroup && ((myNumber && chatNumber && myNumber === chatNumber) || (myLid && chatNumber && myLid === chatNumber));

  // Layer 0: Skip our own messages in non-self chats (prevents infinite echo loops)
  if (key.fromMe && !isSelfChat) return;

  if (!isSelfChat && !isAllowedUser(chatJid)) {
    if (!isGroup)
      console.log(
        `[Pinpoint] Ignored message from: ${chatJid} (not allowed). To allow: /allow ${chatJid.split("@")[0]}`,
      );
    return;
  }

  // Allowed users: "pinpoint" starts session, "bye/stop pinpoint" ends it, 60min idle timeout
  const isAllowed = !isSelfChat && isAllowedUser(chatJid);
  if (isAllowed) {
    const peekText = (
      msg.message?.conversation ||
      msg.message?.extendedTextMessage?.text ||
      msg.message?.imageMessage?.caption ||
      msg.message?.documentMessage?.caption ||
      ""
    ).toLowerCase();
    const hasSession = allowedSessions.has(chatJid) && Date.now() - allowedSessions.get(chatJid) < IDLE_TIMEOUT_MS;

    // End session: "bye pinpoint" / "by pinpoint" / "stop pinpoint" / "exit pinpoint" / "pinpoint bye/stop"
    const isEndCmd =
      /\b(bye|by|stop|exit|quit|close)\b.*\bpinpoint\b|\bpinpoint\b.*\b(bye|by|stop|exit|quit|close)\b/.test(peekText);
    if (hasSession && isEndCmd) {
      allowedSessions.delete(chatJid);
      lastImage.delete(chatJid);
      activeRequests.delete(chatJid);
      clearIntentCache(chatJid);
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
  try {
    await sock.readMessages([key]);
  } catch (_) {}

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
    const number = userMsg
      .slice(7)
      .trim()
      .replace(/[^0-9]/g, "");
    if (!number || number.length < 10) {
      await sock.sendMessage(chatJid, {
        text: `${PREFIX} Invalid number. Use: /allow 919876543210 (include country code)`,
      });
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
    const number = userMsg
      .slice(8)
      .trim()
      .replace(/[^0-9]/g, "");
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
    clearIntentCache(chatJid);
    delete actionLedger[chatJid];
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
    try {
      await apiPost(`/setting?key=memory_enabled&value=${on}`, {});
    } catch (_) {}
    if (on) {
      try {
        const ctx = await apiGet("/memory/context");
        memoryContext = ctx.text || "";
      } catch (_) {}
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
    const reply = `${PREFIX} ${llm.getCostSummary(chatJid)}`;
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
        // Simple messages: only skip tools for greetings/reactions with NO active conversation
        // Confirmations ("yes", "ok", "go ahead") WITH recent context are NOT simple — they need tools
        // Claude Code pattern: never strip tools when there's active context
        const isGreeting = /^(hi|hello|hey|good morning|good night|gm|gn|bye)\s*[.!?]*$/i.test(userMsg.trim());
        const isReaction =
          /^(thanks|thank you|lol|haha|cool|nice|great|awesome|amazing|wow|perfect|oh|damn|omg|hehe|bruh|whoa|dope|sick|sweet|beautiful|wonderful|brilliant|excellent|fantastic|superb|impressive|neat|solid|lit|fire|legit|bet|word|ooh|aah|yay|woah|geez|ty|thx|np|gg|kk|ikr|imo|fyi|asap|🔥|💯|👏|😍|🤩|🥳|💪|🎉|✅|🙌|👌|😭|🤣|😎|💀|🫡|👍|🙏|❤️|😂|😊)\s*[.!?]*$/i.test(
            userMsg.trim(),
          );
        const hasContext = hasActiveIntent(chatJid);
        const isSimple = isGreeting || (isReaction && !hasContext);
        const noTools = isSimple;
        // Re-inject last image only if recent (< 2 min TTL) — prevents stale image re-injection
        const prevImg = lastImage.get(chatJid);
        const imgAge = prevImg ? Date.now() - (prevImg.ts || 0) : Infinity;
        const inlineImage = prevImg && imgAge < 120000 ? prevImg : null;
        // Prepend image path so Gemini knows where it is (for tools like crop_image)
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
    const tokenSummary = sc ? `, ${llm.formatTokens(sc.input + sc.output)} tokens` : "";
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
