const { readFileSync, readdirSync, existsSync } = require("fs");
const pathModule = require("path");
const os = require("os");
const { INTENT_KEYWORDS, SKILL_CATEGORIES } = require("./tools");

const USER_DATA_DIR = process.env.PINPOINT_USER_DIR || pathModule.join(os.homedir(), ".pinpoint");

// --- System paths (WSL-aware) ---
const HOME_DIR = os.homedir();
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

// --- Load skills from skills/*.md at startup (hierarchical: general + task-specific) ---
const SKILLS_DIR = process.env.PINPOINT_SKILLS_DIR || pathModule.join(__dirname, "..", "..", "skills");

// General skills: always injected (core rules, batch awareness, common mistakes)
const GENERAL_SKILL_FILES = ["batch-awareness.md", "common-mistakes.md", "core-rules.md"];

const _skillCache = {}; // filename → content
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
  const allFiles = readdirSync(SKILLS_DIR)
    .filter((f) => f.endsWith(".md"))
    .sort();
  for (const file of allFiles) _loadSkill(file);
  console.log(`[Pinpoint] Loaded ${allFiles.length} skills: ${allFiles.map((f) => f.replace(".md", "")).join(", ")}`);
} catch (err) {
  console.log("[Pinpoint] No skills loaded:", err.message);
}

// Build general skills content (always included)
const generalSkillsContent = GENERAL_SKILL_FILES.map((f) => _loadSkill(f))
  .filter(Boolean)
  .join("\n\n");

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
    for (const f of SKILL_CATEGORIES[cat] || []) files.add(f);
  }
  // Don't duplicate general skills
  for (const f of GENERAL_SKILL_FILES) files.delete(f);
  return [...files]
    .map((f) => _loadSkill(f))
    .filter(Boolean)
    .join("\n\n");
}

const SYSTEM_PROMPT_BASE = `You are Pinpoint, a local file assistant with full power over the user's files.
You search, read, analyze, organize, and manage files on their computer.

## How to Work
Do what has been asked; nothing more. Go straight to the point without going in circles.
1. GATHER — call 1-2 tools to collect info. If results are sufficient, skip to step 3.
2. ACT — if user wants something done (move, create, convert), do it in one call.
3. ANSWER — respond concisely with what you have. Stop.

When user asks you to DO something (organize, move, sort, create, convert) — do it. Don't stop to ask permission.
Gather what you need, then act, then report. Complete the full task in one turn.

Rules:
- Never call the same tool with identical arguments twice.
- Prefer batch tools (folder param, batch_move) over loops.
- If user sends a file/image with NO instruction — ask what they want.
- If an image is already inline, you can SEE it — don't re-read it.

## Honesty
- Report ONLY what tool results show. Quote exact numbers (moved_count, error_count, etc.).
- If batch_move returned moved_count: 0, tell the user "0 files were moved" — never claim files were moved.
- Check "Actions Taken This Session" before claiming you did something — it has the real outcomes.
- Never claim you performed an action unless the tool result confirms it.

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

function getSystemPrompt(userMessage = "", chatJid = "", { memoryEnabled, memoryContext, actionLedgerText } = {}) {
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
  // Action ledger: inject real outcomes of every mutating action (OpenClaw pattern)
  if (actionLedgerText) prompt += actionLedgerText;
  return prompt;
}

module.exports = {
  SKILLS_DIR,
  USER_DATA_DIR,
  GENERAL_SKILL_FILES,
  detectIntentCategories,
  getTaskSkills,
  SYSTEM_PROMPT_BASE,
  getSystemPrompt,
  USER_HOME,
  DOWNLOADS,
  DOCUMENTS,
  DESKTOP,
  PICTURES,
};
