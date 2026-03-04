/**
 * LLM adapter layer — Gemini (default) + Ollama (optional)
 *
 * Extracted from bot/index.js (Seg 22C).
 * Provides unified llmGenerate() that routes to Gemini or Ollama,
 * plus token tracking and cost summaries.
 *
 * Usage:
 *   const llm = require("./src/llm");
 *   llm.init({ ai, OLLAMA_MODEL, OLLAMA_URL, OLLAMA_THINK, USE_OLLAMA, GEMINI_MODEL, sessionCosts, TOKEN_COST_INPUT, TOKEN_COST_OUTPUT });
 *   const response = await llm.llmGenerate({ model, contents, config, tools });
 */

// Module-level references set by init()
let _ai, _USE_OLLAMA, _OLLAMA_MODEL, _OLLAMA_URL, _OLLAMA_THINK, _GEMINI_MODEL;
let _sessionCosts, _TOKEN_COST_INPUT, _TOKEN_COST_OUTPUT;

function init(config) {
  _ai = config.ai;
  _USE_OLLAMA = config.USE_OLLAMA;
  _OLLAMA_MODEL = config.OLLAMA_MODEL;
  _OLLAMA_URL = config.OLLAMA_URL;
  _OLLAMA_THINK = config.OLLAMA_THINK;
  _GEMINI_MODEL = config.GEMINI_MODEL;
  _sessionCosts = config.sessionCosts;
  _TOKEN_COST_INPUT = config.TOKEN_COST_INPUT;
  _TOKEN_COST_OUTPUT = config.TOKEN_COST_OUTPUT;
}

// --- Ollama adapter: translates Gemini format <-> Ollama format ---
// So the rest of the code stays identical regardless of which LLM is used.
function geminiToolsToOllama(geminiTools) {
  // Gemini: [{ functionDeclarations: [{ name, description, parameters: { type: "OBJECT", properties, required } }] }]
  // Ollama: [{ type: "function", function: { name, description, parameters: { type: "object", ... } } }]
  if (!geminiTools?.[0]?.functionDeclarations) return [];
  return geminiTools[0].functionDeclarations.map((fd) => ({
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
    const funcCalls = entry.parts.filter((p) => p.functionCall);
    if (funcCalls.length > 0) {
      // Text part if any
      const textParts = entry.parts
        .filter((p) => p.text)
        .map((p) => p.text)
        .join("\n");
      messages.push({
        role: "assistant",
        content: textParts || "",
        tool_calls: funcCalls.map((p) => ({
          id: `call_${p.functionCall.name}_${Date.now()}`,
          type: "function",
          function: { name: p.functionCall.name, arguments: p.functionCall.args || {} },
        })),
      });
      continue;
    }

    // Check for function responses (tool results)
    const funcResponses = entry.parts.filter((p) => p.functionResponse);
    if (funcResponses.length > 0) {
      for (const p of funcResponses) {
        messages.push({
          role: "tool",
          content: JSON.stringify(p.functionResponse.response?.result ?? ""),
        });
      }
      // Also include any text nudges (round-based efficiency hints)
      const textNudges = entry.parts.filter((p) => p.text);
      for (const p of textNudges) {
        messages.push({ role: "system", content: p.text });
      }
      continue;
    }

    // Regular text + images (Ollama uses "images" array with base64 data)
    const text = entry.parts
      .filter((p) => p.text)
      .map((p) => p.text)
      .join("\n");
    const images = entry.parts.filter((p) => p.inlineData).map((p) => p.inlineData.data);
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
    model: _OLLAMA_MODEL,
    messages,
    stream: false,
    think: _OLLAMA_THINK, // Optional thinking — smarter tool picks but slower (~30s vs ~0.8s)
  };
  if (toolsDefs) body.tools = geminiToolsToOllama(toolsDefs);

  const resp = await fetch(`${_OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`Ollama error: ${resp.status} ${await resp.text()}`);
  const data = await resp.json();
  const msg = data.message || {};

  // Translate Ollama response -> Gemini response shape
  const functionCalls = (msg.tool_calls || []).map((tc) => ({
    name: tc.function.name,
    args: typeof tc.function.arguments === "string" ? JSON.parse(tc.function.arguments) : tc.function.arguments,
  }));

  // Separate thinking content from visible response (Qwen3 uses <think>...</think> tags)
  let visibleText = msg.content || "";
  let thinkingTokens = 0;
  if (_OLLAMA_THINK && visibleText.includes("<think>")) {
    const thinkMatch = visibleText.match(/<think>([\s\S]*?)<\/think>/);
    if (thinkMatch) {
      thinkingTokens = Math.ceil(thinkMatch[1].length / 4); // rough estimate
      visibleText = visibleText.replace(/<think>[\s\S]*?<\/think>\s*/g, "").trim();
    }
  }

  return {
    text: visibleText,
    functionCalls: functionCalls.length > 0 ? functionCalls : null,
    candidates: [
      {
        content: {
          role: "model",
          parts: [
            ...(visibleText ? [{ text: visibleText }] : []),
            ...functionCalls.map((fc) => ({ functionCall: { name: fc.name, args: fc.args } })),
          ],
        },
        finishReason: functionCalls.length > 0 ? "TOOL_CALLS" : "STOP",
      },
    ],
    usageMetadata: {
      promptTokenCount: data.prompt_eval_count || 0,
      candidatesTokenCount: data.eval_count || 0,
      thoughtsTokenCount: thinkingTokens,
    },
  };
}

// Unified LLM call — routes to Gemini or Ollama, with retry on transient errors
async function llmGenerate({ model, contents, config, tools: toolsDefs }) {
  if (_USE_OLLAMA) {
    return ollamaGenerate(contents, config, toolsDefs);
  }
  const maxRetries = 2;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await _ai.models.generateContent({ model, contents, config: { ...config, tools: toolsDefs } });
    } catch (err) {
      const msg = String(err.message || err);
      const isTransient =
        msg.includes("429") ||
        msg.includes("503") ||
        msg.includes("500") ||
        msg.includes("RESOURCE_EXHAUSTED") ||
        msg.includes("Internal error");
      if (isTransient && attempt < maxRetries) {
        const wait = 2 ** (attempt + 1) * 1000; // 2s, 4s
        console.warn(`[Gemini] Transient error (${msg.slice(0, 60)}), retry in ${wait / 1000}s...`);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }
      throw err;
    }
  }
}

// --- Token tracking & cost ---

function trackTokens(chatJid, response) {
  const usage = response.usageMetadata;
  if (!usage) return;
  if (!_sessionCosts[chatJid]) {
    _sessionCosts[chatJid] = { input: 0, output: 0, thinking: 0, rounds: 0, started: Date.now() };
  }
  const s = _sessionCosts[chatJid];
  s.input += usage.promptTokenCount || 0;
  s.output += usage.candidatesTokenCount || 0;
  s.thinking += usage.thoughtsTokenCount || 0;
  s.rounds++;
  return {
    input: usage.promptTokenCount || 0,
    output: usage.candidatesTokenCount || 0,
    thinking: usage.thoughtsTokenCount || 0,
  };
}

function formatTokens(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function getCostSummary(chatJid) {
  const s = _sessionCosts[chatJid];
  if (!s || s.rounds === 0) return "No token usage in this session.";
  const cost = s.input * _TOKEN_COST_INPUT + s.output * _TOKEN_COST_OUTPUT;
  const elapsed = Math.round((Date.now() - s.started) / 60000);
  const thinkStr = s.thinking ? `, thinking: ${formatTokens(s.thinking)}` : "";
  return `*Session tokens:* ${formatTokens(s.input + s.output)} (input: ${formatTokens(s.input)}, output: ${formatTokens(s.output)}${thinkStr})\n*Rounds:* ${s.rounds}\n*Estimated cost:* $${cost.toFixed(4)}\n*Duration:* ${elapsed} min`;
}

module.exports = {
  init,
  geminiToolsToOllama,
  lowerTypes,
  geminiContentsToOllama,
  ollamaGenerate,
  llmGenerate,
  trackTokens,
  formatTokens,
  getCostSummary,
};
