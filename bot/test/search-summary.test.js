const test = require("node:test");
const assert = require("node:assert/strict");

const { summarizeToolResult } = require("../src/tools");

test("search_documents summary includes top match explanation and lexical mode", () => {
  const summary = summarizeToolResult("search_documents", { query: "invoice 4821" }, {
    results: [
      {
        id: 1,
        match_type: "title",
        why_matched: "exact title phrase match; matched identifier 4821; document text matched 2/2 query concepts",
      },
    ],
    search_explanation: {
      relaxed_lexical: false,
      enhanced_search_used: false,
    },
  });

  assert.equal(
    summary,
    "search_documents: 1 result(s) found (lexical-first) — top match via title: exact title phrase match",
  );
});

test("search_documents summary reports relaxed lexical mode", () => {
  const summary = summarizeToolResult("search_documents", { query: "handoff checklist" }, {
    results: [
      {
        id: 1,
        match_type: "content",
        why_matched: "document text matched 3/3 query concepts",
      },
    ],
    search_explanation: {
      relaxed_lexical: true,
      enhanced_search_used: false,
    },
  });

  assert.match(summary, /search_documents: 1 result\(s\) found \(relaxed lexical\)/);
  assert.match(summary, /top match via content: document text matched 3\/3 query concepts/);
});

test("search_documents ambiguous summary still prefers clarification", () => {
  const summary = summarizeToolResult("search_documents", { query: "section 138 case" }, {
    ambiguous_search: true,
    clarification_hint: "Multiple similar matches found. Can you specify the file name, title, date, person, location, or year?",
    results: [{ id: 1 }],
  });

  assert.equal(
    summary,
    "search_documents: ambiguous — Multiple similar matches found. Can you specify the file name, title, date, person, location, or year?",
  );
});
