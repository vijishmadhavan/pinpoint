#!/usr/bin/env node

const { startBot } = require("../index");

console.log("=== Pinpoint WhatsApp Bot ===\n");
startBot().catch((err) => {
  console.error("[Pinpoint] Fatal error:", err);
  process.exit(1);
});
