#!/usr/bin/env node

const { runCliAgent } = require("../index");

function parseArgs(argv) {
  const args = { session: "", message: "" };
  for (let i = 0; i < argv.length; i++) {
    const item = argv[i];
    if (item === "--session") {
      args.session = argv[i + 1] || "";
      i++;
    } else if (item === "--message") {
      args.message = argv[i + 1] || "";
      i++;
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.session || !args.message) {
    process.stderr.write("Usage: pinpoint-cli-agent.js --session SESSION_ID --message TEXT\n");
    process.exit(2);
  }

  try {
    const result = await runCliAgent(args.message, args.session, {
      eventWriter(event) {
        process.stderr.write(`EVENT ${JSON.stringify(event)}\n`);
      },
    });
    process.stdout.write(JSON.stringify(result || {}) + "\n");
  } catch (err) {
    process.stderr.write((err && err.stack) || String(err));
    process.stderr.write("\n");
    process.exit(1);
  }
}

main();
