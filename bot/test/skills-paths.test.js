const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

function loadSkillsWithEnv(env) {
  const modulePath = path.resolve(__dirname, "..", "src", "skills.js");
  const oldEnv = { ...process.env };
  Object.assign(process.env, env);
  delete require.cache[modulePath];
  const mod = require(modulePath);
  process.env = oldEnv;
  delete require.cache[modulePath];
  return mod;
}

test("skills loader respects explicit PINPOINT_SKILLS_DIR", () => {
  const mod = loadSkillsWithEnv({ PINPOINT_SKILLS_DIR: "/tmp/pinpoint-skills" });
  assert.equal(mod.SKILLS_DIR, "/tmp/pinpoint-skills");
});

test("skills loader respects explicit PINPOINT_USER_DIR", () => {
  const mod = loadSkillsWithEnv({ PINPOINT_USER_DIR: "/tmp/pinpoint-user" });
  assert.equal(mod.USER_DATA_DIR, "/tmp/pinpoint-user");
});
