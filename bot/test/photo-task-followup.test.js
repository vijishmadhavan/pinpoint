const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  isPhotoStatusFollowUp,
  rememberPhotoTask,
  getLastPhotoTask,
  formatPhotoTaskStatusReply,
} = require("../src/photo-task-followup");

test("photo status follow-up detector catches vague progress questions", () => {
  assert.equal(isPhotoStatusFollowUp("Status"), true);
  assert.equal(isPhotoStatusFollowUp("Is it done?"), true);
  assert.equal(isPhotoStatusFollowUp("How much left"), true);
  assert.equal(isPhotoStatusFollowUp("Cull the wedding folder"), false);
});

test("photo task refs persist last task per chat", () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pinpoint-photo-task-"));
  const refsPath = path.join(tmpDir, "refs.json");

  rememberPhotoTask(
    "chat-1",
    "cull_photos",
    {
      folder: "/mnt/c/Users/vijish/Pictures/WEDDING",
      status: "done",
      report_path: "/mnt/c/Users/vijish/Pictures/WEDDING/_cull_report.html",
      csv_report_path: "/mnt/c/Users/vijish/Pictures/WEDDING/_cull_report.csv",
    },
    refsPath,
  );

  const task = getLastPhotoTask("chat-1", refsPath);
  assert.equal(task.task_type, "cull_photos");
  assert.equal(task.folder, "/mnt/c/Users/vijish/Pictures/WEDDING");
  assert.equal(task.status, "done");
  assert.match(task.updated_at, /^\d{4}-\d{2}-\d{2}T/);
});

test("photo task status reply includes report paths for finished culls", () => {
  const reply = formatPhotoTaskStatusReply(
    {
      task_type: "cull_photos",
      folder: "/mnt/c/Users/vijish/Pictures/WEDDING",
    },
    {
      status: "done",
      kept: 900,
      rejected: 269,
      report_path: "/mnt/c/Users/vijish/Pictures/WEDDING/_cull_report.html",
      csv_report_path: "/mnt/c/Users/vijish/Pictures/WEDDING/_cull_report.csv",
    },
  );

  assert.match(reply, /photo culling for WEDDING is done/i);
  assert.match(reply, /_cull_report\.html/);
  assert.match(reply, /_cull_report\.csv/);
});
