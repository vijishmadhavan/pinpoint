const { readFileSync, writeFileSync, existsSync } = require("fs");
const pathModule = require("path");

const PHOTO_TASK_TO_STATUS_TOOL = {
  cull_photos: "cull_status",
  group_photos: "group_status",
};

const PHOTO_STATUS_FOLLOW_UP =
  /^(status|progress|done\??|is it done\??|is this done\??|is that done\??|did it finish\??|did that finish\??|finished\??|complete\??|completed\??|how much left\??|how far along\??)$/i;

function readPhotoTaskRefs(filePath) {
  try {
    if (!existsSync(filePath)) return {};
    const raw = readFileSync(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (_) {
    return {};
  }
}

function writePhotoTaskRefs(refs, filePath, logger = console) {
  try {
    writeFileSync(filePath, JSON.stringify(refs, null, 2), "utf-8");
  } catch (err) {
    logger?.warn?.("[Pinpoint] Failed to persist photo task refs:", err.message);
  }
}

function rememberPhotoTask(chatJid, taskType, payload = {}, filePath, logger = console) {
  if (!chatJid || !taskType || !PHOTO_TASK_TO_STATUS_TOOL[taskType]) return;
  const refs = readPhotoTaskRefs(filePath);
  refs[chatJid] = {
    task_type: taskType,
    folder: payload.folder || refs[chatJid]?.folder || "",
    status: payload.status || refs[chatJid]?.status || "",
    report_path: payload.report_path || refs[chatJid]?.report_path || "",
    csv_report_path: payload.csv_report_path || refs[chatJid]?.csv_report_path || "",
    updated_at: new Date().toISOString(),
  };
  writePhotoTaskRefs(refs, filePath, logger);
}

function getLastPhotoTask(chatJid, filePath) {
  const refs = readPhotoTaskRefs(filePath);
  const entry = refs[chatJid];
  return entry && entry.folder && PHOTO_TASK_TO_STATUS_TOOL[entry.task_type] ? entry : null;
}

function isPhotoStatusFollowUp(text) {
  return PHOTO_STATUS_FOLLOW_UP.test(String(text || "").trim());
}

function formatPhotoTaskStatusReply(task, result) {
  const folderName = task?.folder ? pathModule.basename(task.folder) : "that folder";
  if (task.task_type === "cull_photos") {
    if (result.status === "done") {
      return `The photo culling for ${folderName} is done. Kept ${result.kept}, rejected ${result.rejected}. Report: ${result.report_path || "N/A"}${result.csv_report_path ? `\nCSV: ${result.csv_report_path}` : ""}`;
    }
    if (result.status === "cancelled") {
      return `The photo culling for ${folderName} was cancelled after ${result.scored || 0}/${result.total || "?"} photos were scored.`;
    }
    return `The photo culling for ${folderName} is still in progress: ${result.scored || 0}/${result.total || "?"} scored${result.percent != null ? ` (${result.percent}%)` : ""}${result.eta_seconds ? `, about ${Math.max(1, Math.round(result.eta_seconds / 60))} minute(s) left` : ""}.`;
  }
  if (task.task_type === "group_photos") {
    if (result.status === "done") {
      return `The photo grouping for ${folderName} is done. Grouped ${result.moved} photo(s). Report: ${result.report_path || "N/A"}`;
    }
    if (result.status === "cancelled") {
      return `The photo grouping for ${folderName} was cancelled after ${result.classified || 0}/${result.total || "?"} photos were classified.`;
    }
    return `The photo grouping for ${folderName} is still in progress: ${result.classified || 0}/${result.total || "?"} classified${result.percent != null ? ` (${result.percent}%)` : ""}${result.eta_seconds ? `, about ${Math.max(1, Math.round(result.eta_seconds / 60))} minute(s) left` : ""}.`;
  }
  return null;
}

module.exports = {
  PHOTO_TASK_TO_STATUS_TOOL,
  readPhotoTaskRefs,
  writePhotoTaskRefs,
  rememberPhotoTask,
  getLastPhotoTask,
  isPhotoStatusFollowUp,
  formatPhotoTaskStatusReply,
};
