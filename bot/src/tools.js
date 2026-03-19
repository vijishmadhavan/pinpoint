// tools.js — Tool declarations, routing, intent grouping, and summaries
// Extracted from bot/index.js (Seg 22C modularization)
// CommonJS module — no shared state dependencies

const pathModule = require("path");
const { existsSync, statSync } = require("fs");

// --- Intent detection keywords → categories ---
const INTENT_KEYWORDS = {
  image:
    /photo|image|picture|jpg|png|face|person|selfie|detect|object|bounding|visual|heic|camera|screenshot|exif|metadata|gps|lens|aperture|iso|when.*taken|shot.*with|cull|score.*photo|rate.*photo|best.*photo|reject|keeper|group.*photo|segregat|categoriz|classify/i,
  search: /find|search|where|which|document|file.*contain|look.*for|indexed/i,
  data: /excel|csv|spreadsheet|column|row|data|analyze|chart|graph|pandas|filter|sort/i,
  files: /move|copy|rename|delete|duplicate|folder|list|organize|clean.*up|batch|zip|unzip|compress|extract|archive/i,
  write: /write|create|save|put.*file|store|\.txt|\.csv|\.json|\.md|pdf|merge|split|combine|generate/i,
  media: /video|mp4|clip|frame|scene|audio|mp3|wav|voice|transcri|podcast|recording|speech|listen/i,
  web: /download|url|web|search.*online|internet|website|news|weather|score|playing|match|price|stock|latest|current|today|tomorrow|yesterday|live|trending|release/i,
  memory: /remember|forget|memory|preference/i,
  code: /python|code|script|run|execute|program/i,
  google: /email|mail|gmail|send.*mail|inbox|calendar|event|meeting|schedule|appointment|drive|upload.*drive|google/i,
};

// --- Skill file categories (maps intent → skill .md files) ---
const SKILL_CATEGORIES = {
  image: [
    "face-analysis.md",
    "image-analysis.md",
    "image-tools.md",
    "visual-search.md",
    "photo-cull.md",
    "photo-group.md",
  ],
  search: ["search.md"],
  data: ["data-analysis.md"],
  files: ["file-tools.md", "smart-ops.md", "archive-tools.md"],
  write: ["write-create.md", "pdf-tools.md"],
  media: ["video-search.md", "audio-search.md"],
  web: ["web-search.md", "download.md"],
  memory: ["memory.md"],
  code: ["python.md"],
  google: ["google-workspace.md"],
};

// --- Tool grouping: map each tool to intent categories (mirrors SKILL_CATEGORIES) ---
// Core tools are always included. Category tools added based on user message intent.
const CORE_TOOLS = new Set([
  "search_documents",
  "search_facts",
  "read_document",
  "read_file",
  "list_files",
  "find_file",
  "send_file",
  "get_status",
  "calculate",
]);
const TOOL_GROUPS = {
  search: ["search_history", "grep_files", "index_file", "find_file", "search_generated_files"],
  image: [
    "detect_faces",
    "crop_face",
    "find_person",
    "find_person_by_face",
    "count_faces",
    "compare_faces",
    "remember_face",
    "forget_face",
    "search_images_visual",
    "ocr",
    "image_metadata",
    "score_photo",
    "cull_photos",
    "cull_status",
    "suggest_categories",
    "group_photos",
    "group_status",
  ],
  data: ["analyze_data", "read_excel", "generate_chart", "extract_tables", "pdf_to_excel"],
  files: [
    "file_info",
    "move_file",
    "copy_file",
    "batch_move",
    "create_folder",
    "delete_file",
    "find_duplicates",
    "batch_rename",
    "compress_files",
    "extract_archive",
    "search_generated_files",
    "find_file",
  ],
  write: [
    "write_file",
    "generate_excel",
    "merge_pdf",
    "split_pdf",
    "pdf_to_images",
    "images_to_pdf",
    "compress_pdf",
    "add_page_numbers",
    "pdf_to_word",
    "organize_pdf",
    "pdf_to_excel",
    "resize_image",
    "convert_image",
    "crop_image",
  ],
  media: ["search_video", "extract_frame", "transcribe_audio", "search_audio"],
  web: ["web_search", "download_url"],
  memory: ["memory_save", "memory_search", "memory_delete", "memory_forget"],
  code: ["run_python"],
  archive: ["compress_files", "extract_archive"],
  google: ["gmail_send", "gmail_search", "gmail_triage", "calendar_events", "calendar_create", "drive_list", "drive_upload"],
  automation: ["set_reminder", "list_reminders", "cancel_reminder", "watch_folder", "unwatch_folder", "list_watched"],
};

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

// Per-chat intent memory: carry forward intent for short follow-ups (Claude Code pattern)
const lastIntentCats = {}; // chatJid → Set of categories from last substantive message

// Build filtered tools array based on user message intent
// Returns [{ functionDeclarations: [...] }] for Gemini
function getToolsForIntent(message, chatJid) {
  const cats = detectIntentCategories(message);
  // For short follow-ups, merge with previous intent so tools carry over
  // Claude Code always sends all tools; we approximate by carrying forward intent
  if (chatJid && message.split(/\s+/).length <= 6) {
    const prev = lastIntentCats[chatJid];
    if (prev) for (const c of prev) cats.add(c);
  }
  // Action words in follow-ups always need files tools (move, create, organize)
  if (/go ahead|do it|finish|start|proceed|execute|make it|confirm/i.test(message)) {
    cats.add("files");
    cats.add("write");
  }
  const allowedNames = new Set(CORE_TOOLS);
  for (const cat of cats) {
    for (const name of TOOL_GROUPS[cat] || []) allowedNames.add(name);
  }
  // automation tools always available (reminders, watch)
  for (const name of TOOL_GROUPS.automation) allowedNames.add(name);

  // Store intent for follow-ups (only for substantive messages)
  if (chatJid && message.split(/\s+/).length > 3) {
    lastIntentCats[chatJid] = cats;
  }

  const filtered = TOOL_DECLARATIONS.filter((fd) => allowedNames.has(fd.name));
  console.log(`[Pinpoint] Tools: ${filtered.length}/${TOOL_DECLARATIONS.length} (intents: ${[...cats].join(",")})`);
  return [{ functionDeclarations: filtered }];
}

// Clear intent memory for a chat (called on session reset)
function clearIntentCache(chatJid) {
  delete lastIntentCats[chatJid];
}

// Check if chat has active intent context (for follow-up detection)
function hasActiveIntent(chatJid) {
  const cats = lastIntentCats[chatJid];
  return cats && cats.size > 0;
}

// --- Tool declarations for Gemini ---
const TOOL_DECLARATIONS = [
  {
    name: "search_documents",
    description:
      "Search indexed documents by keywords. Returns the exact matching section/paragraph (not just filenames). PREFERRED way to answer questions about document content — always try this FIRST before read_document or read_file. Also searches indexed IMAGE CAPTIONS — use file_type='image' to find photos by description (free, instant). If the file is not indexed yet, use index_file first, then search. Can filter by file type and folder.",
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
    name: "read_document_overview",
    description:
      "Read a compact overview of a document by its ID. Use this BEFORE read_document when search_documents found the right file but you need broader context than the snippet. Returns metadata, a short overview, top sections, and extracted facts when available.",
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
    name: "read_document",
    description:
      "Read the full text of a document by its ID. Use ONLY after search_documents and usually after read_document_overview if you still need broader context — like summarizing an entire document, comparing two full documents, or translating. For specific questions (what does clause 7 say, what's the depreciation amount), search_documents already returns the exact section.",
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
    description:
      "Read specific cells or ranges from an Excel (.xlsx) file. Use when the user asks about specific cells, rows, columns, or ranges. For general search, use search_documents instead.",
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
    description:
      "Evaluate a mathematical expression. Supports +, -, *, /, **, %, parentheses, and functions like round(), abs(), min(), max(), sum(), sqrt(). Use for any arithmetic: sums, averages, percentages, conversions.",
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
    description:
      "List files and folders in a directory. WORKFLOW: 1) Use sort_by='size' to find large files. 2) Use name_contains to search by filename. 3) Use recursive=true to search in subfolders. 4) Use filter_ext or filter_type to narrow by type. The response includes a 'largest' field when sorted by size. Do NOT call repeatedly with different params — if you can't find a file in the first result, try name_contains or recursive.",
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
          description:
            "Filter by category: 'image', 'document', 'spreadsheet', 'presentation', 'video', 'audio', 'archive'. Optional.",
        },
        name_contains: {
          type: "STRING",
          description:
            "Search by filename containing this text (case-insensitive). E.g. 'invoice' finds 'Invoice_2024.pdf'.",
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
    name: "find_file",
    description:
      "Find any file on the computer by filename. Searches a pre-built path registry of all files in common folders (Documents, Desktop, Downloads, Pictures, Videos). INSTANT — no folder scanning needed. Use this FIRST when a user asks about a file and you don't know which folder it's in. Can filter by extension. For files YOU created, also try search_generated_files.",
    parameters: {
      type: "OBJECT",
      properties: {
        query: {
          type: "STRING",
          description: "Filename to search for (case-insensitive). E.g. 'rent', 'invoice', 'budget'.",
        },
        ext: {
          type: "STRING",
          description: "Filter by extension: '.pdf', '.xlsx', '.docx'. Optional.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "search_generated_files",
    description:
      "Search files previously CREATED by Pinpoint tools (write_file, generate_excel, run_python, download_url, merge_pdf, etc.). Use when user asks about a file they asked you to create in a previous conversation — these files are NOT indexed in documents, so search_documents won't find them. Search by filename, description, or tool name.",
    parameters: {
      type: "OBJECT",
      properties: {
        query: {
          type: "STRING",
          description: "Search term — matches filename and description. E.g. 'rent', 'chart', 'excel'.",
        },
        tool_name: {
          type: "STRING",
          description: "Filter by tool: write_file, generate_excel, generate_chart, run_python, download_url, merge_pdf, split_pdf, etc. Optional.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "grep_files",
    description:
      "Search INSIDE files for text content. Finds files containing a pattern and shows matching lines. Use when you need to find which file contains specific text (a name, phone number, keyword). Works on any text file — no indexing needed.",
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
    description:
      "Get detailed information about a file or folder: size, creation date, modification date, file type, and whether it's indexed in Pinpoint's database.",
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
    description:
      "Move or copy multiple files to a destination folder in one call. Much faster than calling move_file repeatedly. Creates destination folder if needed.",
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
    description:
      "Delete a file. SAFETY: ALWAYS ask the user for explicit confirmation before deleting. Never delete without user approval. Cannot delete folders.",
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
    description:
      "Send a file to the user on WhatsApp. ONLY use this when the user explicitly asks to receive/send/share a file. Never send files automatically. Max 16MB for images, 100MB for documents. If too large, use resize_image or compress_files first.",
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
    description:
      "Read a file from disk. For images: you SEE the image visually. If the user sent a photo, it is already visible to you — do NOT call read_file on it again. For documents (PDF, DOCX, TXT): prefer index_file + search_documents for searching. For Excel: use analyze_data instead.",
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
    description:
      "Search past conversation messages from previous sessions. Use when the user refers to something discussed earlier that's not in the current conversation, like 'that file from yesterday' or 'what did I search for last time?'.",
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
    description:
      "Detect and analyze faces in an image or all images in a folder. Returns face count, bounding boxes, confidence, age, gender, head pose. Pass folder for batch processing (one call instead of many).",
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
    description:
      "Crop a specific face from an image and save it as a separate file. Use this when detect_faces found multiple faces and you need to show them to the user so they can pick which person to search for. Returns the path to the cropped face image which you can send via send_file.",
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
    description:
      "Find all photos of a specific person in a folder. The reference image should contain exactly ONE face. If the reference has multiple faces, first use detect_faces + crop_face to let the user pick, then use find_person_by_face instead. Scans all images in the folder and returns matching photos sorted by similarity. First scan may take a while (caches results for instant repeat searches).",
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
    description:
      "Find all photos of a specific person using a face index from a multi-face reference image. Use this when the reference image has multiple faces and the user has picked which face to search for (via detect_faces + crop_face). The face_idx comes from detect_faces.",
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
    description:
      "Count faces in an image, a list of images, or all images in a folder. Returns face count, age/gender breakdown. Use paths array to batch multiple specific images in ONE call instead of calling per-image.",
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
          description:
            "Array of image paths to count faces in batch. Use this instead of calling count_faces per-image.",
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
    description:
      "Compare two specific faces from two images to check if they are the same person. Returns similarity score (0-1) and confidence level. Use when user asks 'is this the same person?' or wants to verify identity across photos.",
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
    description:
      "Save a face for future recognition. After this, detect_faces will auto-identify this person in any photo. One person can have multiple saved faces (different angles improve accuracy). Use detect_faces first to get face_idx.",
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
    description:
      "Delete all saved face data for a person. After this, they will no longer be auto-recognized by detect_faces.",
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
    description:
      "Extract text from an image or scanned PDF using OCR. Use this when you need the text as a string for processing. For just SEEING an image, use read_file instead (sends image visually). Pass folder for batch processing. After OCR, use index_file to make the text searchable.",
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
    name: "analyze_data",
    description:
      "Run pandas data analysis on CSV or Excel files. WORKFLOW: 1) FIRST call with operation='columns' to see all sheets, column names, types, and sample values. 2) Use operation='search' with query to find values across ALL sheets — auto-normalizes phone/ID formats (strips dashes, parens). 3) Use filter/groupby/sort when you know the exact column name. Operations: columns (schema+sheets), search (grep-like across all cells), describe, head, filter, value_counts, groupby, corr, sort, unique, shape, eval. For phone/ID lookups ALWAYS use search — it normalizes automatically. File is cached after first load (instant repeat calls).",
    parameters: {
      type: "OBJECT",
      properties: {
        path: {
          type: "STRING",
          description: "Absolute path to CSV or Excel file.",
        },
        operation: {
          type: "STRING",
          description:
            "Operation: columns (FIRST — see sheets+types), search (find values across all sheets), describe, head, filter, value_counts, groupby, corr, sort, unique, shape, eval.",
        },
        columns: {
          type: "STRING",
          description: "Column name(s). For groupby use 'group_col:agg_col'. For sort prefix with '-' for descending.",
        },
        query: {
          type: "STRING",
          description:
            "For search: value to find (e.g. '9208896630' — auto-normalizes phone formats). For filter: pandas query like 'amount > 1000'. For eval: expression like 'df.groupby(\"Cat\")[[\"Price\"]].sum()'.",
        },
        sheet: {
          type: "STRING",
          description:
            "Sheet name for Excel files. Omit to use first sheet (or search ALL sheets with operation='search').",
        },
      },
      required: ["path", "operation"],
    },
  },
  {
    name: "index_file",
    description:
      "Index a single file into the search database. Extracts text, chunks it into sections, and makes it searchable. Use this BEFORE search_documents when the file hasn't been indexed yet. After indexing, use search_documents to find specific sections within the file.",
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
    description:
      "Create or write a text file. Can append to existing files. Use for creating notes, reports, summaries, text exports.",
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
    description:
      'Create a professionally formatted Excel file. Auto-applies: dark header with white text, alternating row stripes, auto-fit column widths, freeze panes, auto-filter. Optional title row above data.\n\nExample — expense report:\n  title: "March 2026 Expenses"\n  data: [{"Category":"Food","Amount":1200,"Date":"2026-03-01"},{"Category":"Transport","Amount":450,"Date":"2026-03-05"}]\n\nExample — contact list:\n  data: [{"Name":"Alice","Phone":"555-1234","Email":"alice@example.com"}]\n\nTips: Use descriptive column names (not "col1"). Group related data. Add a title for context.',
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "Output path for .xlsx file." },
        data: {
          type: "ARRAY",
          description: 'List of row objects. Keys become column headers. E.g. [{"Name":"Alice","Amount":100}].',
          items: { type: "OBJECT" },
        },
        title: { type: "STRING", description: "Optional title displayed above the table (merged across columns, large bold text). E.g. 'Q1 Sales Report'." },
        sheet_name: { type: "STRING", description: "Sheet name. Default: Sheet1." },
      },
      required: ["path", "data"],
    },
  },
  {
    name: "generate_chart",
    description:
      "Create a chart image (bar, line, pie, scatter, hist) from data using matplotlib. Returns image path — send via send_file.",
    parameters: {
      type: "OBJECT",
      properties: {
        data: { type: "OBJECT", description: 'Chart data: {"labels": [...], "values": [...]}.' },
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
    name: "organize_pdf",
    description:
      "Reorder, duplicate, or remove PDF pages. Pass ordered list of page numbers to define output. E.g. [3,1,2] reverses first 3 pages; [1,1,2] duplicates page 1; omit pages to remove them.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "Source PDF path." },
        pages: { type: "ARRAY", items: { type: "INTEGER" }, description: "Ordered list of 1-based page numbers for output." },
        output_path: { type: "STRING", description: "Output path for reorganized PDF." },
      },
      required: ["path", "pages", "output_path"],
    },
  },
  {
    name: "pdf_to_excel",
    description: "Extract tables from a PDF and save as Excel (.xlsx). Each table becomes a separate sheet. Works on native PDFs with structured tables.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "PDF file path." },
        output_path: { type: "STRING", description: "Output .xlsx path. Default: same name with .xlsx." },
        pages: { type: "STRING", description: "Page range: '1-5', '3', 'all'. Default: all." },
      },
      required: ["path"],
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
    name: "compress_pdf",
    description: "Compress a PDF to reduce file size. Removes unused objects and deflates streams. Reports size reduction %.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "PDF file path to compress." },
        output_path: { type: "STRING", description: "Output path. Omit to overwrite original." },
      },
      required: ["path"],
    },
  },
  {
    name: "add_page_numbers",
    description: "Add page numbers to every page of a PDF. Supports 'Page {n} of {total}' format.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "PDF file path." },
        output_path: { type: "STRING", description: "Output path. Omit to overwrite original." },
        position: { type: "STRING", description: "Position: bottom-left, bottom-center (default), bottom-right." },
        start: { type: "INTEGER", description: "Starting page number. Default 1." },
        format: { type: "STRING", description: "Format string. Use {n} for number, {total} for total. Default: '{n}'." },
      },
      required: ["path"],
    },
  },
  {
    name: "pdf_to_word",
    description: "Convert a PDF to Word (.docx). Extracts text with basic formatting (bold, italic, font size). Best for text-heavy PDFs.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "PDF file path." },
        output_path: { type: "STRING", description: "Output .docx path. Default: same name with .docx extension." },
      },
      required: ["path"],
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
        output_path: {
          type: "STRING",
          description: "Output path. Optional — defaults to same name with new extension.",
        },
      },
      required: ["path", "format"],
    },
  },
  {
    name: "crop_image",
    description:
      "Crop an image to specified rectangle. Coordinates in pixels. Use file_info or read_file first to know image dimensions before cropping.",
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
    name: "image_metadata",
    description:
      "Extract EXIF metadata from photos: date taken, camera model, GPS coordinates, lens, aperture, ISO, dimensions. Use for: 'when was this taken?', 'what camera?', 'where was this shot?', 'show timeline'. Pass folder for batch metadata of all images.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "Absolute path to image." },
        folder: { type: "STRING", description: "Absolute path to folder — get metadata for ALL images." },
      },
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
    description:
      "Rename files matching a regex pattern. ALWAYS call with dry_run=true first to preview changes, show the user, and only execute with dry_run=false after confirmation.",
    parameters: {
      type: "OBJECT",
      properties: {
        folder: { type: "STRING", description: "Folder containing files to rename." },
        pattern: { type: "STRING", description: "Regex pattern to match in filenames." },
        replace: { type: "STRING", description: "Replacement string." },
        dry_run: {
          type: "BOOLEAN",
          description: "Preview only (default true). Set false to execute after user confirms.",
        },
      },
      required: ["folder", "pattern", "replace"],
    },
  },
  {
    name: "run_python",
    description:
      "Execute Python code. Use for any custom operation: image manipulation, data processing, file operations, calculations, generating files. Pre-loaded: PIL, pandas, numpy, matplotlib, os, json. Working dir: /tmp/pinpoint_python/. Print results to stdout.",
    parameters: {
      type: "OBJECT",
      properties: {
        code: {
          type: "STRING",
          description: "Python code to execute. Use print() for output. Save files to WORK_DIR.",
        },
        timeout: { type: "INTEGER", description: "Max execution time in seconds. Default 30, max 120." },
      },
      required: ["code"],
    },
  },
  {
    name: "memory_save",
    description:
      "Save a fact to persistent memory. Persists across sessions and restarts. Smart dedup: skips duplicates, merges related facts, handles contradictions. Only works when memory is enabled.",
    parameters: {
      type: "OBJECT",
      properties: {
        fact: { type: "STRING", description: "The fact to remember. Keep it short and factual." },
        category: {
          type: "STRING",
          description: "Category: people, places, preferences, professional, health, plans, general. Default: general.",
        },
      },
      required: ["fact"],
    },
  },
  {
    name: "memory_search",
    description:
      "Search persistent memories by keyword. Use to recall personal facts about the user before saying 'I don't know'. Only works when memory is enabled.",
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
    description:
      "Forget a memory by description — no ID needed. Searches memories for best match and deletes it. Use when user says 'forget that I like dark mode' or 'remove the thing about Mumbai'. Preferred over memory_delete when you don't have the ID.",
    parameters: {
      type: "OBJECT",
      properties: {
        description: {
          type: "STRING",
          description:
            "Natural language description of what to forget. E.g. 'dark mode preference', 'living in Mumbai'.",
        },
      },
      required: ["description"],
    },
  },
  {
    name: "search_facts",
    description:
      "Search extracted facts from indexed documents. Facts are key details (names, dates, amounts, topics) auto-extracted at index time. Use for quick factual lookups like 'who is the electrician?' or 'what was the invoice amount?'. Falls back to search_documents for full-text search if facts don't have enough detail.",
    parameters: {
      type: "OBJECT",
      properties: {
        query: {
          type: "STRING",
          description: "Search keywords for facts. E.g. 'electrician', 'invoice amount', 'meeting date'.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "web_search",
    description:
      "Search the web for real-world information. Use for news, weather, sports, people, products, prices, current events, comparisons — anything NOT in local files. Returns structured search results with titles, snippets, and URLs. The results are reliable and current — answer directly from them. Do NOT fall back to search_documents or search_facts for web queries. To read full content of a result, call again with that result's URL.",
    parameters: {
      type: "OBJECT",
      properties: {
        query: { type: "STRING", description: "Search query. Be specific." },
        url: {
          type: "STRING",
          description:
            "Optional: a specific URL to read full content. Use to get details from a search result.",
        },
        count: { type: "INTEGER", description: "Number of results (default 10, max 20)." },
        freshness: { type: "STRING", description: "Time filter: noLimit (default), day, week, month." },
        start: {
          type: "INTEGER",
          description: "Character offset for long pages (when reading a URL). Use the end value from previous response.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "search_images_visual",
    description:
      "Search images in a folder by text description using AI vision. IMPORTANT: Try search_documents(query, file_type='image', folder=...) FIRST — indexed image captions are free and instant. Only use this tool if search_documents returns no results or the folder isn't indexed. Returns ranked list of images matching the query. Results are RELIABLE — trust them for categorization, grouping, and answering without manually inspecting individual images. Do NOT call read_file/resize_image/ocr on photos after getting visual search results. First call may take time, subsequent queries on same folder are faster (cached). Pass queries as array for batch search (multiple queries in one call). GROUPING WORKFLOW: When organizing/grouping photos into categories — 1) first run with default limit (10) as a PREVIEW, 2) show user the proposed categories with sample counts, 3) ask 'Shall I do the full folder?', 4) if yes, re-run with limit=200 per category to cover all photos.",
    parameters: {
      type: "OBJECT",
      properties: {
        folder: { type: "STRING", description: "Absolute path to folder containing images." },
        query: { type: "STRING", description: "Single text query. E.g. 'bride cutting cake'." },
        queries: {
          type: "ARRAY",
          items: { type: "STRING" },
          description:
            "Multiple queries for batch search. E.g. ['dancing', 'flowers', 'group photo']. Use instead of query for multiple searches at once.",
        },
        limit: { type: "INTEGER", description: "Max results per query. Default 10." },
      },
      required: ["folder"],
    },
  },
  {
    name: "search_video",
    description:
      "Search inside a video by text description. Gemini analyzes the full video natively (up to 3h). Returns timestamps of matching moments. After finding timestamps, use extract_frame to get the image — it will be auto-sent to the user.",
    parameters: {
      type: "OBJECT",
      properties: {
        video_path: { type: "STRING", description: "Absolute path to video file." },
        query: {
          type: "STRING",
          description: "Text description of what to find. E.g. 'person dancing', 'sunset scene'.",
        },
        fps: {
          type: "NUMBER",
          description: "Frames per second to extract. Default 1. Use 0.5 for long videos, 2 for short clips.",
        },
        limit: { type: "INTEGER", description: "Max results. Default 5." },
      },
      required: ["video_path", "query"],
    },
  },
  {
    name: "extract_frame",
    description:
      "Extract a single frame from a video at a specific timestamp. Returns the frame as an image file. Use after search_video to get the actual frame image for sending.",
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
    name: "transcribe_audio",
    description:
      "Transcribe an audio file to text using AI. Returns full transcript with timestamps. Supports: mp3, wav, flac, aac, ogg, wma, m4a, aiff. Up to 9.5 hours of audio.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "Absolute path to audio file." },
      },
      required: ["path"],
    },
  },
  {
    name: "search_audio",
    description:
      "Search within an audio file for specific content (speech, sounds, topics). Returns timestamps of matching moments with relevance scores. Use for finding specific parts in podcasts, recordings, voice memos.",
    parameters: {
      type: "OBJECT",
      properties: {
        audio_path: { type: "STRING", description: "Absolute path to audio file." },
        query: {
          type: "STRING",
          description:
            "What to find in the audio. E.g. 'discussion about pricing', 'laughter', 'someone saying hello'.",
        },
        limit: { type: "INTEGER", description: "Max results. Default 5." },
      },
      required: ["audio_path", "query"],
    },
  },
  {
    name: "set_reminder",
    description:
      "Set a reminder that will be sent to the user at a specific time. Supports one-time or recurring. Use when user says 'remind me to X at/by Y time' or 'remind me every Monday'. Persists across restarts.",
    parameters: {
      type: "OBJECT",
      properties: {
        message: { type: "STRING", description: "The reminder message. E.g. 'Buy tablets', 'Call dentist'." },
        time: {
          type: "STRING",
          description:
            "When to remind. ISO format preferred: '2026-02-27T17:00:00'. Also accepts: '17:00' (today), '5pm' (today), 'in 2 hours', 'tomorrow 9am'.",
        },
        repeat: {
          type: "STRING",
          description:
            "Optional. Repeat schedule: 'daily', 'weekly', 'monthly', 'weekdays'. Omit for one-time reminder.",
        },
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
    description:
      "Extract structured tables from a PDF. Returns headers + rows for each table found. Works on native PDFs (not scanned). Use for invoices, reports, financial statements, any PDF with tabular data.",
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
    description:
      "Start auto-indexing a folder. New or modified files will be automatically indexed for search. Persists across restarts.",
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
  {
    name: "score_photo",
    description:
      "Score a single photo's quality using Gemini vision (/100). Returns technical (sharpness, exposure, composition, quality) + aesthetic (emotion, interest, keeper) breakdown with reasoning. Use for quick single-photo evaluation.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "Absolute path to the image file." },
      },
      required: ["path"],
    },
  },
  {
    name: "cull_photos",
    description:
      "Auto-cull photos in a folder: score ALL images, keep top N%, move rejects to _rejects subfolder. Generates an HTML report with thumbnail gallery. WORKFLOW: 1) list_files to survey folder 2) confirm with user 3) cull_photos 4) poll cull_status until done 5) report results + send report file. Background job — use cull_status to poll.",
    parameters: {
      type: "OBJECT",
      properties: {
        folder: { type: "STRING", description: "Folder containing photos to cull." },
        keep_pct: { type: "INTEGER", description: "Percentage of photos to keep (1-99). Default 80." },
        rejects_folder: { type: "STRING", description: "Custom folder for rejects. Default: <folder>/_rejects." },
      },
      required: ["folder"],
    },
  },
  {
    name: "cull_status",
    description:
      "Check progress of a running cull_photos job. Returns scored/total count, ETA, and final stats when done. Poll every few seconds until status is 'done'. Set cancel=true to stop the job.",
    parameters: {
      type: "OBJECT",
      properties: {
        folder: { type: "STRING", description: "The folder being culled (same as passed to cull_photos)." },
        cancel: { type: "BOOLEAN", description: "Set true to cancel the running job." },
      },
      required: ["folder"],
    },
  },
  {
    name: "suggest_categories",
    description:
      "Sample ~20 photos from a folder and let Gemini suggest 4-8 grouping categories. Use BEFORE group_photos to auto-discover categories. Returns suggested category names for user confirmation.",
    parameters: {
      type: "OBJECT",
      properties: {
        folder: { type: "STRING", description: "Folder containing photos to analyze." },
      },
      required: ["folder"],
    },
  },
  {
    name: "group_photos",
    description:
      "Auto-group ALL photos in a folder by Gemini vision classification. Each photo is sent to Gemini with the category list — Gemini picks the best match. Photos MOVED to category subfolders (destructive). Generates HTML report. Classifications cached in DB — re-runs are free. IMPORTANT: NEVER call this without user confirmation of categories first. Always show suggest_categories results and WAIT for user approval before calling this. Background job — use group_status to poll.",
    parameters: {
      type: "OBJECT",
      properties: {
        folder: { type: "STRING", description: "Folder containing photos to group." },
        categories: {
          type: "ARRAY",
          items: { type: "STRING" },
          description: "List of category names to classify into (e.g. ['Ceremony', 'Portraits', 'Family', 'Rituals']).",
        },
      },
      required: ["folder", "categories"],
    },
  },
  {
    name: "group_status",
    description:
      "Check progress of a running group_photos job. Returns classified/total count, ETA, and final group counts when done. Poll every few seconds until status is 'done'. Set cancel=true to stop the job.",
    parameters: {
      type: "OBJECT",
      properties: {
        folder: { type: "STRING", description: "The folder being grouped (same as passed to group_photos)." },
        cancel: { type: "BOOLEAN", description: "Set true to cancel the running job." },
      },
      required: ["folder"],
    },
  },
  // --- Google Workspace tools (gws-cli) ---
  {
    name: "gmail_send",
    description:
      "Send an email via Gmail. Can optionally attach a local file. Use when user says 'email this to X' or 'send mail'.",
    parameters: {
      type: "OBJECT",
      properties: {
        to: { type: "STRING", description: "Recipient email address." },
        subject: { type: "STRING", description: "Email subject line." },
        body: { type: "STRING", description: "Email body (plain text)." },
        attach: { type: "STRING", description: "Optional: absolute path to file to attach." },
      },
      required: ["to", "subject", "body"],
    },
  },
  {
    name: "gmail_search",
    description:
      "Search Gmail inbox. Uses same query syntax as Gmail search bar (from:, to:, subject:, has:attachment, after:, before:, etc).",
    parameters: {
      type: "OBJECT",
      properties: {
        query: { type: "STRING", description: "Gmail search query (e.g. 'from:john invoice after:2026/01/01')." },
        limit: { type: "NUMBER", description: "Max results (default 10)." },
      },
      required: ["query"],
    },
  },
  {
    name: "gmail_triage",
    description: "Show unread inbox summary — sender, subject, date for each unread message. Use for 'check my email' or 'any new mail?'.",
    parameters: { type: "OBJECT", properties: {} },
  },
  {
    name: "calendar_events",
    description:
      "List upcoming calendar events. Use for 'what's on my calendar', 'am I free tomorrow', 'meetings this week'.",
    parameters: {
      type: "OBJECT",
      properties: {
        days: { type: "NUMBER", description: "Days ahead to show (default 7, max 30)." },
        today: { type: "BOOLEAN", description: "Set true to show only today's events." },
      },
    },
  },
  {
    name: "calendar_create",
    description:
      "Create a calendar event. Times must be ISO 8601 format (e.g. 2026-03-07T15:00:00+05:30). Use user's timezone (IST = +05:30).",
    parameters: {
      type: "OBJECT",
      properties: {
        summary: { type: "STRING", description: "Event title." },
        start: { type: "STRING", description: "Start time in ISO 8601 (e.g. 2026-03-07T15:00:00+05:30)." },
        end: { type: "STRING", description: "End time in ISO 8601." },
        location: { type: "STRING", description: "Optional: event location." },
        description: { type: "STRING", description: "Optional: event description." },
        attendees: {
          type: "ARRAY",
          items: { type: "STRING" },
          description: "Optional: list of attendee email addresses.",
        },
      },
      required: ["summary", "start", "end"],
    },
  },
  {
    name: "drive_list",
    description:
      "List or search Google Drive files. Empty query shows recent files. Use for 'what files do I have on Drive', 'find report on Drive'.",
    parameters: {
      type: "OBJECT",
      properties: {
        query: { type: "STRING", description: "Search term (searches file names and content). Empty = recent files." },
        limit: { type: "NUMBER", description: "Max results (default 10)." },
      },
    },
  },
  {
    name: "drive_upload",
    description:
      "Upload a local file to Google Drive. Use when user says 'upload to Drive', 'save to Drive', 'put on Drive'.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: { type: "STRING", description: "Absolute path to the local file to upload." },
        name: { type: "STRING", description: "Optional: target filename on Drive (defaults to local filename)." },
      },
      required: ["path"],
    },
  },
];

// --- Declarative tool routing table ---
// Each entry: { m: method, p: path/fn, b: body mapper (POST only) }
// Replaces 320-line switch for simple API-pass-through tools.
// Note: search_documents uses MAX_RESULTS which must be passed in at init time.
const enc = encodeURIComponent;

function buildToolRoutes(maxResults) {
  return {
    // --- GET routes ---
    search_documents: {
      m: "GET",
      p: (a) => {
        let u = `/search?q=${enc(a.query || "")}&limit=${maxResults}`;
        if (a.file_type) u += `&file_type=${enc(a.file_type)}`;
        if (a.folder) u += `&folder=${enc(a.folder)}`;
        return u;
      },
    },
    read_document: { m: "GET", p: (a) => `/document/${a.document_id}` },
    read_document_overview: { m: "GET", p: (a) => `/document/${a.document_id}/overview` },
    list_files: {
      m: "GET",
      p: (a) => {
        let u = `/list_files?folder=${enc(a.folder)}`;
        if (a.sort_by) u += `&sort_by=${enc(a.sort_by)}`;
        if (a.filter_ext) u += `&filter_ext=${enc(a.filter_ext)}`;
        if (a.filter_type) u += `&filter_type=${enc(a.filter_type)}`;
        if (a.name_contains) u += `&name_contains=${enc(a.name_contains)}`;
        if (a.recursive) u += `&recursive=true`;
        return u;
      },
    },
    find_file: {
      m: "GET",
      p: (a) => {
        let u = `/find-file?query=${enc(a.query)}`;
        if (a.ext) u += `&ext=${enc(a.ext)}`;
        return u;
      },
    },
    search_generated_files: {
      m: "GET",
      p: (a) => {
        let u = `/search-generated-files?query=${enc(a.query || "")}`;
        if (a.tool_name) u += `&tool_name=${enc(a.tool_name)}`;
        return u;
      },
    },
    file_info: { m: "GET", p: (a) => `/file_info?path=${enc(a.path)}` },
    get_status: { m: "GET", p: () => "/status" },
    search_history: { m: "GET", p: (a) => `/conversation/search?q=${enc(a.query || "")}&limit=10` },
    search_facts: { m: "GET", p: (a) => `/search-facts?q=${enc(a.query)}&limit=10` },
    list_watched: { m: "GET", p: "/watched-folders" },
    // --- Simple POST routes (tool args → API body) ---
    read_excel: {
      m: "POST",
      p: "/read_excel",
      b: (a) => ({ path: a.path, sheet_name: a.sheet_name || null, cell_range: a.cell_range || null }),
    },
    calculate: { m: "POST", p: "/calculate", b: (a) => ({ expression: a.expression }) },
    grep_files: {
      m: "POST",
      p: "/grep",
      b: (a) => ({ pattern: a.pattern, folder: a.folder, file_filter: a.file_filter }),
    },
    batch_move: {
      m: "POST",
      p: "/batch_move",
      b: (a) => ({ sources: a.sources || [], destination: a.destination, is_copy: a.is_copy || false }),
    },
    move_file: {
      m: "POST",
      p: "/move_file",
      b: (a) => ({ source: a.source, destination: a.destination, is_copy: a.copy || false }),
    },
    copy_file: {
      m: "POST",
      p: "/move_file",
      b: (a) => ({ source: a.source, destination: a.destination, is_copy: true }),
    },
    create_folder: { m: "POST", p: "/create_folder", b: (a) => ({ path: a.path }) },
    delete_file: { m: "POST", p: "/delete_file", b: (a) => ({ path: a.path }) },
    read_file: { m: "POST", p: "/read_file", b: (a) => ({ path: a.path }) },
    detect_faces: { m: "POST", p: "/detect-faces", b: (a) => ({ image_path: a.image_path, folder: a.folder }) },
    crop_face: { m: "POST", p: "/crop-face", b: (a) => ({ image_path: a.image_path, face_idx: a.face_idx }) },
    find_person: { m: "POST", p: "/find-person", b: (a) => ({ reference_image: a.reference_image, folder: a.folder }) },
    find_person_by_face: {
      m: "POST",
      p: "/find-person-by-face",
      b: (a) => ({ reference_image: a.reference_image, face_idx: a.face_idx, folder: a.folder }),
    },
    count_faces: {
      m: "POST",
      p: "/count-faces",
      b: (a) => ({ image_path: a.image_path, paths: a.paths, folder: a.folder }),
    },
    compare_faces: {
      m: "POST",
      p: "/compare-faces",
      b: (a) => ({
        image_path_1: a.image_path_1,
        face_idx_1: a.face_idx_1 || 0,
        image_path_2: a.image_path_2,
        face_idx_2: a.face_idx_2 || 0,
      }),
    },
    remember_face: {
      m: "POST",
      p: "/remember-face",
      b: (a) => ({ image_path: a.image_path, face_idx: a.face_idx || 0, name: a.name }),
    },
    forget_face: { m: "POST", p: "/forget-face", b: (a) => ({ name: a.name }) },
    ocr: { m: "POST", p: "/ocr", b: (a) => ({ path: a.path, folder: a.folder }) },
    analyze_data: {
      m: "POST",
      p: "/analyze-data",
      b: (a) => ({
        path: a.path,
        operation: a.operation || "describe",
        columns: a.columns || null,
        query: a.query || null,
        sheet: a.sheet || null,
      }),
    },
    index_file: { m: "POST", p: "/index-file", b: (a) => ({ path: a.path }) },
    write_file: {
      m: "POST",
      p: "/write-file",
      b: (a) => ({ path: a.path, content: a.content, append: a.append || false }),
    },
    generate_excel: {
      m: "POST",
      p: "/generate-excel",
      b: (a) => ({ path: a.path, data: a.data, sheet_name: a.sheet_name || "Sheet1", title: a.title || null }),
    },
    generate_chart: {
      m: "POST",
      p: "/generate-chart",
      b: (a) => ({
        data: a.data,
        chart_type: a.chart_type,
        title: a.title || "",
        xlabel: a.xlabel || "",
        ylabel: a.ylabel || "",
        output_path: a.output_path || null,
      }),
    },
    merge_pdf: { m: "POST", p: "/merge-pdf", b: (a) => ({ paths: a.paths, output_path: a.output_path }) },
    split_pdf: { m: "POST", p: "/split-pdf", b: (a) => ({ path: a.path, pages: a.pages, output_path: a.output_path }) },
    pdf_to_images: {
      m: "POST",
      p: "/pdf-to-images",
      b: (a) => ({ path: a.path, pages: a.pages || null, dpi: a.dpi || 150, output_folder: a.output_folder || null }),
    },
    images_to_pdf: { m: "POST", p: "/images-to-pdf", b: (a) => ({ paths: a.paths, output_path: a.output_path }) },
    compress_pdf: { m: "POST", p: "/compress-pdf", b: (a) => ({ path: a.path, output_path: a.output_path || null }) },
    add_page_numbers: {
      m: "POST",
      p: "/add-page-numbers",
      b: (a) => ({ path: a.path, output_path: a.output_path || null, position: a.position || "bottom-center", start: a.start || 1, format: a.format || "{n}" }),
    },
    pdf_to_word: { m: "POST", p: "/pdf-to-word", b: (a) => ({ path: a.path, output_path: a.output_path || null }) },
    organize_pdf: { m: "POST", p: "/organize-pdf", b: (a) => ({ path: a.path, pages: a.pages, output_path: a.output_path }) },
    pdf_to_excel: { m: "GET", p: (a) => `/pdf-to-excel?path=${enc(a.path)}${a.output_path ? "&output_path=" + enc(a.output_path) : ""}${a.pages ? "&pages=" + enc(a.pages) : ""}` },
    resize_image: {
      m: "POST",
      p: "/resize-image",
      b: (a) => ({
        path: a.path,
        width: a.width || null,
        height: a.height || null,
        quality: a.quality || 85,
        output_path: a.output_path || null,
      }),
    },
    convert_image: {
      m: "POST",
      p: "/convert-image",
      b: (a) => ({ path: a.path, format: a.format, output_path: a.output_path || null }),
    },
    crop_image: {
      m: "POST",
      p: "/crop-image",
      b: (a) => ({
        path: a.path,
        x: a.x,
        y: a.y,
        width: a.width,
        height: a.height,
        output_path: a.output_path || null,
      }),
    },
    image_metadata: { m: "POST", p: "/image-metadata", b: (a) => ({ path: a.path || null, folder: a.folder || null }) },
    compress_files: { m: "POST", p: "/compress-files", b: (a) => ({ paths: a.paths, output_path: a.output_path }) },
    extract_archive: {
      m: "POST",
      p: "/extract-archive",
      b: (a) => ({ path: a.path, output_path: a.output_path || null }),
    },
    download_url: { m: "POST", p: "/download-url", b: (a) => ({ url: a.url, save_path: a.save_path || null }) },
    find_duplicates: { m: "POST", p: "/find-duplicates", b: (a) => ({ folder: a.folder }) },
    batch_rename: {
      m: "POST",
      p: "/batch-rename",
      b: (a) => ({ folder: a.folder, pattern: a.pattern, replace: a.replace, dry_run: a.dry_run !== false }),
    },
    run_python: { m: "POST", p: "/run-python", b: (a) => ({ code: a.code, timeout: a.timeout || 30 }) },
    search_video: {
      m: "POST",
      p: "/search-video",
      b: (a) => ({ video_path: a.video_path, query: a.query, fps: a.fps || 1.0, limit: a.limit || 5 }),
    },
    extract_frame: { m: "POST", p: "/extract-frame", b: (a) => ({ video_path: a.video_path, seconds: a.seconds }) },
    transcribe_audio: { m: "POST", p: "/transcribe-audio", b: (a) => ({ path: a.path }) },
    search_audio: {
      m: "POST",
      p: "/search-audio",
      b: (a) => ({ audio_path: a.audio_path, query: a.query, limit: a.limit || 5 }),
    },
    extract_tables: {
      m: "POST",
      p: (a) => {
        const p = new URLSearchParams({ path: a.path });
        if (a.pages) p.set("pages", a.pages);
        return `/extract-tables?${p}`;
      },
      b: () => ({}),
    },
    watch_folder: { m: "POST", p: "/watch-folder", b: (a) => ({ path: a.folder }) },
    unwatch_folder: { m: "POST", p: "/unwatch-folder", b: (a) => ({ path: a.folder }) },
    score_photo: { m: "POST", p: "/score-photo", b: (a) => ({ path: a.path }) },
    cull_photos: {
      m: "POST",
      p: "/cull-photos",
      b: (a) => ({ folder: a.folder, keep_pct: a.keep_pct || 80, rejects_folder: a.rejects_folder || null }),
    },
    cull_status: { m: "GET", p: (a) => `/cull-photos/status?folder=${enc(a.folder)}${a.cancel ? "&cancel=true" : ""}` },
    suggest_categories: { m: "POST", p: "/suggest-categories", b: (a) => ({ folder: a.folder }) },
    group_photos: { m: "POST", p: "/group-photos", b: (a) => ({ folder: a.folder, categories: a.categories }) },
    group_status: {
      m: "GET",
      p: (a) => `/group-photos/status?folder=${enc(a.folder)}${a.cancel ? "&cancel=true" : ""}`,
    },
    // --- Google Workspace (gws-cli) ---
    gmail_send: { m: "POST", p: "/google/gmail-send", b: (a) => ({ to: a.to, subject: a.subject, body: a.body, attach: a.attach || null }) },
    gmail_search: { m: "GET", p: (a) => `/google/gmail-search?q=${enc(a.query || "")}&limit=${a.limit || 10}` },
    gmail_triage: { m: "GET", p: () => "/google/gmail-triage" },
    calendar_events: { m: "GET", p: (a) => `/google/calendar-events?days=${a.days || 7}${a.today ? "&today=true" : ""}` },
    calendar_create: {
      m: "POST",
      p: "/google/calendar-create",
      b: (a) => ({ summary: a.summary, start: a.start, end: a.end, location: a.location || null, description: a.description || null, attendees: a.attendees || null }),
    },
    drive_list: { m: "GET", p: (a) => `/google/drive-list?q=${enc(a.query || "")}&limit=${a.limit || 10}` },
    drive_upload: { m: "POST", p: (a) => `/google/drive-upload?path=${enc(a.path)}${a.name ? "&name=" + enc(a.name) : ""}`, b: () => ({}) },
  };
}

// --- Pre-validation: catch bad args before hitting the API (saves a round-trip) ---

// Tools that need a valid file path
const fileTools = [
  "read_file",
  "read_excel",
  "move_file",
  "copy_file",
  "delete_file",
  "ocr",
  "detect_faces",
  "crop_face",
  "find_person",
  "find_person_by_face",
  "resize_image",
  "convert_image",
  "crop_image",
  "merge_pdf",
  "split_pdf",
  "pdf_to_images",
  "compress_pdf",
  "add_page_numbers",
  "pdf_to_word",
  "organize_pdf",
  "pdf_to_excel",
  "index_file",
  "compare_faces",
  "remember_face",
  "transcribe_audio",
  "score_photo",
  "drive_upload",
];

// Tools that need a valid folder
const folderTools = [
  "list_files",
  "grep_files",
  "search_images_visual",
  "find_person",
  "find_person_by_face",
  "create_folder",
  "cull_photos",
  "group_photos",
];

function preValidate(name, args) {
  // File path validation
  const pathKey =
    name === "move_file" || name === "copy_file"
      ? "source"
      : name === "find_person" || name === "find_person_by_face"
        ? "reference_image"
        : name === "compare_faces"
          ? "image_path_1"
          : name === "crop_face" || name === "remember_face"
            ? "image_path"
            : "path";

  if (fileTools.includes(name) && args[pathKey]) {
    try {
      if (!existsSync(args[pathKey])) return `File not found: ${args[pathKey]}. Check the path and try again.`;
    } catch (_) {}
  }

  // Folder validation
  const folderKey = "folder";
  if (folderTools.includes(name) && args[folderKey] && name !== "create_folder") {
    try {
      if (!existsSync(args[folderKey])) return `Folder not found: ${args[folderKey]}. Check the path and try again.`;
      if (!statSync(args[folderKey]).isDirectory()) return `Not a folder: ${args[folderKey]}`;
    } catch (_) {}
  }

  // Array path validation (merge_pdf, images_to_pdf use paths[])
  if ((name === "merge_pdf" || name === "images_to_pdf") && Array.isArray(args.paths)) {
    for (const p of args.paths) {
      try {
        if (!existsSync(p)) return `File not found: ${p}. Check the path and try again.`;
      } catch (_) {}
    }
  }

  // compare_faces: validate both image paths
  if (name === "compare_faces" && args.image_path_2) {
    try {
      if (!existsSync(args.image_path_2)) return `File not found: ${args.image_path_2}. Check the path and try again.`;
    } catch (_) {}
  }

  // Video tools need valid video path
  if ((name === "search_video" || name === "extract_frame") && args.video_path) {
    try {
      if (!existsSync(args.video_path)) return `Video not found: ${args.video_path}. Check the path and try again.`;
    } catch (_) {}
  }

  // Audio search needs valid audio_path
  if (name === "search_audio" && args.audio_path) {
    try {
      if (!existsSync(args.audio_path))
        return `Audio file not found: ${args.audio_path}. Check the path and try again.`;
    } catch (_) {}
  }

  // Empty query check
  if (name === "search_documents" && (!args.query || !args.query.trim())) {
    return "Search query cannot be empty.";
  }

  return null; // All good
}

// --- Tool result summaries (Claude Code pattern: model trusts short summaries) ---
function summarizeToolResult(name, args, result) {
  if (!result) return null;
  if (result.error) return `${name}: ERROR — ${String(result.error).slice(0, 80)}`;
  switch (name) {
    case "search_documents": {
      const n = result.results?.length || result.total_items || 0;
      if (result.ambiguous_search) {
        return `search_documents: ambiguous — ${result.clarification_hint || "ask the user to narrow the result with a title, file name, date, person, location, or year"}`;
      }
      const top = Array.isArray(result.results) && result.results.length > 0 ? result.results[0] : null;
      const why = top?.why_matched ? String(top.why_matched).split(";")[0].trim() : "";
      const matchType = top?.match_type ? ` via ${top.match_type}` : "";
      const searchMode = result.search_explanation?.relaxed_lexical
        ? "relaxed lexical"
        : result.search_explanation?.enhanced_search_used
          ? "enhanced"
          : "lexical-first";
      const reason = why ? ` — top match${matchType}: ${why}` : "";
      return `search_documents: ${n} result(s) found (${searchMode})${reason}`;
    }
    case "read_document_overview": {
      const title = result.title || `document ${args?.document_id ?? ""}`;
      const sectionCount = result.top_sections?.length || 0;
      const factCount = result.facts?.length || 0;
      return `read_document_overview: ${title} — ${sectionCount} section preview(s), ${factCount} fact(s)`;
    }
    case "search_facts": {
      const n = result.count || result.results?.length || 0;
      return `search_facts: ${n} fact(s) found`;
    }
    case "search_images_visual": {
      if (result._ref)
        return `search_images_visual: ${result.total_items || "multiple"} results (stored as ${result._ref})`;
      const keys = Object.keys(result).filter((k) => !k.startsWith("_"));
      return `search_images_visual: results for ${keys.length} queries`;
    }
    case "list_files": {
      const n = result.total || result.total_items || result.showing || 0;
      return `list_files: ${n} item(s) in ${args?.folder || "folder"}`;
    }
    case "find_file": {
      const n = result.count || 0;
      return `find_file: ${n} file(s) matching '${args?.query || ""}'`;
    }
    case "search_generated_files": {
      const n = result.count || 0;
      return `search_generated_files: ${n} file(s) found`;
    }
    case "detect_faces": {
      const n = result.images_processed || result.face_count || 0;
      return result.images_processed ? `detect_faces: ${n} images processed` : `detect_faces: ${n} face(s) found`;
    }
    case "count_faces":
      return `count_faces: ${result.images_processed ? result.images_processed + " images counted" : result.count + " face(s)"}`;
    case "analyze_data": {
      const op = args?.operation || "?";
      return `analyze_data(${op}): ${result.shape ? result.shape[0] + " rows x " + result.shape[1] + " cols" : "done"}`;
    }
    case "find_person":
    case "find_person_by_face": {
      const n = result.count || result.matches?.length || 0;
      return `${name}: ${n} matching photo(s)`;
    }
    case "read_file":
      return `read_file: ${result.type || "text"} file loaded`;
    case "find_duplicates": {
      const n = result.groups?.length || result.total_items || 0;
      return `find_duplicates: ${n} duplicate group(s)`;
    }
    case "batch_move": {
      const moved = result.moved_count ?? 0;
      const skipped = result.skipped_count ?? 0;
      const errors = result.error_count ?? 0;
      const action = result.action || "moved";
      if (moved === 0) return `batch_move: WARNING — 0 files ${action}. ${skipped} skipped, ${errors} errors`;
      return `batch_move: ${moved} ${action}, ${skipped} skipped, ${errors} errors → ${args?.destination || "dest"}`;
    }
    case "move_file":
    case "copy_file": {
      const action = result.action || (name === "copy_file" ? "copied" : "moved");
      return `${name}: ${result.success ? action : "FAILED"} ${args?.source ? pathModule.basename(args.source) : "file"}`;
    }
    case "delete_file":
      return `delete_file: ${result.success ? "deleted" : "FAILED"} ${args?.path ? pathModule.basename(args.path) : "file"}`;
    case "write_file":
      return `write_file: ${result.success ? "created" : "FAILED"} ${result.path ? pathModule.basename(result.path) : "file"}`;
    case "create_folder":
      return `create_folder: ${result.already_existed ? "already existed" : "created"} ${result.path || "folder"}`;
    case "batch_rename": {
      const n = result.renamed_count ?? result.renamed ?? 0;
      return `batch_rename: ${n} renamed, ${result.error_count ?? 0} errors`;
    }
    case "search_video": {
      const n = result.results?.length || 0;
      return `search_video: ${n} matching moment(s) found`;
    }
    case "transcribe_audio":
      return `transcribe_audio: ${result.text ? result.text.length + " chars transcribed" : "done"}`;
    case "search_audio": {
      const n = result.results?.length || 0;
      return `search_audio: ${n} matching moment(s) found`;
    }
    case "compress_pdf":
      return `compress_pdf: ${result.reduction_percent ?? 0}% smaller — ${result.path ? pathModule.basename(result.path) : "file"}`;
    case "add_page_numbers":
      return `add_page_numbers: ${result.pages_numbered ?? 0} pages numbered`;
    case "pdf_to_word":
      return `pdf_to_word: ${result.pages_converted ?? 0} pages → ${result.path ? pathModule.basename(result.path) : "docx"}${result.ocr_pages ? ` (${result.ocr_pages} OCR'd)` : ""}`;
    case "organize_pdf":
      return `organize_pdf: ${result.output_pages ?? 0} pages → ${result.path ? pathModule.basename(result.path) : "pdf"}`;
    case "pdf_to_excel":
      return `pdf_to_excel: ${result.tables_exported ?? 0} table(s) → ${result.path ? pathModule.basename(result.path) : "xlsx"}`;
    case "score_photo":
      return `score_photo: ${result.total ?? "?"}${"/100"} — ${(result.reasoning || "").slice(0, 60)}`;
    case "cull_photos":
      return `cull_photos: ${result.started ? `started ${result.total_images} photos` : "FAILED"}`;
    case "cull_status": {
      if (result.status === "done")
        return `cull_status: done — kept ${result.kept}, rejected ${result.rejected}, report at ${result.report_path || "N/A"}`;
      if (result.status === "cancelled" || result.status === "cancelling")
        return `cull_status: cancelled after ${result.scored || 0}/${result.total || "?"} scored`;
      return `cull_status: ${result.status} — ${result.scored || 0}/${result.total || "?"} scored`;
    }
    case "suggest_categories":
      return `suggest_categories: ${result.categories ? `suggested ${result.categories.length} categories: ${result.categories.join(", ")}` : "FAILED"}`;
    case "group_photos":
      return `group_photos: ${result.started ? `started ${result.total_images} photos → ${result.categories?.length ?? "?"} categories` : "FAILED"}`;
    case "group_status": {
      if (result.status === "done")
        return `group_status: done — ${result.moved} grouped, report at ${result.report_path || "N/A"}`;
      if (result.status === "cancelled" || result.status === "cancelling")
        return `group_status: cancelled after ${result.classified || 0}/${result.total || "?"} classified`;
      return `group_status: ${result.status} — ${result.classified || 0}/${result.total || "?"} classified`;
    }
    case "gmail_send":
      return `gmail_send: ${result.error ? "FAILED" : `sent to ${args?.to || "recipient"}`}`;
    case "gmail_search":
      return `gmail_search: ${result.count ?? 0} email(s) found`;
    case "gmail_triage":
      return `gmail_triage: inbox summary loaded`;
    case "calendar_events":
      return `calendar_events: agenda loaded`;
    case "calendar_create":
      return `calendar_create: ${result.error ? "FAILED" : `event '${args?.summary || ""}' created`}`;
    case "drive_list":
      return `drive_list: ${result.count ?? 0} file(s) found`;
    case "drive_upload":
      return `drive_upload: ${result.error ? "FAILED" : "uploaded to Drive"}`;
    default:
      return `${name}: done`;
  }
}

module.exports = {
  CORE_TOOLS,
  TOOL_GROUPS,
  INTENT_KEYWORDS,
  SKILL_CATEGORIES,
  detectIntentCategories,
  getToolsForIntent,
  clearIntentCache,
  hasActiveIntent,
  TOOL_DECLARATIONS,
  buildToolRoutes,
  preValidate,
  summarizeToolResult,
  fileTools,
  folderTools,
};
