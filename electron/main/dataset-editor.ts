import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import type { FileHandle } from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import type {
  DatasetEditorRequest,
  DatasetEditorResult,
  DatasetFieldMapping,
  DatasetInspection,
  DatasetPreviewRow,
  DatasetTask,
} from "../types.js";

const DATA_EXTENSIONS = new Set([".json", ".jsonl", ".ndjson"]);
const MAX_JSON_DOCUMENT_BYTES = 256 * 1024 * 1024;
const MAX_ISSUES = 100;
const MAX_VALID_PREVIEW = 25;
const MAX_INVALID_PREVIEW = 25;
const MAX_EDITS = 100;
const TOKEN_SAMPLE_LIMIT = 100_000;
const DEDUPLICATION_SCAN_LIMIT = 250_000;

const emptyMapping: DatasetFieldMapping = {
  messages: "",
  prompt: "",
  context: "",
  response: "",
  chosen: "",
  rejected: "",
  reward: "",
  text: "",
};

const aliases = {
  messages: [
    "messages",
    "conversations",
    "conversation",
    "dialog",
    "dialogue",
    "chat",
    "turns",
  ],
  prompt: [
    "prompt",
    "instruction",
    "question",
    "query",
    "problem",
    "task",
    "request",
    "source",
    "src",
    "document",
    "article",
    "description",
    "user",
  ],
  context: ["input", "context", "passage", "background"],
  response: [
    "completion",
    "output",
    "response",
    "answer",
    "target",
    "tgt",
    "solution",
    "summary",
    "highlights",
    "assistant",
    "code",
  ],
  chosen: ["chosen", "preferred", "accepted", "winner", "response_j"],
  rejected: [
    "rejected",
    "non_preferred",
    "dispreferred",
    "unpreferred",
    "loser",
    "response_k",
  ],
  reward: ["reward", "score", "rating", "preference_score", "label"],
  text: ["text", "content", "corpus", "document", "article"],
} satisfies Record<keyof DatasetFieldMapping, string[]>;

const roleAliases: Record<string, string> = {
  assistant: "assistant",
  bot: "assistant",
  gpt: "assistant",
  model: "assistant",
  agent: "assistant",
  user: "user",
  human: "user",
  instruction: "user",
  question: "user",
  system: "system",
  developer: "system",
  tool: "tool",
  function: "tool",
  observation: "tool",
};

const mediaAliases: Record<string, "images" | "videos" | "audio"> = {
  image: "images",
  images: "images",
  image_url: "images",
  video: "videos",
  videos: "videos",
  video_url: "videos",
  audio: "audio",
  audios: "audio",
  audio_url: "audio",
};

type SourceFile = { file: string; split: "train" | "valid" | "test" };
type SourceRow = {
  id: string;
  source: string;
  split: "train" | "valid" | "test";
  line: number;
  row?: Record<string, unknown>;
  error?: string;
  blank?: boolean;
};
type Normalized = {
  task: Exclude<DatasetTask, "auto">;
  format: string;
  modalities: Set<string>;
  row: Record<string, unknown>;
};

function recordId(file: string, split: string, line: number) {
  return createHash("sha256")
    .update(`${path.resolve(file)}\0${split}\0${line}`)
    .digest("hex")
    .slice(0, 24);
}

function sha(value: unknown) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function cleanError(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function splitForFilename(name: string): SourceFile["split"] | null {
  const stem = path.basename(name, path.extname(name)).toLowerCase();
  if (/^(train|training)(?:[-_.].*)?$/.test(stem)) return "train";
  if (/^(valid|validation|validate|dev|eval)(?:[-_.].*)?$/.test(stem))
    return "valid";
  if (/^(test|testing)(?:[-_.].*)?$/.test(stem)) return "test";
  return null;
}

async function sourceFiles(sourceValue: string): Promise<SourceFile[]> {
  if (!sourceValue || !path.isAbsolute(sourceValue))
    throw new Error("Dataset source must be an absolute file or folder path");
  const source = path.resolve(sourceValue);
  const details = await fs.stat(source).catch(() => null);
  if (!details) throw new Error("Dataset source does not exist");
  if (details.isFile()) {
    if (!DATA_EXTENSIONS.has(path.extname(source).toLowerCase()))
      throw new Error("Choose a JSON, JSONL, or NDJSON dataset file");
    return [{ file: source, split: "train" }];
  }
  if (!details.isDirectory())
    throw new Error("Dataset source is not a file or folder");
  const entries = (await fs.readdir(source, { withFileTypes: true }))
    .filter(
      (entry) =>
        entry.isFile() &&
        DATA_EXTENSIONS.has(path.extname(entry.name).toLowerCase()),
    )
    .map((entry) => path.join(source, entry.name))
    .sort((left, right) => left.localeCompare(right));
  if (!entries.length)
    throw new Error(
      "The selected folder contains no JSON, JSONL, or NDJSON files",
    );
  const named = entries
    .map((file) => ({ file, split: splitForFilename(file) }))
    .filter((entry): entry is SourceFile => entry.split !== null);
  if (named.length) return named;
  return entries.map((file) => ({ file, split: "train" }));
}

function recordsFromDocument(
  value: unknown,
  fallbackSplit: SourceFile["split"],
) {
  const result: Array<{ split: SourceFile["split"]; row: unknown }> = [];
  const container =
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;
  const splitContainers: Array<[string, SourceFile["split"]]> = [
    ["train", "train"],
    ["training", "train"],
    ["valid", "valid"],
    ["validation", "valid"],
    ["dev", "valid"],
    ["eval", "valid"],
    ["test", "test"],
  ];
  let foundSplit = false;
  if (container) {
    for (const [key, split] of splitContainers) {
      const rows = container[key];
      if (!Array.isArray(rows)) continue;
      foundSplit = true;
      rows.forEach((row) => result.push({ split, row }));
    }
  }
  if (foundSplit) return result;
  const records = Array.isArray(value)
    ? value
    : container
      ? ["data", "records", "examples", "items", "rows", "instances"]
          .map((key) => container[key])
          .find(Array.isArray) || [value]
      : [value];
  for (const row of records as unknown[])
    result.push({ split: fallbackSplit, row });
  return result;
}

async function* rowsFromFile(source: SourceFile): AsyncGenerator<SourceRow> {
  const extension = path.extname(source.file).toLowerCase();
  if (extension === ".json") {
    const size = (await fs.stat(source.file)).size;
    if (size > MAX_JSON_DOCUMENT_BYTES)
      throw new Error(
        `${path.basename(source.file)} is too large for JSON-array loading; convert it to streaming JSONL`,
      );
    let parsed: unknown;
    try {
      parsed = JSON.parse(await fs.readFile(source.file, "utf8"));
    } catch (error) {
      throw new Error(`Invalid JSON in ${source.file}: ${cleanError(error)}`);
    }
    let line = 0;
    for (const value of recordsFromDocument(parsed, source.split)) {
      line += 1;
      const id = recordId(source.file, value.split, line);
      if (
        !value.row ||
        typeof value.row !== "object" ||
        Array.isArray(value.row)
      ) {
        yield {
          id,
          source: source.file,
          split: value.split,
          line,
          error: "Dataset row is not a JSON object",
        };
      } else {
        yield {
          id,
          source: source.file,
          split: value.split,
          line,
          row: value.row as Record<string, unknown>,
        };
      }
    }
    return;
  }

  const input = readline.createInterface({
    input: createReadStream(source.file, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  let line = 0;
  try {
    for await (const value of input) {
      line += 1;
      const id = recordId(source.file, source.split, line);
      if (!value.trim()) {
        yield {
          id,
          source: source.file,
          split: source.split,
          line,
          error: "Blank row",
          blank: true,
        };
        continue;
      }
      try {
        const parsed = JSON.parse(value) as unknown;
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
          throw new Error("Dataset row is not a JSON object");
        yield {
          id,
          source: source.file,
          split: source.split,
          line,
          row: parsed as Record<string, unknown>,
        };
      } catch (error) {
        yield {
          id,
          source: source.file,
          split: source.split,
          line,
          error: cleanError(error),
        };
      }
    }
  } finally {
    input.close();
  }
}

async function* sourceRows(source: string): AsyncGenerator<SourceRow> {
  for (const file of await sourceFiles(source)) yield* rowsFromFile(file);
}

function dotted(value: Record<string, unknown>, field: string): unknown {
  if (!field) return undefined;
  let current: unknown = value;
  for (const part of field.split(".")) {
    if (!current || typeof current !== "object" || Array.isArray(current))
      return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function firstValue(row: Record<string, unknown>, fields: string[]) {
  for (const field of fields) {
    const value = dotted(row, field);
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function asText(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" || typeof value === "boolean")
    return String(value);
  if (Array.isArray(value)) {
    const strings = value.map(asText).filter(Boolean);
    return strings.join("\n");
  }
  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    for (const key of ["text", "content", "answer", "value", "output"])
      if (object[key] !== undefined) {
        const text = asText(object[key]);
        if (text) return text;
      }
  }
  return "";
}

function normalizeContent(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (!Array.isArray(value)) return asText(value);
  const parts: string[] = [];
  for (const part of value) {
    if (typeof part === "string") {
      if (part.trim()) parts.push(part.trim());
      continue;
    }
    if (!part || typeof part !== "object") continue;
    const item = part as Record<string, unknown>;
    const type = String(item.type || "text").toLowerCase();
    if (["image", "image_url", "input_image"].includes(type))
      parts.push("<image>");
    else if (["video", "video_url", "input_video"].includes(type))
      parts.push("<video>");
    else if (["audio", "audio_url", "input_audio"].includes(type))
      parts.push("<audio>");
    else {
      const text = asText(
        item.text ?? item.content ?? item.output ?? item.value,
      );
      if (text) parts.push(text);
    }
  }
  return parts.join("\n");
}

function normalizeMessages(value: unknown) {
  if (!Array.isArray(value) || !value.length)
    throw new Error("Conversation field must be a non-empty array");
  const messages = value.map((raw, index) => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw))
      throw new Error(`Message ${index + 1} is not an object`);
    const item = raw as Record<string, unknown>;
    const roleValue = String(
      item.role ?? item.from ?? item.speaker ?? item.author ?? "",
    )
      .trim()
      .toLowerCase();
    const role = roleAliases[roleValue];
    if (!role)
      throw new Error(
        `Message ${index + 1} has unknown role ${roleValue || "(empty)"}`,
      );
    const content = normalizeContent(
      item.content ?? item.value ?? item.text ?? item.message,
    );
    if (
      !content &&
      item.tool_calls === undefined &&
      item.function_call === undefined
    )
      throw new Error(`Message ${index + 1} has no content`);
    const message: Record<string, unknown> = { role, content };
    if (item.tool_calls !== undefined) message.tool_calls = item.tool_calls;
    else if (item.function_call !== undefined)
      message.tool_calls = item.function_call;
    if (typeof item.name === "string" && item.name.trim())
      message.name = item.name.trim();
    return message;
  });
  if (!messages.some((message) => message.role === "assistant"))
    throw new Error("Conversation has no assistant response");
  return messages;
}

function localMedia(value: unknown, source: string): unknown {
  if (Array.isArray(value))
    return value.map((item) => localMedia(item, source));
  if (value && typeof value === "object") {
    const item = { ...(value as Record<string, unknown>) };
    if (item.path !== undefined) item.path = localMedia(item.path, source);
    if (item.url !== undefined) item.url = localMedia(item.url, source);
    return item;
  }
  if (typeof value !== "string" || !value.trim()) return value;
  const text = value.trim();
  if (text.startsWith("data:") || text.startsWith("file:")) return text;
  if (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(text) || text.startsWith("//"))
    throw new Error("Remote media is disabled; use a local file path");
  return path.isAbsolute(text)
    ? path.normalize(text)
    : path.resolve(path.dirname(source), text);
}

function attachMedia(
  output: Record<string, unknown>,
  input: Record<string, unknown>,
  source: string,
) {
  const modalities = new Set<string>(["text"]);
  for (const [field, canonical] of Object.entries(mediaAliases)) {
    if (input[field] === undefined || input[field] === null) continue;
    const resolved = localMedia(input[field], source);
    const current = output[canonical];
    const values = Array.isArray(resolved) ? resolved : [resolved];
    output[canonical] = [...(Array.isArray(current) ? current : []), ...values];
    modalities.add(
      canonical === "images"
        ? "image"
        : canonical === "videos"
          ? "video"
          : "audio",
    );
  }
  return modalities;
}

function pair(
  prompt: unknown,
  response: unknown,
  format: string,
  original: Record<string, unknown>,
  source: string,
  context?: unknown,
): Normalized {
  const promptText = asText(prompt);
  const responseText = asText(response);
  if (!promptText) throw new Error("Prompt field is empty");
  if (!responseText) throw new Error("Response field is empty");
  const contextText = asText(context);
  const user = contextText
    ? `${promptText}\n\nContext:\n${contextText}`
    : promptText;
  const row: Record<string, unknown> = {
    messages: [
      { role: "user", content: user },
      { role: "assistant", content: responseText },
    ],
  };
  const modalities = attachMedia(row, original, source);
  return { task: "supervised", format, modalities, row };
}

function inferMapping(fields: string[]): DatasetFieldMapping {
  const lower = new Map(fields.map((field) => [field.toLowerCase(), field]));
  const mapping = { ...emptyMapping };
  for (const key of Object.keys(mapping) as Array<keyof DatasetFieldMapping>) {
    mapping[key] =
      aliases[key].map((candidate) => lower.get(candidate)).find(Boolean) || "";
  }
  return mapping;
}

function normalizeRow(
  original: Record<string, unknown>,
  source: string,
  task: DatasetTask = "auto",
  mapping: DatasetFieldMapping = emptyMapping,
): Normalized {
  const fields = Object.keys(original);
  const inferred = inferMapping(fields);
  const selected = Object.fromEntries(
    Object.keys(emptyMapping).map((key) => [
      key,
      mapping[key as keyof DatasetFieldMapping] ||
        inferred[key as keyof DatasetFieldMapping],
    ]),
  ) as DatasetFieldMapping;
  const value = (key: keyof DatasetFieldMapping) =>
    dotted(original, selected[key]);
  const preferred = task === "auto" ? "" : task;

  const chosen = value("chosen");
  const rejected = value("rejected");
  if (
    preferred === "preference" ||
    (!preferred && chosen != null && rejected != null)
  ) {
    const prompt = asText(value("prompt"));
    const chosenText = asText(chosen);
    const rejectedText = asText(rejected);
    if (!chosenText || !rejectedText)
      throw new Error("Preference rows need chosen and rejected responses");
    const row: Record<string, unknown> = {
      prompt,
      chosen: chosenText,
      rejected: rejectedText,
    };
    return {
      task: "preference",
      format: "preference",
      modalities: attachMedia(row, original, source),
      row,
    };
  }

  const rewardValue = value("reward");
  if (
    preferred === "reward" ||
    (!preferred && rewardValue != null && value("response") != null)
  ) {
    const reward = Number(rewardValue);
    if (!Number.isFinite(reward))
      throw new Error("Reward field must be numeric");
    const prompt = asText(value("prompt"));
    const response = asText(value("response"));
    if (!response) throw new Error("Reward rows need a response");
    const row: Record<string, unknown> = { prompt, response, reward };
    return {
      task: "reward",
      format: "reward",
      modalities: attachMedia(row, original, source),
      row,
    };
  }

  if (preferred === "text") {
    const text = asText(value("text"));
    if (!text) throw new Error("Text field is empty");
    return {
      task: "text",
      format: "text",
      modalities: new Set(["text"]),
      row: { text },
    };
  }

  const messagesValue = value("messages");
  if (messagesValue != null) {
    const row: Record<string, unknown> = {
      messages: normalizeMessages(messagesValue),
    };
    if (original.tools !== undefined) row.tools = original.tools;
    return {
      task: "supervised",
      format: selected.messages === "messages" ? "chat" : "conversation",
      modalities: attachMedia(row, original, source),
      row,
    };
  }

  if (
    original.system !== undefined &&
    original.user !== undefined &&
    original.assistant !== undefined
  ) {
    const system = asText(original.system);
    const prompt = asText(original.user);
    const response = asText(original.assistant);
    if (!prompt || !response)
      throw new Error("System/user/assistant row is incomplete");
    const row: Record<string, unknown> = {
      messages: [
        ...(system ? [{ role: "system", content: system }] : []),
        { role: "user", content: prompt },
        { role: "assistant", content: response },
      ],
    };
    return {
      task: "supervised",
      format: "role-columns",
      modalities: attachMedia(row, original, source),
      row,
    };
  }

  let prompt = value("prompt");
  let response = value("response");
  const context = value("context");
  if (original.translation && typeof original.translation === "object") {
    const translations = Object.entries(
      original.translation as Record<string, unknown>,
    );
    if (translations.length >= 2) {
      prompt = `Translate from ${translations[0][0]} to ${translations[1][0]}:\n${asText(translations[0][1])}`;
      response = translations[1][1];
    }
  }
  if (original.answers && !response) response = original.answers;
  if (prompt != null && response != null)
    return pair(
      prompt,
      response,
      selected.prompt && selected.response ? "mapped-pair" : "pair",
      original,
      source,
      context,
    );

  const text = asText(value("text"));
  if (text && !preferred)
    return {
      task: "text",
      format: "text",
      modalities: new Set(["text"]),
      row: { text },
    };
  throw new Error(
    "Could not find a conversation, prompt/response, preference, reward, or text layout",
  );
}

function collectFields(
  value: Record<string, unknown>,
  counts: Map<string, number>,
  prefix = "",
  depth = 0,
) {
  for (const [key, child] of Object.entries(value)) {
    const field = prefix ? `${prefix}.${key}` : key;
    counts.set(field, (counts.get(field) || 0) + 1);
    if (
      depth < 1 &&
      child &&
      typeof child === "object" &&
      !Array.isArray(child)
    )
      collectFields(child as Record<string, unknown>, counts, field, depth + 1);
  }
}

function estimateTokens(value: unknown, modalities: Set<string>) {
  const text = JSON.stringify(value);
  const mediaTokens =
    [...modalities].filter((item) => item !== "text").length * 256;
  return Math.max(1, Math.ceil(text.length / 4) + mediaTokens);
}

function percentile(sorted: number[], fraction: number) {
  if (!sorted.length) return 0;
  return sorted[
    Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))
  ];
}

function tokenLimit(value: number) {
  const target = Math.max(64, value);
  return Math.min(16_384, 2 ** Math.ceil(Math.log2(target)));
}

export async function inspectDataset(
  source: string,
): Promise<DatasetInspection> {
  const files = await sourceFiles(source);
  const fields = new Map<string, number>();
  const formats = new Set<string>();
  const modalities = new Set<string>(["text"]);
  const tasks = new Map<Exclude<DatasetTask, "auto">, number>();
  const issues: DatasetInspection["issues"] = [];
  const validPreview: DatasetPreviewRow[] = [];
  const invalidPreview: DatasetPreviewRow[] = [];
  const lengths: number[] = [];
  const hashes = new Set<string>();
  let totalRows = 0;
  let validRows = 0;
  let invalidRows = 0;
  let blankRows = 0;
  let duplicateRows = 0;
  for await (const entry of sourceRows(source)) {
    totalRows += 1;
    if (entry.blank) blankRows += 1;
    if (!entry.row) {
      invalidRows += 1;
      const message = entry.error || "Invalid dataset row";
      if (issues.length < MAX_ISSUES)
        issues.push({
          id: entry.id,
          split: entry.split,
          line: entry.line,
          source: entry.source,
          message,
        });
      if (invalidPreview.length < MAX_INVALID_PREVIEW)
        invalidPreview.push({
          id: entry.id,
          split: entry.split,
          line: entry.line,
          source: entry.source,
          status: "invalid",
          format: "invalid",
          estimatedTokens: 0,
          raw: {},
          error: message,
        });
      continue;
    }
    collectFields(entry.row, fields);
    if (hashes.size < DEDUPLICATION_SCAN_LIMIT) {
      const digest = sha(entry.row);
      if (hashes.has(digest)) duplicateRows += 1;
      else hashes.add(digest);
    }
    try {
      const normalized = normalizeRow(entry.row, entry.source);
      validRows += 1;
      formats.add(normalized.format);
      normalized.modalities.forEach((item) => modalities.add(item));
      tasks.set(normalized.task, (tasks.get(normalized.task) || 0) + 1);
      const estimatedTokens = estimateTokens(
        normalized.row,
        normalized.modalities,
      );
      if (lengths.length < TOKEN_SAMPLE_LIMIT) lengths.push(estimatedTokens);
      if (validPreview.length < MAX_VALID_PREVIEW)
        validPreview.push({
          id: entry.id,
          split: entry.split,
          line: entry.line,
          source: entry.source,
          status: "valid",
          format: normalized.format,
          estimatedTokens,
          raw: entry.row,
          normalized: normalized.row,
        });
    } catch (error) {
      invalidRows += 1;
      const message = cleanError(error);
      if (issues.length < MAX_ISSUES)
        issues.push({
          id: entry.id,
          split: entry.split,
          line: entry.line,
          source: entry.source,
          message,
        });
      if (invalidPreview.length < MAX_INVALID_PREVIEW)
        invalidPreview.push({
          id: entry.id,
          split: entry.split,
          line: entry.line,
          source: entry.source,
          status: "invalid",
          format: "unrecognized",
          estimatedTokens: 0,
          raw: entry.row,
          error: message,
        });
    }
  }
  if (!totalRows) throw new Error("Dataset contains no rows");
  const orderedFields = [...fields.entries()].sort(
    (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
  );
  const orderedTasks = [...tasks.entries()].sort(
    (left, right) => right[1] - left[1],
  );
  lengths.sort((left, right) => left - right);
  const p95Tokens = percentile(lengths, 0.95);
  return {
    source: path.resolve(source),
    files: files.map((item) => item.file),
    totalRows,
    validRows,
    invalidRows,
    blankRows,
    duplicateRows,
    formats: [...formats].sort(),
    modalities: [...modalities].sort(),
    fields: orderedFields.map(([field]) => field),
    fieldCounts: Object.fromEntries(orderedFields),
    recommendedTask: orderedTasks[0]?.[0] || "supervised",
    recommendedTokenLimit: tokenLimit(p95Tokens || 64),
    medianTokens: percentile(lengths, 0.5),
    p95Tokens,
    mapping: inferMapping(orderedFields.map(([field]) => field)),
    issues,
    preview: [...invalidPreview, ...validPreview].slice(
      0,
      MAX_INVALID_PREVIEW + MAX_VALID_PREVIEW,
    ),
    truncatedIssues: invalidRows > issues.length,
  };
}

function deepTrim(value: unknown): unknown {
  if (typeof value === "string") return value.trim();
  if (Array.isArray(value)) return value.map(deepTrim);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, child]) => [
      key,
      deepTrim(child),
    ]),
  );
}

function validateSaveRequest(input: DatasetEditorRequest) {
  if (!input || typeof input !== "object")
    throw new Error("Invalid data editor request");
  if (!path.isAbsolute(input.outputDirectory))
    throw new Error("Clean dataset location must be an absolute path");
  if (
    !["auto", "supervised", "preference", "reward", "text"].includes(input.task)
  )
    throw new Error("Invalid dataset task");
  for (const [label, value] of [
    ["Validation percentage", input.validationPercent],
    ["Test percentage", input.testPercent],
  ] as const)
    if (!Number.isFinite(value) || value < 0 || value > 40)
      throw new Error(`${label} must be between 0 and 40`);
  if (input.validationPercent + input.testPercent > 50)
    throw new Error(
      "Validation and test percentages cannot exceed 50% together",
    );
  if (!Array.isArray(input.edits) || input.edits.length > MAX_EDITS)
    throw new Error(`At most ${MAX_EDITS} preview rows can be edited at once`);
}

function assignedSplit(
  entry: SourceRow,
  validationPercent: number,
  testPercent: number,
) {
  if (entry.split !== "train" || entry.line === 1) return entry.split;
  const bucket = Number.parseInt(entry.id.slice(0, 8), 16) % 10_000;
  if (bucket < testPercent * 100) return "test" as const;
  if (bucket < (testPercent + validationPercent) * 100) return "valid" as const;
  return "train" as const;
}

export async function saveDataset(
  input: DatasetEditorRequest,
): Promise<DatasetEditorResult> {
  validateSaveRequest(input);
  const source = path.resolve(input.source);
  const output = path.resolve(input.outputDirectory);
  if (output === source || output.startsWith(`${source}${path.sep}`))
    throw new Error("Save the clean copy outside the source dataset");
  if (await fs.stat(output).catch(() => null))
    throw new Error(
      "The clean dataset folder already exists; choose a new name",
    );
  const parent = path.dirname(output);
  await fs.mkdir(parent, { recursive: true, mode: 0o700 });
  const temporary = path.join(
    parent,
    `.${path.basename(output)}.tmp-${randomUUID()}`,
  );
  const edits = new Map(input.edits.map((edit) => [edit.id, edit.row]));
  const handles = new Map<string, FileHandle>();
  const hashes = new Set<string>();
  const splitCounts: Record<string, number> = { train: 0, valid: 0, test: 0 };
  let writtenRows = 0;
  let skippedRows = 0;
  let duplicateRows = 0;
  try {
    await fs.mkdir(temporary, { recursive: false, mode: 0o700 });
    for await (const entry of sourceRows(source)) {
      const raw = edits.get(entry.id) || entry.row;
      if (!raw) {
        if (input.skipInvalid) {
          skippedRows += 1;
          continue;
        }
        throw new Error(
          `${path.basename(entry.source)}:${entry.line}: ${entry.error || "Invalid row"}`,
        );
      }
      try {
        const normalized = normalizeRow(
          raw,
          entry.source,
          input.task,
          input.mapping,
        );
        const row = (
          input.trimWhitespace ? deepTrim(normalized.row) : normalized.row
        ) as Record<string, unknown>;
        const serialized = JSON.stringify(row);
        const digest = createHash("sha256").update(serialized).digest("hex");
        if (input.removeDuplicates && hashes.has(digest)) {
          duplicateRows += 1;
          continue;
        }
        hashes.add(digest);
        const split = assignedSplit(
          entry,
          input.validationPercent,
          input.testPercent,
        );
        let handle = handles.get(split);
        if (!handle) {
          handle = await fs.open(
            path.join(temporary, `${split}.jsonl`),
            "wx",
            0o600,
          );
          handles.set(split, handle);
        }
        await handle.write(`${serialized}\n`);
        splitCounts[split] += 1;
        writtenRows += 1;
      } catch (error) {
        if (input.skipInvalid) {
          skippedRows += 1;
          continue;
        }
        throw new Error(
          `${path.basename(entry.source)}:${entry.line}: ${cleanError(error)}`,
        );
      }
    }
    for (const handle of handles.values()) await handle.close();
    handles.clear();
    if (!splitCounts.train)
      throw new Error("The cleaned dataset has no training rows");
    await fs.writeFile(
      path.join(temporary, "dataset.json"),
      `${JSON.stringify(
        {
          schemaVersion: 1,
          source,
          task: input.task,
          mapping: input.mapping,
          rows: writtenRows,
          skippedRows,
          duplicateRows,
          splits: splitCounts,
        },
        null,
        2,
      )}\n`,
      { mode: 0o600 },
    );
    await fs.rename(temporary, output);
    return {
      outputDirectory: output,
      writtenRows,
      skippedRows,
      duplicateRows,
      splitCounts,
      inspection: await inspectDataset(output),
    };
  } catch (error) {
    for (const handle of handles.values())
      await handle.close().catch(() => undefined);
    await fs.rm(temporary, { recursive: true, force: true });
    throw error;
  }
}
