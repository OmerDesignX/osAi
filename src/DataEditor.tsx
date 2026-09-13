import { useEffect, useMemo, useState } from "react";
import feather from "feather-icons";
import type {
  DatasetEditorResult,
  DatasetFieldMapping,
  DatasetInspection,
  DatasetPreviewRow,
  DatasetTask,
} from "./types.js";

type IconName = keyof typeof feather.icons;

function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const markup = feather.icons[name].toSvg({
    width: size,
    height: size,
    "stroke-width": 1.8,
    "aria-hidden": "true",
  });
  return <span className="icon" dangerouslySetInnerHTML={{ __html: markup }} />;
}

const fieldLabels: Array<[keyof DatasetFieldMapping, string]> = [
  ["messages", "Conversation"],
  ["prompt", "Prompt / instruction"],
  ["context", "Optional context"],
  ["response", "Response / answer"],
  ["chosen", "Chosen response"],
  ["rejected", "Rejected response"],
  ["reward", "Reward / score"],
  ["text", "Raw text"],
];

const taskLabels: Record<DatasetTask, string> = {
  auto: "Auto-detect each row",
  supervised: "Supervised fine-tuning",
  preference: "Preference pairs",
  reward: "Responses with rewards",
  text: "Raw language modelling",
};

function readableError(error: unknown) {
  return (error instanceof Error ? error.message : String(error))
    .replace(/^Error invoking remote method '[^']+':\s*(?:Error:\s*)?/i, "")
    .replace(/^Error:\s*/i, "")
    .trim();
}

function pathParts(source: string) {
  const separator = source.includes("\\") ? "\\" : "/";
  const trimmed = source.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  const parent = index >= 0 ? trimmed.slice(0, index) : trimmed;
  const name = (index >= 0 ? trimmed.slice(index + 1) : trimmed).replace(
    /\.(jsonl|ndjson|json)$/i,
    "",
  );
  return { separator, parent, name: name || "dataset" };
}

function suggestedOutput(source: string) {
  if (!source) return "";
  const { separator, parent, name } = pathParts(source);
  const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 13);
  return `${parent}${separator}${name}-osai-clean-${stamp}`;
}

function childPath(parent: string, name: string) {
  const separator = parent.includes("\\") ? "\\" : "/";
  return `${parent.replace(/[\\/]+$/, "")}${separator}${name}`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat().format(value);
}

function mappingFields(task: Exclude<DatasetTask, "auto">) {
  if (task === "preference") return ["prompt", "context", "chosen", "rejected"];
  if (task === "reward") return ["prompt", "context", "response", "reward"];
  if (task === "text") return ["text"];
  return ["messages", "prompt", "context", "response"];
}

export function DataEditor({
  initialSource,
  onUseForFineTune,
  onUseForAlignment,
  onError,
}: {
  initialSource: string;
  onUseForFineTune(source: string, tokenLimit: number): void;
  onUseForAlignment(source: string, tokenLimit: number): void;
  onError(message: string): void;
}) {
  const [source, setSource] = useState(initialSource);
  const [outputDirectory, setOutputDirectory] = useState(
    suggestedOutput(initialSource),
  );
  const [inspection, setInspection] = useState<DatasetInspection | null>(null);
  const [task, setTask] = useState<DatasetTask>("auto");
  const [mapping, setMapping] = useState<DatasetFieldMapping | null>(null);
  const [selectedRowId, setSelectedRowId] = useState("");
  const [editorText, setEditorText] = useState("");
  const [edits, setEdits] = useState<Record<string, Record<string, unknown>>>(
    {},
  );
  const [trimWhitespace, setTrimWhitespace] = useState(true);
  const [removeDuplicates, setRemoveDuplicates] = useState(true);
  const [skipInvalid, setSkipInvalid] = useState(false);
  const [validationPercent, setValidationPercent] = useState(5);
  const [testPercent, setTestPercent] = useState(0);
  const [busy, setBusy] = useState<"" | "inspect" | "save">("");
  const [result, setResult] = useState<DatasetEditorResult | null>(null);
  const [localError, setLocalError] = useState("");

  useEffect(() => {
    if (!initialSource) return;
    setSource(initialSource);
    setOutputDirectory(suggestedOutput(initialSource));
    setInspection(null);
    setMapping(null);
    setSelectedRowId("");
    setEdits({});
    setResult(null);
  }, [initialSource]);

  const effectiveTask =
    task === "auto" ? inspection?.recommendedTask || "supervised" : task;
  const selectedRow = inspection?.preview.find(
    (row) => row.id === selectedRowId,
  );
  const visibleMapping = useMemo(
    () => new Set(mappingFields(effectiveTask)),
    [effectiveTask],
  );

  const reportError = (error: unknown) => {
    const message = readableError(error);
    setLocalError(message);
    onError(message);
  };

  const inspect = async (nextSource = source) => {
    if (!nextSource.trim()) {
      reportError("Choose a dataset before inspecting it");
      return;
    }
    setBusy("inspect");
    setLocalError("");
    setResult(null);
    try {
      const next = await window.osai.inspectDataset(nextSource.trim());
      setSource(next.source);
      setInspection(next);
      setMapping(next.mapping);
      setTask("auto");
      setOutputDirectory(suggestedOutput(next.source));
      setEdits({});
      const first = next.preview[0];
      setSelectedRowId(first?.id || "");
      setEditorText(first ? JSON.stringify(first.raw, null, 2) : "");
    } catch (error) {
      setInspection(null);
      setMapping(null);
      reportError(error);
    } finally {
      setBusy("");
    }
  };

  const chooseSource = async () => {
    const value = await window.osai.chooseDataset(
      "Choose data to inspect or repair",
    );
    if (!value) return;
    setSource(value);
    await inspect(value);
  };

  const chooseOutput = async () => {
    const parent = await window.osai.chooseDirectory(
      "Choose where to save the clean dataset",
    );
    if (!parent) return;
    const name = `${pathParts(source).name}-osai-clean-${Date.now()}`;
    setOutputDirectory(childPath(parent, name));
  };

  const selectRow = (row: DatasetPreviewRow) => {
    setSelectedRowId(row.id);
    setEditorText(JSON.stringify(edits[row.id] || row.raw, null, 2));
    setLocalError("");
  };

  const applyRowEdit = () => {
    if (!selectedRow) return;
    try {
      const parsed = JSON.parse(editorText) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
        throw new Error("Edited rows must be JSON objects");
      setEdits((current) => ({
        ...current,
        [selectedRow.id]: parsed as Record<string, unknown>,
      }));
      setLocalError("");
    } catch (error) {
      reportError(error);
    }
  };

  const resetMapping = () => {
    if (!inspection) return;
    setTask("auto");
    setMapping(inspection.mapping);
  };

  const save = async () => {
    if (!inspection || !mapping) return;
    setBusy("save");
    setLocalError("");
    try {
      const saved = await window.osai.saveDataset({
        source: inspection.source,
        outputDirectory,
        task,
        mapping,
        trimWhitespace,
        removeDuplicates,
        skipInvalid,
        validationPercent,
        testPercent,
        edits: Object.entries(edits).map(([id, row]) => ({ id, row })),
      });
      setResult(saved);
      setOutputDirectory(saved.outputDirectory);
    } catch (error) {
      reportError(error);
    } finally {
      setBusy("");
    }
  };

  return (
    <div className="data-editor-view">
      <header className="data-editor-header">
        <div>
          <span className="data-editor-tag">Local data workspace</span>
          <h2>Data Editor</h2>
          <p>
            Inspect, map, repair and export training data without sending it
            anywhere.
          </p>
        </div>
        <button
          type="button"
          className="primary-button"
          onClick={() => void chooseSource()}
          disabled={Boolean(busy)}
        >
          <Icon name="folder" />
          Choose data
        </button>
      </header>

      <div className="data-source-strip">
        <label className="field path-field">
          <span>Dataset file or folder</span>
          <div>
            <input
              value={source}
              placeholder="Choose JSON, JSONL, NDJSON, or a split folder"
              spellCheck={false}
              onChange={(event) => setSource(event.target.value)}
            />
            <button
              type="button"
              className="icon-button"
              aria-label="Choose dataset"
              onClick={() => void chooseSource()}
            >
              <Icon name="folder" />
            </button>
          </div>
        </label>
        <button
          type="button"
          className="quiet-button data-inspect-button"
          onClick={() => void inspect()}
          disabled={Boolean(busy) || !source.trim()}
        >
          <Icon name={busy === "inspect" ? "loader" : "search"} />
          {busy === "inspect" ? "Inspecting…" : "Inspect"}
        </button>
      </div>

      {!inspection ? (
        <div className="data-editor-empty">
          <Icon name="database" size={26} />
          <h3>Choose a dataset to begin</h3>
          <p>
            Large JSONL files are streamed. JSON arrays, split folders and
            common Hugging Face-style containers are detected automatically.
          </p>
        </div>
      ) : (
        <div className="data-editor-grid">
          <aside className="data-editor-sidebar">
            <section className="data-card data-health-card">
              <div className="data-card-heading">
                <div>
                  <span>Dataset health</span>
                  <small>{inspection.files.length} local file(s)</small>
                </div>
                <strong
                  className={inspection.invalidRows ? "has-issues" : "healthy"}
                >
                  {inspection.invalidRows ? "Needs review" : "Ready"}
                </strong>
              </div>
              <div className="data-stat-grid">
                <div>
                  <b>{formatNumber(inspection.totalRows)}</b>
                  <span>Rows</span>
                </div>
                <div>
                  <b>{formatNumber(inspection.validRows)}</b>
                  <span>Recognized</span>
                </div>
                <div>
                  <b>{formatNumber(inspection.invalidRows)}</b>
                  <span>Issues</span>
                </div>
                <div>
                  <b>{formatNumber(inspection.duplicateRows)}</b>
                  <span>Duplicates</span>
                </div>
              </div>
              <div className="data-recommendation">
                <Icon name="cpu" />
                <span>
                  <b>{inspection.recommendedTokenLimit} token limit</b>
                  <small>
                    Estimated from the 95th percentile ({inspection.p95Tokens}
                    tokens); hardware fitting remains the upper safety bound.
                  </small>
                </span>
              </div>
            </section>

            <section className="data-card">
              <div className="data-card-heading">
                <div>
                  <span>Format and fields</span>
                  <small>
                    {inspection.formats.join(", ") || "Unrecognized"}
                  </small>
                </div>
                <button
                  type="button"
                  className="text-button"
                  onClick={resetMapping}
                >
                  Reset auto
                </button>
              </div>
              <label className="field">
                <span>Training data type</span>
                <select
                  value={task}
                  onChange={(event) =>
                    setTask(event.target.value as DatasetTask)
                  }
                >
                  {Object.entries(taskLabels).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <small>
                  Auto currently recommends{" "}
                  {taskLabels[inspection.recommendedTask]}.
                </small>
              </label>
              <div className="data-mapping-grid">
                {mapping &&
                  fieldLabels
                    .filter(([key]) => visibleMapping.has(key))
                    .map(([key, label]) => (
                      <label className="field" key={key}>
                        <span>{label}</span>
                        <select
                          value={mapping[key]}
                          onChange={(event) =>
                            setMapping({
                              ...mapping,
                              [key]: event.target.value,
                            })
                          }
                        >
                          <option value="">Auto-detect</option>
                          {inspection.fields.map((field) => (
                            <option key={field} value={field}>
                              {field} · {inspection.fieldCounts[field]}
                            </option>
                          ))}
                        </select>
                      </label>
                    ))}
              </div>
            </section>

            <section className="data-card data-options">
              <div className="data-card-heading">
                <div>
                  <span>Clean and split</span>
                  <small>Applied while saving the new copy</small>
                </div>
              </div>
              <label className="toggle-row compact">
                <input
                  type="checkbox"
                  checked={trimWhitespace}
                  onChange={(event) => setTrimWhitespace(event.target.checked)}
                />
                <span>
                  <b>Trim whitespace</b>
                </span>
              </label>
              <label className="toggle-row compact">
                <input
                  type="checkbox"
                  checked={removeDuplicates}
                  onChange={(event) =>
                    setRemoveDuplicates(event.target.checked)
                  }
                />
                <span>
                  <b>Remove exact duplicates</b>
                </span>
              </label>
              <label className="toggle-row compact">
                <input
                  type="checkbox"
                  checked={skipInvalid}
                  onChange={(event) => setSkipInvalid(event.target.checked)}
                />
                <span>
                  <b>Skip unresolved rows</b>
                  <small>
                    Off by default so data is never silently discarded.
                  </small>
                </span>
              </label>
              <div className="field-row two">
                <label className="field">
                  <span>Validation %</span>
                  <input
                    type="number"
                    min="0"
                    max="40"
                    value={validationPercent}
                    onChange={(event) =>
                      setValidationPercent(Number(event.target.value))
                    }
                  />
                </label>
                <label className="field">
                  <span>Test %</span>
                  <input
                    type="number"
                    min="0"
                    max="40"
                    value={testPercent}
                    onChange={(event) =>
                      setTestPercent(Number(event.target.value))
                    }
                  />
                </label>
              </div>
            </section>
          </aside>

          <main className="data-editor-main">
            <section className="data-card data-preview-card">
              <div className="data-card-heading">
                <div>
                  <span>Row preview</span>
                  <small>
                    Invalid rows appear first. Select one to repair its JSON.
                  </small>
                </div>
                <span className="data-modalities">
                  {inspection.modalities.join(" · ")}
                </span>
              </div>
              <div className="data-preview-list" role="list">
                {inspection.preview.map((row) => (
                  <button
                    type="button"
                    role="listitem"
                    key={row.id}
                    className={
                      "data-preview-row " +
                      (selectedRowId === row.id ? "active " : "") +
                      row.status
                    }
                    onClick={() => selectRow(row)}
                  >
                    <span className="data-row-status">
                      <Icon
                        name={row.status === "valid" ? "check" : "alert-circle"}
                      />
                    </span>
                    <span>
                      <b>
                        {row.split} · row {row.line}
                      </b>
                      <small>
                        {row.error ||
                          `${row.format} · about ${row.estimatedTokens} tokens`}
                      </small>
                    </span>
                    {edits[row.id] && <em>Edited</em>}
                  </button>
                ))}
              </div>
            </section>

            <section className="data-card data-json-card">
              <div className="data-card-heading">
                <div>
                  <span>Selected row</span>
                  <small>
                    {selectedRow
                      ? `${selectedRow.source} · row ${selectedRow.line}`
                      : "Select a preview row"}
                  </small>
                </div>
                <button
                  type="button"
                  className="quiet-button compact-button"
                  disabled={!selectedRow}
                  onClick={applyRowEdit}
                >
                  <Icon name="check" />
                  Apply edit
                </button>
              </div>
              <textarea
                className="data-json-editor"
                value={editorText}
                disabled={!selectedRow}
                spellCheck={false}
                onChange={(event) => setEditorText(event.target.value)}
                aria-label="Edit selected dataset row as JSON"
              />
              {selectedRow?.normalized && (
                <details className="normalized-preview">
                  <summary>Canonical preview</summary>
                  <pre>{JSON.stringify(selectedRow.normalized, null, 2)}</pre>
                </details>
              )}
            </section>

            {inspection.issues.length > 0 && (
              <section className="data-card data-issues-card">
                <div className="data-card-heading">
                  <div>
                    <span>Detected issues</span>
                    <small>
                      {inspection.truncatedIssues
                        ? `Showing the first ${inspection.issues.length}`
                        : `${inspection.issues.length} issue(s)`}
                    </small>
                  </div>
                </div>
                <div className="data-issue-list">
                  {inspection.issues.map((issue) => (
                    <button
                      type="button"
                      key={`${issue.id}-${issue.message}`}
                      onClick={() => {
                        const row = inspection.preview.find(
                          (item) => item.id === issue.id,
                        );
                        if (row) selectRow(row);
                      }}
                    >
                      <Icon name="alert-circle" />
                      <span>
                        <b>
                          {issue.split} · row {issue.line}
                        </b>
                        <small>{issue.message}</small>
                      </span>
                    </button>
                  ))}
                </div>
              </section>
            )}

            <section className="data-card data-save-card">
              <label className="field path-field">
                <span>Save clean dataset as</span>
                <div>
                  <input
                    value={outputDirectory}
                    spellCheck={false}
                    onChange={(event) => setOutputDirectory(event.target.value)}
                  />
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Choose clean dataset location"
                    onClick={() => void chooseOutput()}
                  >
                    <Icon name="folder" />
                  </button>
                </div>
              </label>
              <button
                type="button"
                className="primary-button"
                disabled={Boolean(busy) || !outputDirectory.trim()}
                onClick={() => void save()}
              >
                <Icon name={busy === "save" ? "loader" : "save"} />
                {busy === "save" ? "Saving…" : "Save clean copy"}
              </button>
            </section>

            {localError && (
              <div className="data-editor-error" role="alert">
                <Icon name="alert-triangle" />
                <span>{localError}</span>
              </div>
            )}

            {result && (
              <div className="data-editor-result">
                <Icon name="check-circle" />
                <span>
                  <b>{formatNumber(result.writtenRows)} clean rows saved</b>
                  <small>{result.outputDirectory}</small>
                </span>
                <div>
                  <button
                    type="button"
                    className="quiet-button compact-button"
                    onClick={() =>
                      onUseForFineTune(
                        result.outputDirectory,
                        result.inspection.recommendedTokenLimit,
                      )
                    }
                  >
                    Use for fine-tuning
                  </button>
                  <button
                    type="button"
                    className="quiet-button compact-button"
                    onClick={() =>
                      onUseForAlignment(
                        result.outputDirectory,
                        result.inspection.recommendedTokenLimit,
                      )
                    }
                  >
                    Use for alignment
                  </button>
                </div>
              </div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}
