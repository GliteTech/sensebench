(function () {
  const basePath = window.SENSEBENCH_BASE_PATH || "/";
  const table = document.getElementById("leaderboard-table");
  const searchInput = document.getElementById("leaderboard-search");
  const datasetFilter = document.getElementById("dataset-filter");
  const promptFilter = document.getElementById("prompt-filter");
  const sourceFilter = document.getElementById("source-filter");
  const hostingFilter = document.getElementById("hosting-filter");
  const gpuFilter = document.getElementById("gpu-filter");
  const quantFilter = document.getElementById("quant-filter");
  const schemeGoldSelect = document.getElementById("scheme-gold-select");
  const schemeGranularitySelect = document.getElementById("scheme-granularity-select");
  const xMetricSelect = document.getElementById("x-metric-select");
  const xScaleSelect = document.getElementById("x-scale-select");
  const maxCostFilter = document.getElementById("max-cost-filter");
  const viewFilter = document.getElementById("view-filter");
  const frontierOnly = document.getElementById("frontier-only");
  const sortSelect = document.getElementById("sort-select");
  const downloadCsvButton = document.getElementById("download-csv");
  const downloadJsonButton = document.getElementById("download-json");
  const compareBar = document.getElementById("compare-bar");
  const compareBarCount = document.getElementById("compare-bar-count");
  const compareBarClear = document.getElementById("compare-bar-clear");
  const chartElement = document.getElementById("leaderboard-chart");
  const chartNote = document.getElementById("chart-note");
  const compareCharts = document.getElementById("compare-charts");
  const compareEmpty = document.getElementById("compare-empty");
  const comparePairwise = document.getElementById("compare-pairwise");
  const compareTable = document.getElementById("compare-table");
  const baselinesTableBody = document.querySelector(".baselines-table tbody");
  const dataVersion = window.SENSEBENCH_DATA_VERSION || "";

  if (!table) {
    return;
  }

  const state = {
    entries: [],
    baselines: [],
    selected: new Set(),
    sortKey: "rank",
    sortDirection: 1,
    dataSchemaVersion: "",
    generatedAt: null
  };

  // Scheme ids in display order; the official default (lexen_fine) is first. The download
  // reads accuracy from these immutable scheme_scores, never the top-level accuracy field
  // (which applyScheme() rewrites in place for whichever scheme the dropdowns select).
  const EXPORT_SCHEME_IDS = [
    "lexen_fine",
    "lexen_coarse",
    "maru2022_fine",
    "maru2022_coarse",
    "raganato_fine",
    "raganato_coarse"
  ];
  const DEFAULT_SCHEME_ID = "lexen_fine";

  const runDetailCache = new Map();
  let pairwiseToken = 0;
  let mainChart = null;

  const SIGNIFICANCE_LEVEL = 0.05;
  const EXACT_MCNEMAR_MAX_DISCORDANT = 25;
  const Z_95 = 1.959963984540054;

  const X_METRICS = {
    cost_per_million_items: {
      axisLabel: "Cost per million items, USD",
      value: (entry) => entry.cost_per_million_items,
      format: formatMoney,
      missingNote: ""
    },
    machine_hours_per_million_items: {
      axisLabel: "Machine-hours per 1M items",
      value: (entry) => entry.machine_hours_per_million_items,
      format: formatMachineHours,
      missingNote: " Cloud API runs record no machine time and are not plotted."
    }
  };

  function activeXMetric() {
    return X_METRICS[xMetricSelect?.value] || X_METRICS.cost_per_million_items;
  }

  function activeScheme() {
    const gold = schemeGoldSelect?.value || "lexen";
    const granularity = schemeGranularitySelect?.value || "fine";
    return `${gold}_${granularity}`;
  }

  // Project the active scheme's accuracy/CI onto each entry and baseline so the rest of the
  // pipeline (table, chart, frontier, ranks, tooltips) reads one accuracy field. Idempotent:
  // each render re-derives from the immutable scheme_scores, not the previous projection.
  function applyScheme() {
    const scheme = activeScheme();
    for (const entry of state.entries) {
      const score = entry.scheme_scores && entry.scheme_scores[scheme];
      if (score) {
        entry.accuracy = score.accuracy;
        entry.accuracy_ci = score.accuracy_ci;
      }
    }
    for (const baseline of state.baselines) {
      const score = baseline.scheme_scores && baseline.scheme_scores[scheme];
      if (score) {
        baseline.accuracy = score.accuracy;
        baseline.accuracy_ci = score.accuracy_ci;
      }
    }
  }

  function isDefaultScheme() {
    return activeScheme() === "lexen_fine";
  }

  function activeSchemeLabel() {
    const gold = (schemeGoldSelect?.selectedOptions?.[0]?.text || "lexEN v1").replace(
      " (default)",
      ""
    );
    const granularity = (
      schemeGranularitySelect?.selectedOptions?.[0]?.text || "WordNet fine-grained"
    ).replace(" (default)", "");
    return `${gold} · ${granularity}`;
  }

  // When a non-default scheme is active, label every accuracy column header so it is clear the
  // numbers are not the default lexEN v1 · WordNet fine-grained score.
  function updateAccuracyHeaders() {
    const note = isDefaultScheme() ? "" : `Scoring: ${activeSchemeLabel()}`;
    document.querySelectorAll(".accuracy-scheme-note").forEach((element) => {
      element.textContent = note;
      element.hidden = note === "";
    });
  }

  function activeXScale() {
    return xScaleSelect?.value === "linear" ? "linear" : "log";
  }

  // A log axis cannot plot non-positive x; drop those points (and report them).
  function plottablePoints(points, scale) {
    return scale === "log" ? points.filter((point) => point.x > 0) : points;
  }

  // Fixed family -> colour map. Brand-anchored where it doesn't hurt contrast
  // (Google blue, Meta navy, NVIDIA green, Mistral orange, OpenAI teal,
  // Anthropic rust, DeepSeek indigo); the rest are spread around the wheel for
  // distinguishability. Fixed (not frequency-ranked) so a family keeps its
  // colour across filters and sessions.
  const FAMILY_COLORS = {
    GPT: "#10A37F", // teal (OpenAI)
    Claude: "#C2410C", // rust/clay (Anthropic)
    Gemini: "#4285F4", // Google blue
    Gemma: "#34A853", // Google green (keeps Gemma distinct from Gemini)
    Qwen: "#8E24AA", // magenta (Alibaba)
    DeepSeek: "#5C7CFA", // indigo
    Llama: "#0D2C8B", // deep navy (Meta)
    GLM: "#00ACC1", // cyan (Z.ai)
    Mistral: "#F57C00", // orange
    Kimi: "#7C3AED", // violet (Moonshot)
    Grok: "#C2185B", // crimson (xAI)
    MiniMax: "#455A64", // dark slate
    Nemotron: "#6FAE00", // green (NVIDIA)
    Granite: "#90A4AE", // blue-grey (IBM)
    Phi: "#FFB300", // amber
    OLMo: "#827717" // olive
  };
  const OTHER_FAMILY_LABEL = "Other";
  const OTHER_FAMILY_COLOR = "#9aa0a6";
  // Model name -> family. Order matters; first match wins. Families absent from
  // FAMILY_COLORS (e.g. Command, Hunyuan) keep an accurate name in the hover
  // tooltip but collapse into "Other" in the legend.
  const FAMILY_PATTERNS = [
    [/gemma/, "Gemma"],
    [/gemini/, "Gemini"],
    [/qwen/, "Qwen"],
    [/glm/, "GLM"],
    [/llama|maverick|scout/, "Llama"],
    [/mistral|mixtral|magistral|ministral|pixtral/, "Mistral"],
    [/nemotron/, "Nemotron"],
    [/deepseek/, "DeepSeek"],
    [/granite/, "Granite"],
    [/phi-?\d/, "Phi"],
    [/hunyuan/, "Hunyuan"],
    [/olmo/, "OLMo"],
    [/command|c4ai/, "Command"],
    [/kimi|moonshot/, "Kimi"],
    [/grok/, "Grok"],
    [/gpt|davinci/, "GPT"],
    [/claude/, "Claude"],
    [/minimax/, "MiniMax"]
  ];

  function familyOf(entry) {
    const name = String(entry.model || entry.requested_model || "").toLowerCase();
    for (const [pattern, label] of FAMILY_PATTERNS) {
      if (pattern.test(name)) {
        return label;
      }
    }
    return entry.llm_vendor || OTHER_FAMILY_LABEL;
  }

  function isSelfHosted(entry) {
    return entry.hosting_kind === "self_hosted";
  }

  // Shape encodes where the run executed (and thus how its cost is derived).
  function pointSymbol(entry) {
    return isSelfHosted(entry) ? "triangle" : "circle";
  }

  // Group points by family using the fixed colour map; unknown families -> "Other".
  // Colour per family is fixed; only the legend order follows frequency.
  function familyAssignments(points) {
    const counts = new Map();
    let hasOther = false;
    for (const point of points) {
      const family = familyOf(point.entry);
      if (FAMILY_COLORS[family]) {
        counts.set(family, (counts.get(family) || 0) + 1);
      } else {
        hasOther = true;
      }
    }
    const legendFamilies = [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map((entry) => entry[0]);
    if (hasOther) {
      legendFamilies.push(OTHER_FAMILY_LABEL);
    }
    const colorOf = new Map(Object.entries(FAMILY_COLORS));
    colorOf.set(OTHER_FAMILY_LABEL, OTHER_FAMILY_COLOR);
    const display = (entry) => {
      const family = familyOf(entry);
      return FAMILY_COLORS[family] ? family : OTHER_FAMILY_LABEL;
    };
    return { colorOf, display, legendFamilies };
  }

  const sourceLabels = {
    open_source: "Open weights",
    proprietary: "Proprietary"
  };

  const baselineKindLabels = {
    computed_wordnet_mfs: "Computed at build time",
    published_predictions: "Published predictions",
    reproduced_predictions: "Reproduced predictions"
  };

  const compareMetrics = [
    {
      key: "accuracy",
      title: "Accuracy %",
      value: (entry) => (entry.accuracy == null ? null : entry.accuracy * 100),
      format: (value) => `${formatNumber(value, 2)}%`
    },
    {
      key: "cost_per_million_items",
      title: "Cost / M items",
      value: (entry) => entry.cost_per_million_items,
      format: formatMoney
    },
    {
      key: "tokens_per_item",
      title: "Tokens / item",
      value: (entry) => entry.tokens_per_item,
      format: (value) => formatNumber(value, 1)
    },
    {
      key: "machine_hours_per_million_items",
      title: "Machine-h / 1M",
      value: (entry) => entry.machine_hours_per_million_items,
      format: formatMachineHours
    }
  ];

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatPercent(value) {
    return value == null ? "n/a" : `${(value * 100).toFixed(2)}%`;
  }

  function formatNumber(value, digits = 2) {
    if (value == null) {
      return "n/a";
    }
    return Number(value).toLocaleString(undefined, {
      maximumFractionDigits: digits
    });
  }

  function formatMoney(value) {
    if (value == null) {
      return "n/a";
    }
    const amount = Number(value);
    if (amount >= 100) {
      return `$${Math.round(amount).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
    }
    if (amount >= 1) {
      return `$${amount.toFixed(2)}`;
    }
    if (amount <= 0) {
      return "$0.00";
    }
    const decimals = 2 - Math.floor(Math.log10(amount));
    return `$${amount.toFixed(decimals)}`;
  }

  function formatSeconds(value) {
    if (value == null) {
      return "n/a";
    }
    const seconds = Number(value);
    if (seconds < 60) {
      return `${seconds.toFixed(2)}s`;
    }
    return `${(seconds / 60).toFixed(2)}m`;
  }

  function formatMachineHours(value) {
    if (value == null) {
      return "n/a";
    }
    const hours = Number(value);
    if (hours < 10) {
      return `${hours.toFixed(2)} h/M`;
    }
    return `${formatNumber(hours, 1)} h/M`;
  }

  function gpuLabel(entry) {
    if (!entry.gpu) {
      return null;
    }
    if (entry.gpu_count != null && entry.gpu_count > 1) {
      return `${entry.gpu_count}×${entry.gpu}`;
    }
    return entry.gpu;
  }

  function logoHtml(entry) {
    if (entry.logo_slug) {
      return `<img class="vendor-logo" src="${basePath}assets/logos/${escapeHtml(entry.logo_slug)}.svg" alt="" width="18" height="18" loading="lazy">`;
    }
    const family = entry.family || familyOf(entry);
    const initial =
      entry.vendor_initial ||
      (entry.llm_vendor || entry.model || "?").trim().charAt(0).toUpperCase() ||
      "?";
    return `<span class="vendor-initial fam-${escapeHtml(String(family).toLowerCase())}">${escapeHtml(initial)}</span>`;
  }

  function modelLabel(entry) {
    const base = entry.display_label || entry.model;
    if (entry.reasoning_effort) {
      return `${base} (${entry.reasoning_effort})`;
    }
    return base;
  }

  function shortModelName(model) {
    const slash = model.indexOf("/");
    return slash === -1 ? model : model.slice(slash + 1);
  }

  function shortModelLabel(entry) {
    const short = shortModelName(entry.model);
    if (entry.reasoning_effort) {
      return `${short} (${entry.reasoning_effort})`;
    }
    return short;
  }

  function sourceLabel(entry) {
    return sourceLabels[entry.source_kind] || "Unknown source";
  }

  function promptLabel(entry) {
    if (entry.prompt_name) {
      return `${entry.prompt_id} — ${entry.prompt_name}`;
    }
    return entry.prompt_id;
  }

  function ciHalfWidth(entry) {
    const ci = entry.accuracy_ci;
    if (!ci || ci.low == null || ci.high == null) {
      return null;
    }
    return (ci.high - ci.low) / 2;
  }

  function searchableText(entry) {
    return [
      entry.model,
      entry.reasoning_effort,
      entry.requested_model,
      entry.resolved_model,
      entry.llm_vendor,
      entry.api_provider,
      entry.gpu,
      entry.inference_engine,
      entry.quantization,
      entry.runner_github_handle,
      entry.runner_name,
      entry.source_kind,
      entry.license,
      entry.run_id
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
  }

  function bestPerModel(entries) {
    const best = new Map();
    for (const entry of entries) {
      const current = best.get(entry.best_group_key);
      if (!current || entry.rank < current.rank) {
        best.set(entry.best_group_key, entry);
      }
    }
    return Array.from(best.values());
  }

  function activeMaxCost() {
    const raw = (maxCostFilter?.value || "").trim();
    if (raw === "") {
      return null;
    }
    const parsed = Number(raw);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
  }

  function filteredEntries() {
    const query = (searchInput?.value || "").trim().toLowerCase();
    const dataset = datasetFilter?.value || "";
    const prompt = promptFilter?.value || "";
    const source = sourceFilter?.value || "";
    const hosting = hostingFilter?.value || "";
    const gpu = gpuFilter?.value || "";
    const quant = quantFilter?.value || "";
    const maxCost = activeMaxCost();
    let entries = state.entries.filter((entry) => {
      if (dataset && entry.dataset_version !== dataset) {
        return false;
      }
      if (prompt && entry.prompt_id !== prompt) {
        return false;
      }
      if (source && entry.source_kind !== source) {
        return false;
      }
      if (hosting && entry.hosting_kind !== hosting) {
        return false;
      }
      if (gpu && entry.gpu !== gpu) {
        return false;
      }
      if (quant && entry.quantization !== quant) {
        return false;
      }
      if (maxCost != null) {
        if (entry.cost_per_million_items == null || entry.cost_per_million_items > maxCost) {
          return false;
        }
      }
      return !query || searchableText(entry).includes(query);
    });
    if ((viewFilter?.value || "all") === "best") {
      entries = bestPerModel(entries);
    }
    return entries;
  }

  function leaderboardOrderKey(entry) {
    const accuracy = entry.accuracy != null ? entry.accuracy : -1;
    const cost =
      entry.cost_per_million_items != null ? entry.cost_per_million_items : Infinity;
    const createdAt = Date.parse(entry.created_at);
    const createdAtValue = Number.isFinite(createdAt) ? createdAt : -Infinity;
    return { accuracy, cost, createdAtValue };
  }

  function compareLeaderboardOrder(a, b) {
    const left = leaderboardOrderKey(a);
    const right = leaderboardOrderKey(b);
    if (left.accuracy !== right.accuracy) {
      return right.accuracy - left.accuracy;
    }
    if (left.cost !== right.cost) {
      return left.cost < right.cost ? -1 : 1;
    }
    if (left.createdAtValue !== right.createdAtValue) {
      return right.createdAtValue - left.createdAtValue;
    }
    return a.run_id.localeCompare(b.run_id);
  }

  function rankRanges(rows) {
    const ranges = new Map();
    const withCi = rows.filter((row) => ciHalfWidth(row.entry) != null);
    for (const row of rows) {
      const ci = row.entry.accuracy_ci;
      if (ciHalfWidth(row.entry) == null) {
        ranges.set(row.entry.run_id, null);
        continue;
      }
      let certainlyBetter = 0;
      let possiblyAtLeast = 0;
      for (const other of withCi) {
        if (other.entry.accuracy_ci.low > ci.high) {
          certainlyBetter += 1;
        }
        if (other.entry.accuracy_ci.high >= ci.low) {
          possiblyAtLeast += 1;
        }
      }
      ranges.set(row.entry.run_id, { low: 1 + certainlyBetter, high: possiblyAtLeast });
    }
    return ranges;
  }

  function decorateRows(entries, frontierIds) {
    const ordered = [...entries].sort(compareLeaderboardOrder);
    const rows = ordered.map((entry, index) => ({
      entry,
      displayRank: index + 1,
      onFrontier: frontierIds.has(entry.run_id),
      rankRange: null
    }));
    const ranges = rankRanges(rows);
    for (const row of rows) {
      row.rankRange = ranges.get(row.entry.run_id) || null;
    }
    return rows;
  }

  function sortRows(rows) {
    const key = state.sortKey;
    const direction = state.sortDirection;
    return [...rows].sort((a, b) => {
      let left;
      let right;
      if (key === "rank") {
        left = a.displayRank;
        right = b.displayRank;
      } else if (key === "model") {
        left = modelLabel(a.entry);
        right = modelLabel(b.entry);
      } else {
        left = a.entry[key];
        right = b.entry[key];
      }
      if (left == null && right == null) {
        return a.entry.run_id.localeCompare(b.entry.run_id);
      }
      if (left == null) {
        return 1;
      }
      if (right == null) {
        return -1;
      }
      if (typeof left === "number" && typeof right === "number") {
        return (left - right) * direction;
      }
      return String(left).localeCompare(String(right)) * direction;
    });
  }

  function renderTable(rows) {
    const tbody = table.querySelector("tbody");
    if (!tbody) {
      return;
    }
    tbody.innerHTML = rows
      .map((row) => {
        const entry = row.entry;
        const checked = state.selected.has(entry.run_id) ? " checked" : "";
        const disabled = !checked && state.selected.size >= 6 ? " disabled" : "";
        const range = row.rankRange;
        const rangeHtml =
          range && range.low !== range.high
            ? `<div class="cell-secondary" title="Plausible rank range from overlapping 95% confidence intervals">${range.low}–${range.high}</div>`
            : "";
        const half = ciHalfWidth(entry);
        const ciTitle =
          half == null
            ? ""
            : `95% CI: ${formatPercent(entry.accuracy_ci.low)} – ${formatPercent(entry.accuracy_ci.high)}`;
        const ciHtml =
          half == null
            ? ""
            : `<div class="cell-secondary" title="${escapeHtml(ciTitle)}">±${formatPercent(half)}</div>`;
        const vendorParts = [];
        if (entry.llm_vendor) {
          vendorParts.push(escapeHtml(entry.llm_vendor));
        }
        vendorParts.push(escapeHtml(sourceLabel(entry)));
        if (entry.quantization) {
          vendorParts.push(escapeHtml(entry.quantization));
        }
        const gpu = gpuLabel(entry);
        if (gpu != null) {
          vendorParts.push(escapeHtml(gpu));
        }
        const frontierHtml = row.onFrontier
          ? '<span class="badge badge-frontier" title="On the accuracy-cost Pareto frontier">★</span>'
          : "";
        return `<tr>
          <td class="col-rank"><div class="cell-primary">${row.displayRank}</div>${rangeHtml}</td>
          <td class="col-compare"><input class="compare-checkbox" type="checkbox" data-run-id="${escapeHtml(entry.run_id)}"${checked}${disabled}></td>
          <td class="col-model">
            <div class="cell-primary">${logoHtml(entry)}<a class="model-link" href="${basePath}${escapeHtml(entry.run_url)}" title="${escapeHtml(entry.run_id)}">${escapeHtml(modelLabel(entry))}</a></div>
            <div class="cell-secondary">${vendorParts.join(" · ")}</div>
            <div class="cell-secondary provenance-mobile"><a href="${basePath}prompts/${encodeURIComponent(entry.prompt_id)}/">${escapeHtml(entry.prompt_id)}</a></div>
          </td>
          <td class="col-accuracy"><div class="cell-primary">${frontierHtml ? frontierHtml + " " : ""}${formatPercent(entry.accuracy)}</div>${ciHtml}</td>
          <td class="col-cost">${formatMoney(entry.cost_per_million_items)}</td>
          <td class="col-prompt"><a href="${basePath}prompts/${encodeURIComponent(entry.prompt_id)}/">${escapeHtml(entry.prompt_id)}</a></td>
        </tr>`;
      })
      .join("");
    tbody.querySelectorAll(".compare-checkbox").forEach((checkbox) => {
      checkbox.addEventListener("change", (event) => {
        const target = event.currentTarget;
        const runId = target.dataset.runId;
        if (!runId) {
          return;
        }
        if (target.checked) {
          state.selected.add(runId);
        } else {
          state.selected.delete(runId);
        }
        render();
      });
    });
  }

  function updateSortIndicators() {
    table.querySelectorAll("[data-sort]").forEach((button) => {
      const existing = button.querySelector(".sort-arrow");
      if (existing) {
        existing.remove();
      }
      const isActive = button.dataset.sort === state.sortKey;
      const th = button.closest("th");
      if (isActive) {
        const arrow = document.createElement("span");
        arrow.className = "sort-arrow";
        arrow.setAttribute("aria-hidden", "true");
        arrow.textContent = state.sortDirection === 1 ? " ▲" : " ▼";
        button.appendChild(arrow);
        if (th) {
          th.setAttribute("aria-sort", state.sortDirection === 1 ? "ascending" : "descending");
        }
      } else if (th) {
        th.removeAttribute("aria-sort");
      }
    });
    if (sortSelect && sortSelect.value !== state.sortKey) {
      sortSelect.value = state.sortKey;
    }
  }

  function updateCompareBar() {
    if (!compareBar) {
      return;
    }
    const count = state.selected.size;
    compareBar.hidden = count < 2;
    if (compareBarCount) {
      compareBarCount.textContent = `Compare (${count})`;
    }
  }

  function paretoFrontier(points) {
    return points.filter((point) => {
      return !points.some((other) => {
        const atLeastAsAccurate = other.entry.accuracy >= point.entry.accuracy;
        const atMostAsExpensive = other.x <= point.x;
        const strictImprovement = other.entry.accuracy > point.entry.accuracy || other.x < point.x;
        return atLeastAsAccurate && atMostAsExpensive && strictImprovement;
      });
    });
  }

  function chartPoints(entries, metric) {
    return entries
      .filter((entry) => entry.accuracy != null && metric.value(entry) != null)
      .map((entry) => ({
        x: metric.value(entry),
        y: entry.accuracy * 100,
        entry
      }));
  }

  function visibleBaselines() {
    const dataset = datasetFilter?.value || "";
    if (!dataset) {
      return state.baselines;
    }
    return state.baselines.filter((baseline) => baseline.dataset_version === dataset);
  }

  function baselineShortLabel(baseline) {
    return baseline.label.split(" (")[0];
  }

  function renderMainChart(entries, frontierPoints, metric, scale) {
    if (!chartElement || !window.echarts) {
      return;
    }
    const allPoints = chartPoints(entries, metric);
    const points = plottablePoints(allPoints, scale);
    const frontier = plottablePoints([...frontierPoints], scale).sort((a, b) => a.x - b.x);
    const scatterPoints = frontierOnly?.checked ? frontier : points;
    const frontierSet = new Set(frontier.map((point) => point.entry.run_id));
    const baselines = visibleBaselines();
    const { colorOf, display, legendFamilies } = familyAssignments(scatterPoints);
    const byFamily = new Map();
    for (const point of scatterPoints) {
      const family = display(point.entry);
      if (!byFamily.has(family)) {
        byFamily.set(family, []);
      }
      byFamily.get(family).push(point);
    }
    // Only the Pareto frontier is labelled (below); all other points are
    // hover-only to keep the dense cluster readable.
    const familySeries = legendFamilies
      .filter((family) => byFamily.has(family))
      .map((family) => ({
        name: family,
        type: "scatter",
        symbolSize: 11,
        itemStyle: { color: colorOf.get(family) },
        emphasis: { focus: "series" },
        labelLayout: { hideOverlap: true },
        data: byFamily.get(family).map((point) => {
          const onFrontier = frontierSet.has(point.entry.run_id);
          const selfHosted = isSelfHosted(point.entry);
          // Frontier points keep their family colour and cloud/self-hosted shape
          // but are drawn hollow (empty* = white-filled, coloured ring) so they
          // read as the frontier instead of a misleading uniform marker.
          const symbol = onFrontier
            ? selfHosted
              ? "emptyTriangle"
              : "emptyCircle"
            : selfHosted
              ? "triangle"
              : "circle";
          return {
            value: [point.x, point.y],
            entry: point.entry,
            symbol,
            symbolSize: onFrontier ? 13 : 11,
            itemStyle: onFrontier
              ? {
                  color: colorOf.get(family),
                  borderColor: colorOf.get(family),
                  borderWidth: 2
                }
              : { color: colorOf.get(family) },
            // Label only the frontier points (the markers now live here, not on
            // the marker-less frontier line); everything else is hover-only.
            label: onFrontier
              ? {
                  show: true,
                  position: "top",
                  fontSize: 11,
                  formatter: () => shortModelLabel(point.entry)
                }
              : { show: false }
          };
        })
      }));
    if (!mainChart) {
      mainChart = window.echarts.init(chartElement);
      mainChart.on("click", (params) => {
        const runUrl = params?.data?.entry?.run_url;
        if (runUrl) {
          window.location.href = `${basePath}${runUrl}`;
        }
      });
    }
    const yValues = [
      ...scatterPoints.map((point) => point.y),
      ...baselines.map((baseline) => baseline.accuracy * 100)
    ];
    const yMin = yValues.length > 0 ? Math.max(0, Math.floor(Math.min(...yValues)) - 1) : 0;
    mainChart.setOption(
      {
        animation: false,
        grid: { left: 58, right: 24, top: 52, bottom: 62 },
        legend: {
          type: "scroll",
          top: 0,
          data: legendFamilies,
          textStyle: { fontSize: 11 }
        },
        tooltip: {
          trigger: "item",
          formatter: (params) => {
            const entry = params.data.entry;
            if (!entry) {
              return "";
            }
            const half = ciHalfWidth(entry);
            const accuracyText =
              half == null
                ? `Accuracy: ${formatPercent(entry.accuracy)}`
                : `Accuracy: ${formatPercent(entry.accuracy)} ±${formatPercent(half)}`;
            const lines = [
              `<strong>${escapeHtml(shortModelLabel(entry))}</strong>`,
              `Family: ${escapeHtml(familyOf(entry))} · ${isSelfHosted(entry) ? "self-hosted" : "cloud API"}`,
              `Prompt: ${escapeHtml(promptLabel(entry))}`,
              accuracyText,
              `${metric.axisLabel}: ${metric.format(metric.value(entry))}`
            ];
            const gpu = gpuLabel(entry);
            if (gpu != null) {
              lines.push(`GPU: ${escapeHtml(gpu)}`);
            }
            if (entry.quantization) {
              lines.push(`Quantization: ${escapeHtml(entry.quantization)}`);
            }
            return lines.join("<br>");
          }
        },
        xAxis: {
          name: scale === "log" ? `${metric.axisLabel} (log scale)` : metric.axisLabel,
          nameLocation: "middle",
          nameGap: 42,
          type: scale === "log" ? "log" : "value"
        },
        yAxis: {
          name: "Accuracy %",
          type: "value",
          min: yMin
        },
        series: [
          ...familySeries,
          {
            name: "Pareto frontier",
            type: "line",
            showSymbol: false,
            itemStyle: { color: "#374151" },
            lineStyle: { width: 2, color: "#374151" },
            emphasis: { disabled: true },
            blur: { lineStyle: { opacity: 0.9 }, itemStyle: { opacity: 0.9 } },
            z: 5,
            data: frontier.map((point) => ({
              value: [point.x, point.y],
              entry: point.entry
            }))
          },
          {
            name: "_baselines",
            type: "line",
            data: [],
            silent: true,
            markLine: {
              silent: true,
              symbol: "none",
              animation: false,
              lineStyle: { type: "dashed", width: 1 },
              label: {
                position: "insideEndTop",
                formatter: (params) => params.name,
                fontSize: 11
              },
              data: baselines.map((baseline, index) => ({
                name: `${baselineShortLabel(baseline)} ${formatPercent(baseline.accuracy)}`,
                yAxis: baseline.accuracy * 100,
                label: { position: index % 2 === 0 ? "insideStartTop" : "insideEndTop" }
              }))
            }
          }
        ]
      },
      { notMerge: true }
    );
    const unavailable = entries.length - allPoints.length;
    const droppedByLog = allPoints.length - points.length;
    if (chartNote) {
      const baselineNote =
        baselines.length > 0
          ? ` Dashed lines mark reference baselines scored on the same items.`
          : "";
      const missingNote = unavailable > 0 ? metric.missingNote : "";
      const logNote =
        droppedByLog > 0
          ? ` ${droppedByLog} row${droppedByLog === 1 ? "" : "s"} with a zero value omitted on the log scale.`
          : "";
      const shapeNote =
        " Colour = model family; ● = cloud API, ▲ = self-hosted. Hover a point or legend to highlight its family.";
      const costCaveat =
        metric === X_METRICS.cost_per_million_items
          ? " Self-hosted cost is estimated machine time; cloud cost is API list price, so the two are not directly comparable."
          : "";
      chartNote.textContent = `${points.length} rows plotted. ${unavailable} rows have unavailable values for this chart.${missingNote}${logNote}${shapeNote}${costCaveat}${baselineNote}`;
    }
  }

  function disposeCompareCharts() {
    if (!compareCharts || !window.echarts) {
      return;
    }
    compareCharts.querySelectorAll(".compare-metric-chart").forEach((element) => {
      const instance = window.echarts.getInstanceByDom(element);
      if (instance) {
        instance.dispose();
      }
    });
  }

  function accuracyWhiskerSeries(selectedEntries) {
    const data = selectedEntries
      .map((entry, index) => {
        const ci = entry.accuracy_ci;
        if (!ci || ci.low == null || ci.high == null) {
          return null;
        }
        return [index, ci.low * 100, ci.high * 100];
      })
      .filter(Boolean);
    return {
      name: "95% CI",
      type: "custom",
      silent: true,
      z: 10,
      renderItem: (params, api) => {
        const low = api.coord([api.value(0), api.value(1)]);
        const high = api.coord([api.value(0), api.value(2)]);
        const half = Math.min(10, api.size([1, 0])[0] * 0.12);
        const style = { stroke: "#7a868f", lineWidth: 1.5 };
        return {
          type: "group",
          children: [
            { type: "line", shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] }, style },
            {
              type: "line",
              shape: { x1: low[0] - half, y1: high[1], x2: low[0] + half, y2: high[1] },
              style
            },
            {
              type: "line",
              shape: { x1: low[0] - half, y1: low[1], x2: low[0] + half, y2: low[1] },
              style
            }
          ]
        };
      },
      data
    };
  }

  function renderCompareCharts() {
    if (!compareCharts || !window.echarts) {
      return;
    }
    const selectedEntries = state.entries.filter((entry) => state.selected.has(entry.run_id));
    if (compareEmpty) {
      compareEmpty.style.display = selectedEntries.length === 0 ? "block" : "none";
    }
    disposeCompareCharts();
    if (selectedEntries.length === 0) {
      compareCharts.innerHTML = "";
      renderCompareTable(selectedEntries);
      return;
    }
    compareCharts.innerHTML = compareMetrics
      .map(
        (metric) => `<div class="compare-chart-block">
          <h3>${escapeHtml(metric.title)}</h3>
          <div class="chart compare-metric-chart" data-metric="${escapeHtml(metric.key)}"></div>
        </div>`
      )
      .join("");
    compareCharts.querySelectorAll(".compare-metric-chart").forEach((element) => {
      const metric = compareMetrics.find((candidate) => candidate.key === element.dataset.metric);
      if (!metric) {
        return;
      }
      const isAccuracy = metric.key === "accuracy";
      const series = [
        {
          name: metric.title,
          type: "bar",
          data: selectedEntries.map((entry) => metric.value(entry)),
          label: {
            show: true,
            position: isAccuracy ? "inside" : "top",
            fontSize: 11,
            formatter: (params) => (params.value == null ? "" : metric.format(params.value))
          }
        }
      ];
      if (isAccuracy) {
        series.push(accuracyWhiskerSeries(selectedEntries));
      }
      const chart = window.echarts.init(element);
      chart.setOption({
        animation: false,
        grid: { left: 58, right: 16, top: 22, bottom: 72 },
        tooltip: {
          trigger: "axis",
          formatter: (params) => {
            const point = params[0];
            const entry = selectedEntries[point.dataIndex];
            return [
              `<strong>${escapeHtml(modelLabel(entry))}</strong>`,
              `Prompt: ${escapeHtml(promptLabel(entry))}`,
              `${escapeHtml(metric.title)}: ${metric.format(point.value)}`
            ].join("<br>");
          }
        },
        xAxis: {
          type: "category",
          data: selectedEntries.map((entry) => modelLabel(entry)),
          axisLabel: { interval: 0, rotate: 20 }
        },
        yAxis: {
          name: metric.title,
          type: "value",
          nameGap: 42
        },
        series
      });
    });
    renderCompareTable(selectedEntries);
  }

  function renderCompareTable(selectedEntries) {
    if (!compareTable) {
      return;
    }
    if (selectedEntries.length === 0) {
      compareTable.innerHTML = "";
      return;
    }
    compareTable.innerHTML = `<table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy<span class="accuracy-scheme-note" hidden></span></th>
          <th>Cost / M items</th>
          <th>Tokens / item</th>
          <th>Machine-h / 1M</th>
          <th>Run</th>
        </tr>
      </thead>
      <tbody>
        ${selectedEntries
          .map((entry) => {
            const half = ciHalfWidth(entry);
            const ciHtml =
              half == null
                ? ""
                : `<div class="cell-secondary">±${formatPercent(half)}</div>`;
            const secondaryParts = [
              `<a href="${basePath}prompts/${encodeURIComponent(entry.prompt_id)}/">${escapeHtml(entry.prompt_id)}</a>`,
              escapeHtml(entry.dataset_version ?? "")
            ];
            if (entry.quantization) {
              secondaryParts.push(escapeHtml(entry.quantization));
            }
            const gpu = gpuLabel(entry);
            if (gpu != null) {
              secondaryParts.push(escapeHtml(gpu));
            }
            return `<tr>
              <td>
                <div class="cell-primary">${escapeHtml(modelLabel(entry))}</div>
                <div class="cell-secondary">${secondaryParts.join(" · ")}</div>
              </td>
              <td><div class="cell-primary">${formatPercent(entry.accuracy)}</div>${ciHtml}</td>
              <td>${formatMoney(entry.cost_per_million_items)}</td>
              <td>${formatNumber(entry.tokens_per_item, 1)}</td>
              <td>${formatMachineHours(entry.machine_hours_per_million_items)}</td>
              <td><a href="${basePath}${escapeHtml(entry.run_url)}">${escapeHtml(entry.run_id)}</a></td>
            </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;
  }

  function erfc(x) {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * z);
    const poly =
      t *
      (0.254829592 +
        t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    const value = poly * Math.exp(-z * z);
    return x >= 0 ? value : 2 - value;
  }

  function mcnemarPValue(b, c) {
    const discordant = b + c;
    if (discordant === 0) {
      return { p: 1, method: "no discordant pairs" };
    }
    if (discordant <= EXACT_MCNEMAR_MAX_DISCORDANT) {
      const k = Math.min(b, c);
      let logPmf = discordant * Math.log(0.5);
      let cdf = Math.exp(logPmf);
      for (let i = 1; i <= k; i += 1) {
        logPmf += Math.log((discordant - i + 1) / i);
        cdf += Math.exp(logPmf);
      }
      return { p: Math.min(1, 2 * cdf), method: "exact binomial" };
    }
    const chi = (Math.abs(b - c) - 1) ** 2 / discordant;
    return { p: erfc(Math.sqrt(chi / 2)), method: "chi-square with continuity correction" };
  }

  function pairedDifferenceCi(b, c, n) {
    const diff = (b - c) / n;
    const variance = Math.max(0, (b + c) / n - diff * diff) / n;
    const margin = Z_95 * Math.sqrt(variance);
    return { diff, low: diff - margin, high: diff + margin };
  }

  function formatPValue(p) {
    if (p < 0.001) {
      return "p &lt; 0.001";
    }
    return `p = ${p.toFixed(3)}`;
  }

  function formatPp(value) {
    const pp = value * 100;
    const sign = pp > 0 ? "+" : "";
    return `${sign}${pp.toFixed(2)}`;
  }

  function fetchRunDetail(runId) {
    if (!runDetailCache.has(runId)) {
      const promise = fetch(
        `${basePath}data/runs/${encodeURIComponent(runId)}.json?v=${encodeURIComponent(dataVersion)}`
      ).then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      });
      promise.catch(() => runDetailCache.delete(runId));
      runDetailCache.set(runId, promise);
    }
    return runDetailCache.get(runId);
  }

  function discordantCounts(correctnessA, correctnessB) {
    let b = 0;
    let c = 0;
    for (let i = 0; i < correctnessA.length; i += 1) {
      const aCorrect = correctnessA[i] === "1";
      const bCorrect = correctnessB[i] === "1";
      if (aCorrect && !bCorrect) {
        b += 1;
      } else if (!aCorrect && bCorrect) {
        c += 1;
      }
    }
    return { b, c };
  }

  function pairwiseRowHtml(first, second) {
    const pairLabel = `${escapeHtml(modelLabel(first.entry))} vs ${escapeHtml(modelLabel(second.entry))}`;
    if (
      first.entry.dataset_version !== second.entry.dataset_version ||
      first.correctness.length !== second.correctness.length
    ) {
      return `<tr>
        <td>${pairLabel}</td>
        <td colspan="4" class="pairwise-not-significant">Different dataset versions — not comparable item by item.</td>
      </tr>`;
    }
    const { b, c } = discordantCounts(first.correctness, second.correctness);
    const n = first.correctness.length;
    const { p } = mcnemarPValue(b, c);
    const ci = pairedDifferenceCi(b, c, n);
    const significant = p < SIGNIFICANCE_LEVEL;
    const verdict = significant
      ? '<span class="pairwise-significant">significant difference</span>'
      : '<span class="pairwise-not-significant">no significant difference</span>';
    return `<tr>
      <td>${pairLabel}</td>
      <td>${formatPp(ci.diff)} pp</td>
      <td>[${formatPp(ci.low)}, ${formatPp(ci.high)}]</td>
      <td>${b} / ${c} of ${formatNumber(n, 0)}</td>
      <td>${formatPValue(p)} — ${verdict}</td>
    </tr>`;
  }

  function renderPairwise() {
    if (!comparePairwise) {
      return;
    }
    const selectedEntries = state.entries.filter((entry) => state.selected.has(entry.run_id));
    if (selectedEntries.length < 2) {
      comparePairwise.innerHTML = "";
      return;
    }
    const token = ++pairwiseToken;
    if (comparePairwise.innerHTML === "") {
      comparePairwise.innerHTML =
        '<p class="pairwise-caption">Computing pairwise comparison…</p>';
    }
    Promise.allSettled(selectedEntries.map((entry) => fetchRunDetail(entry.run_id))).then(
      (results) => {
        if (token !== pairwiseToken) {
          return;
        }
        const scheme = activeScheme();
        const available = [];
        const unavailable = [];
        results.forEach((result, index) => {
          const entry = selectedEntries[index];
          const bits =
            result.status === "fulfilled"
              ? result.value?.correctness_by_scheme?.[scheme]
              : null;
          if (typeof bits === "string" && bits.length > 0) {
            available.push({ entry, correctness: bits });
          } else {
            unavailable.push(entry);
          }
        });
        const noteHtml =
          unavailable.length > 0
            ? `<p class="pairwise-caption">Per-item data unavailable for: ${unavailable
                .map((entry) => escapeHtml(entry.run_id))
                .join(", ")}.</p>`
            : "";
        if (available.length < 2) {
          comparePairwise.innerHTML = noteHtml;
          return;
        }
        const rows = [];
        for (let i = 0; i < available.length; i += 1) {
          for (let j = i + 1; j < available.length; j += 1) {
            rows.push(pairwiseRowHtml(available[i], available[j]));
          }
        }
        comparePairwise.innerHTML = `<h3>Statistical Comparison</h3>
        <table>
          <thead>
            <tr>
              <th>Pair</th>
              <th>Δ accuracy</th>
              <th>95% CI (pp)</th>
              <th>Discordant items</th>
              <th>McNemar test</th>
            </tr>
          </thead>
          <tbody>${rows.join("")}</tbody>
        </table>
        <p class="pairwise-caption">
          McNemar's test on paired per-item correctness over the shared dataset; Δ accuracy is the
          first run minus the second with a Wald 95% interval. Discordant items count where only
          the first (left) or only the second (right) run is correct. No multiple-comparison
          correction is applied.
        </p>${noteHtml}`;
      }
    );
  }

  function renderBaselines() {
    if (!baselinesTableBody) {
      return;
    }
    baselinesTableBody.innerHTML = state.baselines
      .map((baseline) => {
        const half = ciHalfWidth(baseline);
        const ciHtml =
          half == null
            ? ""
            : `<div class="cell-secondary" title="95% CI: ${formatPercent(
                baseline.accuracy_ci.low
              )} – ${formatPercent(baseline.accuracy_ci.high)}">±${formatPercent(half)}</div>`;
        const labelHtml = baseline.source_url
          ? `<a href="${escapeHtml(baseline.source_url)}" rel="noopener">${escapeHtml(
              baseline.label
            )}</a>`
          : escapeHtml(baseline.label);
        return `<tr>
          <td>
            <div class="cell-primary">${labelHtml}</div>
            <div class="cell-secondary">${escapeHtml(
              baselineKindLabels[baseline.kind] || baseline.kind
            )}</div>
          </td>
          <td>
            <div class="cell-primary">${formatPercent(baseline.accuracy)}</div>${ciHtml}
          </td>
          <td>${escapeHtml(baseline.dataset_version ?? "")}</td>
          <td class="baseline-note">${escapeHtml(baseline.source_note)}</td>
        </tr>`;
      })
      .join("");
  }

  function render() {
    applyScheme();
    renderBaselines();
    const entries = filteredEntries();
    const metric = activeXMetric();
    const scale = activeXScale();
    const points = chartPoints(entries, metric);
    const frontierPoints = paretoFrontier(points);
    const frontierIds = new Set(frontierPoints.map((point) => point.entry.run_id));
    const rows = sortRows(decorateRows(entries, frontierIds));
    renderTable(rows);
    updateSortIndicators();
    renderMainChart(entries, frontierPoints, metric, scale);
    renderCompareCharts();
    renderPairwise();
    updateAccuracyHeaders();
    updateCompareBar();
  }

  // ----- Downloads: CSV / JSON of the current filtered + sorted view -----

  function schemeScore(entry, schemeId) {
    return (entry.scheme_scores && entry.scheme_scores[schemeId]) || null;
  }

  // Column model for the CSV: { header, get(entry) }. Every accuracy column reads from the
  // immutable scheme_scores so the file is identical regardless of the active scheme dropdown.
  function buildExportColumns() {
    const columns = [
      { header: "rank", get: (entry) => entry.rank },
      { header: "run_id", get: (entry) => entry.run_id },
      { header: "run_url", get: (entry) => entry.run_url },
      { header: "created_at", get: (entry) => entry.created_at },
      { header: "git_commit", get: (entry) => entry.git_commit },
      { header: "display_label", get: (entry) => entry.display_label },
      { header: "model", get: (entry) => entry.model },
      { header: "requested_model", get: (entry) => entry.requested_model },
      { header: "resolved_model", get: (entry) => entry.resolved_model },
      { header: "model_url", get: (entry) => entry.model_url },
      { header: "model_kind", get: (entry) => entry.model_kind },
      { header: "llm_vendor", get: (entry) => entry.llm_vendor },
      { header: "family", get: (entry) => entry.family },
      { header: "api_provider", get: (entry) => entry.api_provider },
      { header: "source_kind", get: (entry) => entry.source_kind },
      { header: "license", get: (entry) => entry.license },
      { header: "reasoning_effort", get: (entry) => entry.reasoning_effort },
      { header: "hosting_kind", get: (entry) => entry.hosting_kind },
      { header: "quantization", get: (entry) => entry.quantization },
      { header: "inference_engine", get: (entry) => entry.inference_engine },
      { header: "inference_engine_version", get: (entry) => entry.inference_engine_version },
      { header: "hf_revision", get: (entry) => entry.hf_revision },
      { header: "gpu", get: (entry) => entry.gpu },
      { header: "gpu_count", get: (entry) => entry.gpu_count },
      { header: "hourly_rate_usd", get: (entry) => entry.hourly_rate_usd },
      { header: "prompt_id", get: (entry) => entry.prompt_id },
      { header: "prompt_name", get: (entry) => entry.prompt_name },
      { header: "dataset_id", get: (entry) => entry.dataset_id },
      { header: "dataset_version", get: (entry) => entry.dataset_version },
      { header: "dataset_content_hash", get: (entry) => entry.dataset_content_hash },
      { header: "item_count", get: (entry) => entry.item_count }
    ];
    for (const schemeId of EXPORT_SCHEME_IDS) {
      columns.push({
        header: `${schemeId}_accuracy`,
        get: (entry) => {
          const score = schemeScore(entry, schemeId);
          return score ? score.accuracy : null;
        }
      });
      columns.push({
        header: `${schemeId}_ci_low`,
        get: (entry) => {
          const score = schemeScore(entry, schemeId);
          return score && score.accuracy_ci ? score.accuracy_ci.low : null;
        }
      });
      columns.push({
        header: `${schemeId}_ci_high`,
        get: (entry) => {
          const score = schemeScore(entry, schemeId);
          return score && score.accuracy_ci ? score.accuracy_ci.high : null;
        }
      });
    }
    columns.push({
      header: "lexen_fine_correct_count",
      get: (entry) => {
        const score = schemeScore(entry, DEFAULT_SCHEME_ID);
        return score ? score.correct_count : null;
      }
    });
    const directKeys = [
      "call_count",
      "success_count",
      "monosemous_count",
      "no_candidates_count",
      "no_valid_vote_count",
      "invalid_output_vote_count",
      "transport_error_vote_count",
      "input_tokens",
      "input_uncached_tokens",
      "cached_input_tokens",
      "output_tokens",
      "reasoning_output_tokens",
      "total_tokens",
      "tokens_per_item",
      "cost_source",
      "cost_usd",
      "input_uncached_usd",
      "input_cached_usd",
      "output_usd",
      "input_uncached_unit_price_usd",
      "input_cached_unit_price_usd",
      "output_unit_price_usd",
      "cost_per_million_items",
      "elapsed_seconds",
      "benchmark_seconds",
      "seconds_per_item",
      "machine_hours_per_million_items",
      "concurrency",
      "runner_github_handle",
      "runner_name"
    ];
    for (const key of directKeys) {
      columns.push({ header: key, get: (entry) => entry[key] });
    }
    return columns;
  }

  const EXPORT_COLUMNS = buildExportColumns();

  function csvCell(value) {
    if (value == null) {
      return "";
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    return String(value);
  }

  // RFC-4180 quoting plus a formula-injection guard for text cells (model / runner / prompt
  // names come from pull requests). Numeric cells are never prefixed so they stay numbers.
  function csvField(value, isNumeric) {
    let text = csvCell(value);
    if (!isNumeric && /^[=+\-@\t\r]/.test(text)) {
      text = `'${text}`;
    }
    if (/[",\n\r]/.test(text)) {
      text = `"${text.replaceAll('"', '""')}"`;
    }
    return text;
  }

  function toCsv(rows, columns) {
    const header = columns.map((column) => csvField(column.header, false)).join(",");
    const lines = rows.map((entry) =>
      columns
        .map((column) => {
          const value = column.get(entry);
          return csvField(value, typeof value === "number");
        })
        .join(",")
    );
    return [header, ...lines].join("\r\n") + "\r\n";
  }

  // The exact rows the user currently sees: filters + "best per model" view + active sort.
  function exportRows() {
    const entries = filteredEntries();
    const points = chartPoints(entries, activeXMetric());
    const frontierIds = new Set(paretoFrontier(points).map((point) => point.entry.run_id));
    return sortRows(decorateRows(entries, frontierIds)).map((row) => row.entry);
  }

  function cloneEntry(entry) {
    if (typeof structuredClone === "function") {
      return structuredClone(entry);
    }
    return JSON.parse(JSON.stringify(entry));
  }

  // Faithful per-entry dump for JSON: clone (so state.entries is untouched), then pin the
  // top-level accuracy fields to the default scheme so they are deterministic; scheme_scores
  // already carries all six. Drop presentation-only keys.
  function cleanEntryForJson(entry) {
    const clone = cloneEntry(entry);
    const headline = schemeScore(clone, DEFAULT_SCHEME_ID);
    if (headline) {
      clone.accuracy = headline.accuracy;
      clone.accuracy_ci = headline.accuracy_ci;
      clone.correct_count = headline.correct_count;
    }
    delete clone.logo_slug;
    delete clone.vendor_initial;
    return clone;
  }

  function triggerDownload(filename, text, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function downloadStamp() {
    return new Date().toISOString().slice(0, 10);
  }

  function downloadCsv() {
    // Leading BOM so Excel reads the file as UTF-8.
    const csv = "﻿" + toCsv(exportRows(), EXPORT_COLUMNS);
    triggerDownload(
      `sensebench-leaderboard-${downloadStamp()}.csv`,
      csv,
      "text/csv;charset=utf-8"
    );
  }

  function downloadJson() {
    const rows = exportRows();
    const payload = {
      schema_version: state.dataSchemaVersion,
      exported_at: new Date().toISOString(),
      source_generated_at: state.generatedAt,
      row_count: rows.length,
      default_scheme: DEFAULT_SCHEME_ID,
      entries: rows.map(cleanEntryForJson)
    };
    triggerDownload(
      `sensebench-leaderboard-${downloadStamp()}.json`,
      JSON.stringify(payload, null, 2),
      "application/json;charset=utf-8"
    );
  }

  function attachControls() {
    [
      searchInput,
      datasetFilter,
      promptFilter,
      sourceFilter,
      hostingFilter,
      gpuFilter,
      quantFilter,
      schemeGoldSelect,
      schemeGranularitySelect,
      xMetricSelect,
      xScaleSelect,
      maxCostFilter,
      viewFilter,
      frontierOnly
    ].forEach((control) => {
      if (control) {
        control.addEventListener("input", render);
        control.addEventListener("change", render);
      }
    });
    table.querySelectorAll("[data-sort]").forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.sort;
        if (!key) {
          return;
        }
        if (state.sortKey === key) {
          state.sortDirection *= -1;
        } else {
          state.sortKey = key;
          state.sortDirection = key === "rank" || key === "model" ? 1 : -1;
        }
        render();
      });
    });
    if (sortSelect) {
      sortSelect.addEventListener("change", () => {
        const key = sortSelect.value;
        state.sortKey = key;
        state.sortDirection = key === "rank" || key === "model" ? 1 : -1;
        render();
      });
    }
    if (downloadCsvButton) {
      downloadCsvButton.addEventListener("click", downloadCsv);
    }
    if (downloadJsonButton) {
      downloadJsonButton.addEventListener("click", downloadJson);
    }
    if (compareBarClear) {
      compareBarClear.addEventListener("click", () => {
        state.selected.clear();
        render();
      });
    }
    window.addEventListener("resize", () => {
      if (mainChart) {
        mainChart.resize();
      }
      if (compareCharts && window.echarts) {
        compareCharts.querySelectorAll(".compare-metric-chart").forEach((element) => {
          const instance = window.echarts.getInstanceByDom(element);
          if (instance) {
            instance.resize();
          }
        });
      }
    });
  }

  fetch(`${basePath}data/leaderboard.json?v=${encodeURIComponent(dataVersion)}`)
    .then((response) => response.json())
    .then((data) => {
      state.entries = data.entries || [];
      state.baselines = data.baselines || [];
      state.dataSchemaVersion = data.schema_version || "";
      state.generatedAt = (data.summary && data.summary.generated_at) || null;
      attachControls();
      const canDownload = state.entries.length > 0;
      if (downloadCsvButton) {
        downloadCsvButton.disabled = !canDownload;
      }
      if (downloadJsonButton) {
        downloadJsonButton.disabled = !canDownload;
      }
      render();
    })
    .catch(() => {
      attachControls();
    });
})();
