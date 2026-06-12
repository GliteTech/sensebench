(function () {
  const basePath = window.SENSEBENCH_BASE_PATH || "/";
  const table = document.getElementById("leaderboard-table");
  const searchInput = document.getElementById("leaderboard-search");
  const datasetFilter = document.getElementById("dataset-filter");
  const promptFilter = document.getElementById("prompt-filter");
  const sourceFilter = document.getElementById("source-filter");
  const maxCostFilter = document.getElementById("max-cost-filter");
  const viewFilter = document.getElementById("view-filter");
  const frontierOnly = document.getElementById("frontier-only");
  const chartElement = document.getElementById("leaderboard-chart");
  const chartNote = document.getElementById("chart-note");
  const compareCharts = document.getElementById("compare-charts");
  const compareEmpty = document.getElementById("compare-empty");
  const comparePairwise = document.getElementById("compare-pairwise");
  const compareTable = document.getElementById("compare-table");
  const dataVersion = window.SENSEBENCH_DATA_VERSION || "";

  if (!table) {
    return;
  }

  const state = {
    entries: [],
    baselines: [],
    selected: new Set(),
    sortKey: "rank",
    sortDirection: 1
  };

  const runDetailCache = new Map();
  let pairwiseToken = 0;
  let mainChart = null;

  const SIGNIFICANCE_LEVEL = 0.05;
  const EXACT_MCNEMAR_MAX_DISCORDANT = 25;
  const Z_95 = 1.959963984540054;

  const COST_METRIC = "cost_per_million_items";
  const COST_AXIS_LABEL = "Cost per million items, USD";

  const sourceLabels = {
    open_source: "Open weights",
    proprietary: "Proprietary"
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

  function modelLabel(entry) {
    if (entry.reasoning_effort) {
      return `${entry.model} (${entry.reasoning_effort})`;
    }
    return entry.model;
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
        const frontierHtml = row.onFrontier
          ? '<span class="badge badge-frontier" title="On the accuracy-cost Pareto frontier">★</span>'
          : "";
        return `<tr>
          <td><div class="cell-primary">${row.displayRank}</div>${rangeHtml}</td>
          <td><input class="compare-checkbox" type="checkbox" data-run-id="${escapeHtml(entry.run_id)}"${checked}${disabled}></td>
          <td>
            <div class="cell-primary">${escapeHtml(modelLabel(entry))}</div>
            <div class="cell-secondary">${vendorParts.join(" · ")}</div>
          </td>
          <td><div class="cell-primary">${formatPercent(entry.accuracy)}</div>${ciHtml}</td>
          <td>${formatMoney(entry.cost_per_million_items)}</td>
          <td>${formatNumber(entry.tokens_per_item, 1)}</td>
          <td>${escapeHtml(entry.prompt_id)}</td>
          <td>${escapeHtml(entry.dataset_version)}</td>
          <td><a href="${basePath}${escapeHtml(entry.run_url)}">${escapeHtml(entry.run_id)}</a></td>
          <td>${frontierHtml}</td>
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

  function chartPoints(entries) {
    return entries
      .filter((entry) => entry.accuracy != null && entry[COST_METRIC] != null)
      .map((entry) => ({
        x: entry[COST_METRIC],
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

  function renderMainChart(entries, frontierPoints) {
    if (!chartElement || !window.echarts) {
      return;
    }
    const points = chartPoints(entries);
    const frontier = [...frontierPoints].sort((a, b) => a.x - b.x);
    const visible = frontierOnly?.checked ? frontier : points;
    const baselines = visibleBaselines();
    if (!mainChart) {
      mainChart = window.echarts.init(chartElement);
    }
    const yValues = [
      ...visible.map((point) => point.y),
      ...baselines.map((baseline) => baseline.accuracy * 100)
    ];
    const yMin = yValues.length > 0 ? Math.max(0, Math.floor(Math.min(...yValues)) - 1) : 0;
    mainChart.setOption(
      {
        animation: false,
        grid: { left: 58, right: 24, top: 28, bottom: 62 },
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
            return [
              `<strong>${escapeHtml(modelLabel(entry))}</strong>`,
              `Prompt: ${escapeHtml(promptLabel(entry))}`,
              accuracyText,
              `${COST_AXIS_LABEL}: ${formatMoney(entry[COST_METRIC])}`
            ].join("<br>");
          }
        },
        xAxis: {
          name: COST_AXIS_LABEL,
          nameLocation: "middle",
          nameGap: 42,
          type: "value"
        },
        yAxis: {
          name: "Accuracy %",
          type: "value",
          min: yMin
        },
        series: [
          {
            name: "Runs",
            type: "scatter",
            symbolSize: 10,
            data: visible.map((point) => ({
              value: [point.x, point.y],
              entry: point.entry
            })),
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
          },
          {
            name: "Pareto frontier",
            type: "line",
            showSymbol: true,
            symbolSize: 12,
            lineStyle: { width: 2 },
            label: {
              show: true,
              position: "top",
              fontSize: 11,
              formatter: (params) =>
                params.data.entry ? modelLabel(params.data.entry) : ""
            },
            labelLayout: { hideOverlap: true },
            data: frontier.map((point) => ({
              value: [point.x, point.y],
              entry: point.entry
            }))
          }
        ]
      },
      { notMerge: true }
    );
    const missing = entries.length - points.length;
    if (chartNote) {
      const baselineNote =
        baselines.length > 0
          ? ` Dashed lines mark reference baselines scored on the same items.`
          : "";
      chartNote.textContent = `${points.length} rows plotted. ${missing} rows have unavailable values for this chart.${baselineNote}`;
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
          <th>Accuracy</th>
          <th>Cost / M items</th>
          <th>Tokens / item</th>
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
            return `<tr>
              <td>
                <div class="cell-primary">${escapeHtml(modelLabel(entry))}</div>
                <div class="cell-secondary">${escapeHtml(entry.prompt_id)} · ${escapeHtml(entry.dataset_version ?? "")}</div>
              </td>
              <td><div class="cell-primary">${formatPercent(entry.accuracy)}</div>${ciHtml}</td>
              <td>${formatMoney(entry.cost_per_million_items)}</td>
              <td>${formatNumber(entry.tokens_per_item, 1)}</td>
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
        const available = [];
        const unavailable = [];
        results.forEach((result, index) => {
          const entry = selectedEntries[index];
          if (
            result.status === "fulfilled" &&
            typeof result.value?.correctness === "string" &&
            result.value.correctness.length > 0
          ) {
            available.push({ entry, correctness: result.value.correctness });
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

  function render() {
    const entries = filteredEntries();
    const points = chartPoints(entries);
    const frontierPoints = paretoFrontier(points);
    const frontierIds = new Set(frontierPoints.map((point) => point.entry.run_id));
    const rows = sortRows(decorateRows(entries, frontierIds));
    renderTable(rows);
    renderMainChart(entries, frontierPoints);
    renderCompareCharts();
    renderPairwise();
  }

  function attachControls() {
    [
      searchInput,
      datasetFilter,
      promptFilter,
      sourceFilter,
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
      attachControls();
      render();
    })
    .catch(() => {
      attachControls();
    });
})();
