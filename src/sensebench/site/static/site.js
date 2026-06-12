(function () {
  const basePath = window.SENSEBENCH_BASE_PATH || "/";
  const table = document.getElementById("leaderboard-table");
  const searchInput = document.getElementById("leaderboard-search");
  const datasetFilter = document.getElementById("dataset-filter");
  const promptFilter = document.getElementById("prompt-filter");
  const viewFilter = document.getElementById("view-filter");
  const chartMode = document.getElementById("chart-mode");
  const frontierOnly = document.getElementById("frontier-only");
  const chartElement = document.getElementById("leaderboard-chart");
  const chartNote = document.getElementById("chart-note");
  const compareCharts = document.getElementById("compare-charts");
  const compareEmpty = document.getElementById("compare-empty");
  const compareTable = document.getElementById("compare-table");
  const dataVersion = window.SENSEBENCH_DATA_VERSION || "";

  if (!table) {
    return;
  }

  const state = {
    entries: [],
    selected: new Set(),
    sortKey: "rank",
    sortDirection: 1
  };

  const metricLabels = {
    cost_per_million_items: "Cost per million items, USD",
    tokens_per_item: "Tokens per item"
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
    const digits = value < 1 ? 4 : 2;
    return `$${Number(value).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    })}`;
  }

  function searchableText(entry) {
    return [
      entry.model,
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

  function filteredEntries() {
    const query = (searchInput?.value || "").trim().toLowerCase();
    const dataset = datasetFilter?.value || "";
    const prompt = promptFilter?.value || "";
    let entries = state.entries.filter((entry) => {
      if (dataset && entry.dataset_version !== dataset) {
        return false;
      }
      if (prompt && entry.prompt_id !== prompt) {
        return false;
      }
      return !query || searchableText(entry).includes(query);
    });
    if ((viewFilter?.value || "best") === "best") {
      entries = bestPerModel(entries);
    }
    return sortEntries(entries);
  }

  function sortEntries(entries) {
    const key = state.sortKey;
    const direction = state.sortDirection;
    return [...entries].sort((a, b) => {
      const left = a[key];
      const right = b[key];
      if (left == null && right == null) {
        return a.run_id.localeCompare(b.run_id);
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

  function renderTable(entries) {
    const tbody = table.querySelector("tbody");
    if (!tbody) {
      return;
    }
    tbody.innerHTML = entries
      .map((entry) => {
        const checked = state.selected.has(entry.run_id) ? " checked" : "";
        const disabled = !checked && state.selected.size >= 6 ? " disabled" : "";
        return `<tr>
          <td>${entry.rank}</td>
          <td><input class="compare-checkbox" type="checkbox" data-run-id="${escapeHtml(entry.run_id)}"${checked}${disabled}></td>
          <td>${escapeHtml(entry.model)}</td>
          <td>${formatPercent(entry.accuracy)}</td>
          <td>${formatMoney(entry.cost_per_million_items)}</td>
          <td>${formatNumber(entry.tokens_per_item, 1)}</td>
          <td>${escapeHtml(entry.prompt_id)}</td>
          <td>${escapeHtml(entry.dataset_version)}</td>
          <td><a href="${basePath}${escapeHtml(entry.run_url)}">${escapeHtml(entry.run_id)}</a></td>
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

  function chartPoints(entries, metric) {
    return entries
      .filter((entry) => entry.accuracy != null && entry[metric] != null)
      .map((entry) => ({
        x: entry[metric],
        y: entry.accuracy * 100,
        entry
      }));
  }

  function renderMainChart(entries) {
    if (!chartElement || !window.echarts) {
      return;
    }
    const metric = chartMode?.value || "cost_per_million_items";
    const points = chartPoints(entries, metric);
    const frontier = paretoFrontier(points).sort((a, b) => a.x - b.x);
    const visible = frontierOnly?.checked ? frontier : points;
    const chart = window.echarts.init(chartElement);
    chart.setOption({
      animation: false,
      grid: { left: 58, right: 24, top: 28, bottom: 62 },
      tooltip: {
        trigger: "item",
        formatter: (params) => {
          const entry = params.data.entry;
          return [
            `<strong>${escapeHtml(entry.model)}</strong>`,
            escapeHtml(entry.run_id),
            `Accuracy: ${formatPercent(entry.accuracy)}`,
            `${metricLabels[metric]}: ${formatMetric(metric, entry[metric])}`
          ].join("<br>");
        }
      },
      xAxis: {
        name: metricLabels[metric],
        nameLocation: "middle",
        nameGap: 42,
        type: "value"
      },
      yAxis: {
        name: "Accuracy %",
        type: "value",
        min: "dataMin"
      },
      series: [
        {
          name: "Runs",
          type: "scatter",
          symbolSize: 10,
          data: visible.map((point) => ({
            value: [point.x, point.y],
            entry: point.entry
          }))
        },
        {
          name: "Pareto frontier",
          type: "line",
          showSymbol: true,
          symbolSize: 12,
          lineStyle: { width: 2 },
          data: frontier.map((point) => ({
            value: [point.x, point.y],
            entry: point.entry
          }))
        }
      ]
    });
    const missing = entries.length - points.length;
    if (chartNote) {
      chartNote.textContent = `${points.length} rows plotted. ${missing} rows have unavailable values for this chart.`;
    }
    window.addEventListener("resize", () => chart.resize(), { once: true });
  }

  function formatMetric(metric, value) {
    if (metric === "cost_per_million_items") {
      return formatMoney(value);
    }
    return formatNumber(value, 2);
  }

  function renderCompareCharts() {
    if (!compareCharts || !window.echarts) {
      return;
    }
    const selectedEntries = state.entries.filter((entry) => state.selected.has(entry.run_id));
    if (compareEmpty) {
      compareEmpty.style.display = selectedEntries.length === 0 ? "block" : "none";
    }
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
      const chart = window.echarts.init(element);
      chart.setOption({
        animation: false,
        grid: { left: 58, right: 16, top: 12, bottom: 72 },
        tooltip: {
          trigger: "axis",
          formatter: (params) => {
            const point = params[0];
            const entry = selectedEntries[point.dataIndex];
            return [
              `<strong>${escapeHtml(entry.model)}</strong>`,
              escapeHtml(entry.run_id),
              `${escapeHtml(metric.title)}: ${metric.format(point.value)}`
            ].join("<br>");
          }
        },
        xAxis: {
          type: "category",
          data: selectedEntries.map((entry) => entry.model),
          axisLabel: { interval: 0, rotate: 20 }
        },
        yAxis: {
          name: metric.title,
          type: "value",
          nameGap: 42
        },
        series: [
          {
            name: metric.title,
            type: "bar",
            data: selectedEntries.map((entry) => metric.value(entry))
          }
        ]
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
          .map(
            (entry) => `<tr>
              <td>${escapeHtml(entry.model)}</td>
              <td>${formatPercent(entry.accuracy)}</td>
              <td>${formatMoney(entry.cost_per_million_items)}</td>
              <td>${formatNumber(entry.tokens_per_item, 1)}</td>
              <td><a href="${basePath}${escapeHtml(entry.run_url)}">${escapeHtml(entry.run_id)}</a></td>
            </tr>`
          )
          .join("")}
      </tbody>
    </table>`;
  }

  function render() {
    const entries = filteredEntries();
    renderTable(entries);
    renderMainChart(entries);
    renderCompareCharts();
  }

  function attachControls() {
    [searchInput, datasetFilter, promptFilter, viewFilter, chartMode, frontierOnly].forEach((control) => {
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
  }

  fetch(`${basePath}data/leaderboard.json?v=${encodeURIComponent(dataVersion)}`)
    .then((response) => response.json())
    .then((data) => {
      state.entries = data.entries || [];
      attachControls();
      render();
    })
    .catch(() => {
      attachControls();
    });
})();
