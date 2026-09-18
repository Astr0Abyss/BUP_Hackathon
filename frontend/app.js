"use strict";

const elements = {
  apiDot: document.querySelector("#api-dot"),
  apiStatus: document.querySelector("#api-status"),
  sampleSelect: document.querySelector("#sample-select"),
  requestJson: document.querySelector("#request-json"),
  runButton: document.querySelector("#run-button"),
  httpStatus: document.querySelector("#http-status"),
  latency: document.querySelector("#latency"),
  message: document.querySelector("#request-message"),
  scenarioId: document.querySelector("#scenario-id"),
  totalGrid: document.querySelector("#total-grid"),
  totalCost: document.querySelector("#total-cost"),
  peakGrid: document.querySelector("#peak-grid"),
  directiveCount: document.querySelector("#directive-count"),
  directives: document.querySelector("#directives"),
  planBody: document.querySelector("#plan-body"),
  rawJson: document.querySelector("#raw-json"),
};

let samples = [];

function formatNumber(value) {
  return typeof value === "number"
    ? new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value)
    : "—";
}

function setApiState(ok, label) {
  elements.apiDot.className = `status-dot ${ok ? "ok" : "error"}`;
  elements.apiStatus.textContent = label;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const body = await response.json();
    setApiState(response.ok && body.status === "ok", response.ok ? body.status : `HTTP ${response.status}`);
  } catch (_error) {
    setApiState(false, "offline");
  }
}

function selectSample(index) {
  const sample = samples[index];
  if (sample) elements.requestJson.value = JSON.stringify(sample.input, null, 2);
}

async function loadSamples() {
  try {
    const response = await fetch("/sample-cases");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    samples = payload.cases;
    elements.sampleSelect.replaceChildren(...samples.map((sample, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${sample.id} — ${sample.label}`;
      return option;
    }));
    elements.sampleSelect.disabled = false;
    elements.runButton.disabled = false;
    selectSample(0);
  } catch (error) {
    elements.sampleSelect.replaceChildren(new Option("Samples unavailable", ""));
    elements.message.textContent = `Could not load public samples: ${error.message}`;
    elements.message.className = "request-message error";
  }
}

function renderDirectives(items = []) {
  elements.directiveCount.textContent = `${items.length} ${items.length === 1 ? "note" : "notes"}`;
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No directive interpretations returned.";
    elements.directives.replaceChildren(empty);
    return;
  }

  elements.directives.replaceChildren(...items.map((item) => {
    const card = document.createElement("article");
    card.className = `directive-card${item.directive_type === "no_op" ? " no-op" : ""}`;
    const header = document.createElement("header");
    const type = document.createElement("strong");
    type.textContent = item.directive_type.replaceAll("_", " ");
    const index = document.createElement("span");
    index.textContent = `NOTE ${item.note_index + 1}`;
    header.append(type, index);
    const explanation = document.createElement("p");
    explanation.textContent = item.explanation;
    const adjustment = document.createElement("code");
    adjustment.textContent = item.structured_adjustment
      ? JSON.stringify(item.structured_adjustment)
      : "No schedule adjustment";
    card.append(header, explanation, adjustment);
    return card;
  }));
}

function renderPlan(rows = []) {
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty-cell";
    cell.textContent = "No plan returned.";
    row.append(cell);
    elements.planBody.replaceChildren(row);
    return;
  }

  elements.planBody.replaceChildren(...rows.map((item) => {
    const row = document.createElement("tr");
    const values = [
      `${String(item.hour).padStart(2, "0")}:00`,
      formatNumber(item.grid_kwh),
      formatNumber(item.solar_used_kwh),
      item.battery_action,
      formatNumber(item.battery_kwh),
      formatNumber(item.battery_energy_after_kwh),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 3) cell.className = `action ${item.battery_action}`;
      row.append(cell);
    });
    return row;
  }));
}

function renderResponse(body) {
  elements.scenarioId.textContent = body.scenario_id || "Unknown scenario";
  elements.totalGrid.textContent = formatNumber(body.total_grid_kwh);
  elements.totalCost.textContent = formatNumber(body.total_cost_bdt);
  elements.peakGrid.textContent = formatNumber(body.peak_grid_kwh);
  renderDirectives(body.directive_interpretation);
  renderPlan(body.hourly_plan);
  elements.rawJson.textContent = JSON.stringify(body, null, 2);
}

async function runOptimization() {
  elements.message.textContent = "";
  elements.message.className = "request-message";
  let request;
  try {
    request = JSON.parse(elements.requestJson.value);
  } catch (error) {
    elements.message.textContent = `Request JSON is invalid: ${error.message}`;
    elements.message.className = "request-message error";
    return;
  }

  elements.runButton.disabled = true;
  elements.runButton.firstElementChild.textContent = "Optimizing…";
  elements.httpStatus.textContent = "pending";
  elements.latency.textContent = "—";
  const startedAt = performance.now();

  try {
    const response = await fetch("/optimize-energy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const elapsed = performance.now() - startedAt;
    const body = await response.json();
    elements.httpStatus.textContent = String(response.status);
    elements.latency.textContent = `${Math.round(elapsed)} ms`;
    elements.rawJson.textContent = JSON.stringify(body, null, 2);
    if (!response.ok) {
      const message = body?.error?.message || body?.detail || "Optimization failed.";
      throw new Error(typeof message === "string" ? message : JSON.stringify(message));
    }
    renderResponse(body);
  } catch (error) {
    elements.message.textContent = error.message;
    elements.message.className = "request-message error";
    if (elements.httpStatus.textContent === "pending") {
      elements.httpStatus.textContent = "network error";
      elements.latency.textContent = `${Math.round(performance.now() - startedAt)} ms`;
    }
  } finally {
    elements.runButton.disabled = false;
    elements.runButton.firstElementChild.textContent = "Run optimization";
  }
}

elements.sampleSelect.addEventListener("change", (event) => selectSample(Number(event.target.value)));
elements.runButton.addEventListener("click", runOptimization);
checkHealth();
loadSamples();
