// Must stay in sync with common/preprocessing.py (FINAL_FEATURE_ORDER, PROTOCOL_MAP, FLAG_MAP).
const FEATURES = [
  "duration", "protocol_type", "flag", "src_bytes", "dst_bytes", "land",
  "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
  "num_compromised", "root_shell", "su_attempted", "num_file_creations",
  "num_shells", "num_access_files", "is_guest_login", "count", "srv_count",
  "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
  "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
  "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
  "dst_host_srv_diff_host_rate",
];

const PROTOCOLS = ["icmp", "tcp", "udp"];
const FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "RSTOS0", "S3", "OTH"];

const CLASS_COLOR_CLASS = { normal: "cls-normal", dos: "cls-dos", probe: "cls-probe", r2l: "cls-r2l", u2r: "cls-u2r" };

const fieldsContainer = document.getElementById("fields");
const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const resultPanel = document.getElementById("result-panel");
const resultEl = document.getElementById("result");
const statusEl = document.getElementById("model-status");
const presetsEl = document.getElementById("presets");

function buildFields() {
  for (const name of FEATURES) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.textContent = name;
    label.htmlFor = `f-${name}`;
    wrapper.appendChild(label);

    let input;
    if (name === "protocol_type" || name === "flag") {
      input = document.createElement("select");
      const options = name === "protocol_type" ? PROTOCOLS : FLAGS;
      for (const opt of options) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        input.appendChild(o);
      }
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = "0";
    }
    input.id = `f-${name}`;
    input.name = name;
    wrapper.appendChild(input);
    fieldsContainer.appendChild(wrapper);
  }
}

function fillForm(values) {
  for (const name of FEATURES) {
    const el = document.getElementById(`f-${name}`);
    if (el && values[name] !== undefined) el.value = values[name];
  }
}

function readForm() {
  const payload = {};
  for (const name of FEATURES) {
    const el = document.getElementById(`f-${name}`);
    payload[name] = el.tagName === "SELECT" ? el.value : Number(el.value);
  }
  return payload;
}

async function loadPresets() {
  try {
    const res = await fetch("/examples.json");
    const examples = await res.json();
    for (const cls of Object.keys(examples)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = cls.toUpperCase();
      btn.addEventListener("click", () => fillForm(examples[cls]));
      presetsEl.appendChild(btn);
    }
  } catch (err) {
    presetsEl.textContent = "Could not load sample presets.";
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (res.ok && data.status === "ok") {
      const src = data.model.source;
      statusEl.textContent = `Model ready (${src} data, ${data.model.test_accuracy_pct}% test accuracy)` +
        (src === "synthetic" ? " — demo model, retrain on real data for production use." : "");
      statusEl.classList.add("ok");
    } else {
      statusEl.textContent = "Model artifacts missing — run model/train.py and redeploy.";
      statusEl.classList.add("error");
    }
  } catch (err) {
    statusEl.textContent = "Could not reach /api/health.";
    statusEl.classList.add("error");
  }
}

function renderResult(data) {
  resultPanel.hidden = false;
  const colorClass = CLASS_COLOR_CLASS[data.prediction] || "";
  const rows = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, p]) => `
      <div class="prob-row">
        <span class="prob-label">${cls}</span>
        <span class="prob-track"><span class="prob-fill ${CLASS_COLOR_CLASS[cls] || ""}" style="width:${(p * 100).toFixed(1)}%"></span></span>
        <span class="prob-value">${(p * 100).toFixed(1)}%</span>
      </div>`)
    .join("");
  resultEl.innerHTML = `<div class="verdict ${colorClass}">Predicted: ${data.prediction.toUpperCase()}</div>${rows}`;
}

function renderError(message) {
  resultPanel.hidden = false;
  resultEl.innerHTML = `<div class="error-box">${message}</div>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const data = await res.json();
    if (!res.ok) {
      renderError(data.error || `Request failed (${res.status})`);
    } else {
      renderResult(data);
    }
  } catch (err) {
    renderError("Network error: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});

buildFields();
loadPresets();
checkHealth();
