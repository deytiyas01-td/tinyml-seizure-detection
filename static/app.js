// EEG Seizure Detection Clinical Frontend Application

let currentChart = null;
let referenceExamples = [];
let activeExampleIndex = 0;

let API_BASE_URL = localStorage.getItem("seizure_api_base") || "";

function getApiUrl(endpoint) {
  if (!API_BASE_URL) return endpoint;
  return API_BASE_URL.replace(/\/+$/, "") + endpoint;
}

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  setupApiConfig();
  fetchExamples();
});

function setupApiConfig() {
  const btnConfig = document.getElementById("btnConfigApi");
  const configBar = document.getElementById("apiConfigBar");
  const input = document.getElementById("apiUrlInput");
  const btnSave = document.getElementById("btnSaveApiUrl");
  const btnReset = document.getElementById("btnResetApiUrl");

  if (input) input.value = API_BASE_URL;

  if (btnConfig && configBar) {
    btnConfig.addEventListener("click", () => {
      configBar.style.display = configBar.style.display === "none" ? "block" : "none";
    });
  }

  if (btnSave && input) {
    btnSave.addEventListener("click", () => {
      API_BASE_URL = input.value.trim();
      localStorage.setItem("seizure_api_base", API_BASE_URL);
      if (configBar) configBar.style.display = "none";
      fetchExamples();
    });
  }

  if (btnReset && input) {
    btnReset.addEventListener("click", () => {
      API_BASE_URL = "";
      input.value = "";
      localStorage.removeItem("seizure_api_base");
      if (configBar) configBar.style.display = "none";
      fetchExamples();
    });
  }
}

function setupEventListeners() {
  const btnCustom = document.getElementById("btnDetectCustom");
  if (btnCustom) {
    btnCustom.addEventListener("click", handleCustomInput);
  }

  const btnReset = document.getElementById("btnRestorePreset");
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      if (referenceExamples[activeExampleIndex]) {
        selectExample(activeExampleIndex);
      }
    });
  }
}

// Fetch 4 test examples from backend
async function fetchExamples() {
  const container = document.getElementById("examplesContainer");
  const statusIndicator = document.getElementById("serverStatusIndicator");
  const statusText = document.getElementById("serverStatusText");

  try {
    const res = await fetch(getApiUrl("/api/examples"));
    const json = await res.json();
    if (json.status === "success" && json.examples) {
      referenceExamples = json.examples;
      renderExamplesList(referenceExamples);
      if (statusIndicator) statusIndicator.style.backgroundColor = "#22c55e";
      if (statusText) statusText.textContent = "Backend Connected";
      if (referenceExamples.length > 0) {
        selectExample(0);
      }
    } else {
      container.innerHTML = `<div class="empty-cell">Failed to load example data.</div>`;
    }
  } catch (err) {
    console.error("Failed to load reference examples:", err);
    if (statusIndicator) statusIndicator.style.backgroundColor = "#ef4444";
    if (statusText) statusText.textContent = "Backend Disconnected";
    container.innerHTML = `<div class="empty-cell">Server connection error. If your backend is deployed on Render, click the ⚙️ icon above to enter your Render backend URL.</div>`;
  }
}

// Render the 4 test sample cards
function renderExamplesList(examples) {
  const container = document.getElementById("examplesContainer");
  container.innerHTML = "";

  examples.forEach((ex, idx) => {
    const card = document.createElement("div");
    card.className = `example-item ${idx === 0 ? "active" : ""}`;
    card.id = `example-item-${idx}`;

    const isSeizure = ex.type === "seizure";
    const tagClass = isSeizure ? "tag-seizure" : "tag-non-seizure";
    const tagLabel = isSeizure ? "Seizure" : "Non-Seizure";

    card.innerHTML = `
      <div class="example-top">
        <span class="example-name">${ex.name}</span>
        <span class="example-tag ${tagClass}">${tagLabel}</span>
      </div>
      <p class="example-desc">${ex.description}</p>
      <button class="btn-select" onclick="selectExample(${idx})">Load & Evaluate</button>
    `;
    container.appendChild(card);
  });
}

window.selectExample = function(idx) {
  if (!referenceExamples[idx]) return;
  activeExampleIndex = idx;

  document.querySelectorAll(".example-item").forEach((item, i) => {
    item.classList.toggle("active", i === idx);
  });

  const ex = referenceExamples[idx];
  const labelEl = document.getElementById("currentSampleName");
  if (labelEl) {
    labelEl.textContent = `Sample: ${ex.name} (${ex.type === "seizure" ? "Seizure" : "Non-Seizure"})`;
  }

  const field = document.getElementById("sequenceField");
  if (field) {
    field.value = JSON.stringify(ex.data, null, 2);
  }

  renderWaveform(ex.data, ex.channels);
  runDetection(ex.data);
};

// Render multi-channel waveform on clinical white background
function renderWaveform(channelData4x128, channelLabels) {
  const canvas = document.getElementById("waveformCanvas");
  const ctx = canvas.getContext("2d");

  const timeAxis = Array.from({ length: 128 }, (_, i) => (i / 64.0).toFixed(2));

  // Professional clinical EEG trace colors on white canvas
  const traceColors = ["#0284c7", "#2563eb", "#4f46e5", "#7c3aed"];
  const verticalOffsets = [6.0, 2.0, -2.0, -6.0];

  const datasets = channelData4x128.map((chArr, i) => {
    const name = (channelLabels && channelLabels[i]) ? channelLabels[i] : `Channel ${i+1}`;
    const offset = verticalOffsets[i] || 0;
    return {
      label: `${name} (${offset >= 0 ? '+' : ''}${offset} μV)`,
      data: chArr.map(v => v + offset),
      borderColor: traceColors[i % traceColors.length],
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.15,
      fill: false
    };
  });

  if (currentChart) {
    currentChart.destroy();
  }

  currentChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: timeAxis,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 250
      },
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "#334155",
            font: { family: "Roboto Mono", size: 11 },
            boxWidth: 12
          }
        },
        tooltip: {
          backgroundColor: "#ffffff",
          titleColor: "#0f172a",
          bodyColor: "#334155",
          borderColor: "#cbd5e1",
          borderWidth: 1,
          padding: 8
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Time (seconds) @ 64 Hz",
            color: "#64748b",
            font: { size: 11 }
          },
          grid: { color: "#f1f5f9" },
          ticks: { color: "#64748b", maxTicksLimit: 10, font: { family: "Roboto Mono" } }
        },
        y: {
          title: {
            display: true,
            text: "Standardized Amplitude (μV) [Stacked]",
            color: "#64748b",
            font: { size: 11 }
          },
          grid: { color: "#f1f5f9" },
          ticks: { color: "#64748b", font: { family: "Roboto Mono" } }
        }
      }
    }
  });
}

// Call backend prediction endpoint
async function runDetection(sequenceMatrix) {
  const displayBox = document.getElementById("resultDisplayBox");
  const statusIcon = document.getElementById("resultStatusIcon");
  const resultTitle = document.getElementById("resultTitle");
  const resultDesc = document.getElementById("resultDescription");
  const confText = document.getElementById("confidencePercent");
  const confBar = document.getElementById("confidenceBar");
  const probText = document.getElementById("rawProbability");
  const timestampTag = document.getElementById("evaluationTimestamp");

  try {
    const res = await fetch(getApiUrl("/api/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: sequenceMatrix })
    });

    const data = await res.json();
    if (!res.ok || data.error) {
      alert(`Evaluation Error: ${data.error || 'Server returned invalid response.'}`);
      return;
    }

    const isSeizure = data.is_seizure;
    const confidence = data.confidence_percentage;
    const probability = data.probability;

    // Classification Box
    displayBox.className = `result-display ${isSeizure ? 'status-seizure' : 'status-normal'}`;
    statusIcon.textContent = isSeizure ? "!" : "✓";
    resultTitle.textContent = isSeizure ? "Seizure Activity Detected" : "No Seizure Detected (Normal EEG)";
    resultDesc.textContent = isSeizure
      ? "Epileptiform discharges identified in the current EEG window."
      : "Baseline background activity within normal limits.";

    // Confidence
    confText.textContent = `${confidence.toFixed(2)}%`;
    confBar.style.width = `${Math.min(100, Math.max(8, confidence))}%`;
    confBar.className = `meter-bar ${isSeizure ? 'bar-seizure' : 'bar-normal'}`;
    probText.textContent = probability.toFixed(4);

    // Spectrum
    if (data.frequency_bands) {
      document.getElementById("deltaVal").textContent = `${data.frequency_bands.delta}%`;
      document.getElementById("deltaBar").style.width = `${data.frequency_bands.delta}%`;

      document.getElementById("thetaVal").textContent = `${data.frequency_bands.theta}%`;
      document.getElementById("thetaBar").style.width = `${data.frequency_bands.theta}%`;

      document.getElementById("alphaVal").textContent = `${data.frequency_bands.alpha}%`;
      document.getElementById("alphaBar").style.width = `${data.frequency_bands.alpha}%`;

      document.getElementById("betaVal").textContent = `${data.frequency_bands.beta}%`;
      document.getElementById("betaBar").style.width = `${data.frequency_bands.beta}%`;
    }

    // Channel Stats Table
    if (data.channel_stats) {
      const tbody = document.getElementById("channelMetricsTable");
      tbody.innerHTML = data.channel_stats.map(ch => `
        <tr>
          <td><strong>${ch.channel}</strong></td>
          <td>${ch.mean}</td>
          <td>${ch.std}</td>
          <td>${ch.peak_to_peak}</td>
        </tr>
      `).join("");
    }

    if (timestampTag) {
      timestampTag.textContent = `Evaluated at ${new Date().toLocaleTimeString()}`;
    }

  } catch (err) {
    console.error("Detection error:", err);
    alert("Connection error occurred while processing sequence.");
  }
}

// Handle manual input sequence
async function handleCustomInput() {
  const field = document.getElementById("sequenceField");
  const raw = field.value.trim();

  if (!raw) {
    alert("Please provide an EEG sequence to evaluate.");
    return;
  }

  let parsed = null;
  try {
    if (raw.startsWith("[") && raw.endsWith("]")) {
      parsed = JSON.parse(raw);
    } else {
      const nums = raw.split(/[\s,]+/).filter(t => t.length > 0).map(Number);
      if (nums.some(isNaN)) {
        alert("Sequence contains non-numeric values. Please provide numbers only.");
        return;
      }
      if (nums.length === 512) {
        parsed = [
          nums.slice(0, 128),
          nums.slice(128, 256),
          nums.slice(256, 384),
          nums.slice(384, 512)
        ];
      } else if (nums.length === 128) {
        parsed = [nums, nums, nums, nums];
      } else {
        alert(`Expected 128 or 512 numeric values, received ${nums.length}.`);
        return;
      }
    }
  } catch (e) {
    alert(`Failed to parse sequence data: ${e.message}`);
    return;
  }

  document.querySelectorAll(".example-item").forEach(item => item.classList.remove("active"));
  const labelEl = document.getElementById("currentSampleName");
  if (labelEl) {
    labelEl.textContent = `Manual Input Sequence (${parsed.length} channels × ${parsed[0]?.length || 0} samples)`;
  }

  renderWaveform(parsed, ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]);
  await runDetection(parsed);
}
