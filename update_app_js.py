with open('app.js', 'r', encoding='utf-8') as f:
    js = f.read()

old_func_start = 'async function fetchExamples() {'
idx = js.find(old_func_start)
idx_end = js.find('function renderExamplesList', idx)

new_func = '''async function fetchExamples() {
  const container = document.getElementById("examplesContainer");
  const statusIndicator = document.getElementById("serverStatusIndicator");
  const statusText = document.getElementById("serverStatusText");

  // Render embedded examples immediately so UI is always responsive
  if (referenceExamples && referenceExamples.length > 0) {
    renderExamplesList(referenceExamples);
    selectExample(activeExampleIndex || 0);
  }

  const endpoint = getApiUrl("/api/examples");
  if (!API_BASE_URL && window.location.hostname.includes("vercel.app")) {
    if (statusIndicator) statusIndicator.style.backgroundColor = "#eab308";
    if (statusText) statusText.textContent = "Paste Render URL Above";
    return;
  }

  try {
    const res = await fetch(endpoint, { signal: AbortSignal.timeout(8000) });
    const json = await res.json();
    if (json.status === "success" && json.examples) {
      referenceExamples = json.examples;
      renderExamplesList(referenceExamples);
      if (statusIndicator) statusIndicator.style.backgroundColor = "#22c55e";
      if (statusText) statusText.textContent = "Backend Connected";
    }
  } catch (err) {
    console.warn("Backend not reachable yet:", err);
    if (statusIndicator) statusIndicator.style.backgroundColor = "#ef4444";
    if (statusText) statusText.textContent = API_BASE_URL ? "Backend Waking Up..." : "Backend Disconnected";
  }
}

'''

updated_js = js[:idx] + new_func + js[idx_end:]

updated_js = updated_js.replace(
    'alert("Connection error occurred while processing sequence.");',
    'if (!API_BASE_URL && window.location.hostname.includes("vercel.app")) { alert("Please paste your Render Backend URL in the top bar to run live seizure detection!"); } else { alert("Could not reach backend. If using Render free tier, it may be waking up from sleep (takes ~30s). Please try again in a moment."); }'
)

with open('app.js', 'w', encoding='utf-8') as f:
    f.write(updated_js)

with open('static/app.js', 'w', encoding='utf-8') as f:
    f.write(updated_js)

print("Updated app.js and static/app.js successfully!")
