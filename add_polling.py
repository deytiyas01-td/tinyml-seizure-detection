with open('app.js', 'r', encoding='utf-8') as f:
    js = f.read()

# Add auto-retry polling to fetchExamples
target = 'if (statusText) statusText.textContent = API_BASE_URL ? "Backend Waking Up..." : "Backend Disconnected";'
replacement = '''if (statusText) statusText.textContent = API_BASE_URL ? "Backend Waking Up (takes ~30s)..." : "Backend Disconnected";
    if (API_BASE_URL && !window._pollTimer) {
      window._pollTimer = setTimeout(() => {
        window._pollTimer = null;
        fetchExamples();
      }, 5000);
    }'''

js = js.replace(target, replacement)

target2 = 'if (statusText) statusText.textContent = "Backend Connected";'
replacement2 = '''if (statusText) statusText.textContent = "Backend Connected";
      if (window._pollTimer) { clearTimeout(window._pollTimer); window._pollTimer = null; }'''

js = js.replace(target2, replacement2)

with open('app.js', 'w', encoding='utf-8') as f:
    f.write(js)

with open('static/app.js', 'w', encoding='utf-8') as f:
    f.write(js)

print("Added auto-reconnect polling to app.js!")
