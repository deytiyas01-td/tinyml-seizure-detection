import sys
sys.path.append(r'C:\Users\TIYAS DEY\.gemini\antigravity\scratch\tinyml-seizure-detection-chbmit')
import engine
import json

print("Exporting reference examples to JSON...")
with open(r'C:\Users\TIYAS DEY\.gemini\antigravity\scratch\tinyml-seizure-detection-chbmit\examples.json', 'w') as f:
    json.dump(engine.REFERENCE_EXAMPLES, f, indent=2)

print("Saved examples.json")
