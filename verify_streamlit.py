import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine
print("Engine loaded successfully!")
# Test sample inference
res = engine.run_inference(engine.REFERENCE_EXAMPLES[0]["data"])
print(f"Sample 1: {res['prediction']} ({res['confidence_percentage']}%)")
res2 = engine.run_inference(engine.REFERENCE_EXAMPLES[2]["data"])
print(f"Sample 2: {res2['prediction']} ({res2['confidence_percentage']}%)")

print("Streamlit code ready for deployment!")
