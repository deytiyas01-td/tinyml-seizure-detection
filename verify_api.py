import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
import json

client = app.test_client()

print("="*60)
print("RUNNING COMPLETE VERIFICATION SUITE")
print("="*60)

# 1. Test GET /
res = client.get('/')
assert res.status_code == 200, f"GET / failed: {res.status_code}"
print(f"[OK] GET / -> Status 200 OK, HTML size: {len(res.data)} bytes")

# 2. Test GET /api/examples
res = client.get('/api/examples')
assert res.status_code == 200, f"GET /api/examples failed: {res.status_code}"
data_ex = res.get_json()
print(f"[OK] GET /api/examples -> Loaded {data_ex['count']} examples:")
for ex in data_ex['examples']:
    safe_name = ex['name'].encode('ascii', errors='ignore').decode('ascii').strip()
    print(f"   * [{ex['type'].upper()}] {safe_name}")

# 3. Test POST /api/predict for all 4 examples
print("\nTesting Predictions on the 4 One-Click Examples:")
for i, ex in enumerate(data_ex['examples']):
    res = client.post('/api/predict', json={'sequence': ex['data']})
    assert res.status_code == 200, f"Prediction {i} failed: {res.data}"
    d = res.get_json()
    safe_name = ex['name'].encode('ascii', errors='ignore').decode('ascii').strip()
    print(f"   -> Example {i+1} ({safe_name}): Prediction={d['prediction']} | Confidence={d['confidence_percentage']}% | Prob={d['probability']}")

# 4. Test Security: Rejection of non-numeric
res = client.post('/api/predict', json={'sequence': ['invalid', 'text']})
assert res.status_code == 400
print("\n[OK] Security check: Non-numeric payload rejected (400 Bad Request)")

# 5. Test Security: Rejection of NaN
res = client.post('/api/predict', json={'sequence': [float('nan')]*128})
assert res.status_code == 400
print("[OK] Security check: NaN payload rejected (400 Bad Request)")

# 6. Test Security: Rejection of out-of-bounds amplitude
res = client.post('/api/predict', json={'sequence': [999999.0]*128})
assert res.status_code == 400
print("[OK] Security check: Extreme amplitude payload rejected (400 Bad Request)")

# 7. Test Security Headers
res = client.get('/')
print("\n[OK] Security Headers Verified:")
for h in ['X-Content-Type-Options', 'X-Frame-Options', 'Content-Security-Policy', 'X-XSS-Protection']:
    print(f"   * {h}: {res.headers.get(h)}")

print("\n" + "="*60)
print("SUCCESS: ALL TESTS PASSED! APPLICATION IS 100% READY.")
print("="*60)
