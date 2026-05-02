"""Spike test: verify DashScope API + Qwen-Plus can generate a Python function.

Reads DASHSCOPE_API_KEY from environment. DO NOT hard-code the key here.
"""
import os
import sys

try:
    import dashscope
    from dashscope import Generation
except ImportError as e:
    print(f"IMPORT FAIL: {e}")
    sys.exit(1)

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("ERROR: DASHSCOPE_API_KEY not set in environment")
    sys.exit(1)

dashscope.api_key = api_key

prompt = (
    "Write a Python function `basis(x)` that takes a 6-element numpy array "
    "representing a planar quadrotor state [p_x, p_z, theta, v_x, v_z, omega] "
    "and returns the list [1.0, p_x, p_z, np.sin(theta), np.cos(theta), v_x, v_z, omega]. "
    "Use numpy. Return ONLY the function definition, no explanation, no markdown fencing."
)

print("Calling Qwen-Plus via DashScope ...")
import time
t0 = time.time()

response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": prompt}],
    result_format="message",
    temperature=0.3,
    max_tokens=300,
)
elapsed = time.time() - t0

if response.status_code == 200:
    text = response.output.choices[0].message.content
    print(f"DashScope OK (latency: {elapsed:.2f}s, status: {response.status_code})")
    # Backend identifier (what Qwen version is backing qwen-plus)
    if hasattr(response, "request_id"):
        print(f"Request ID: {response.request_id}")
    if hasattr(response, "usage"):
        print(f"Usage: {response.usage}")
    print("\n--- response text ---")
    print(text)
    print("--- end response ---")
    print("\nSPIKE 2 (dashscope) PASSED")
else:
    print(f"DashScope FAIL: status {response.status_code}")
    print(f"Code: {response.code}")
    print(f"Message: {response.message}")
    sys.exit(1)
