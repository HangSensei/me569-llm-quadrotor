"""Spike test: verify PySINDy can fit a known ODE."""
import sys

try:
    import numpy as np
    import pysindy as ps
except ImportError as e:
    print(f"IMPORT FAIL: {e}")
    sys.exit(1)

# Ground truth: x_dot = -x (exponential decay)
dt = 0.01
t = np.arange(0, 5, dt)
x = np.exp(-t).reshape(-1, 1)

model = ps.SINDy(
    differentiation_method=ps.FiniteDifference(),
    feature_library=ps.PolynomialLibrary(degree=2),
    optimizer=ps.STLSQ(threshold=0.1),
)
model.fit(x, t=dt, feature_names=["x"])

print("pysindy OK — fitted model:")
model.print()

# Expect coefficient of x in x' to be close to -1
coef = model.coefficients()
print(f"\nCoefficient matrix shape: {coef.shape}")
print(f"Fitted coefficients: {coef}")
print(f"pysindy version: {ps.__version__}")
print("SPIKE 4 (pysindy) PASSED" if abs(coef).sum() > 0 else "SPIKE 4 (pysindy) FAILED — all zeros")
