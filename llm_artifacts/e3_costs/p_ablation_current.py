"""P-condition prompt ablation variant 'current'. Saved by scripts/e3_prompt_ablation.py.
"""
# Raw LLM response follows.
# ```python
# def stage_cost(x, u):
#     pos = 20.0 * x[0] ** 2 + 20.0 * x[1] ** 2
#     att = 3.0 * x[2] ** 2
#     vel = 2.0 * x[3] ** 2 + 2.0 * x[4] ** 2 + 1.0 * x[5] ** 2
#     ctrl = 0.05 * (u[0] - 4.905) ** 2 + 0.05 * (u[1] - 4.905) ** 2
#     return pos + att + vel + ctrl
# ```
# Extracted stage_cost implementation follows.
def stage_cost(x, u):
    pos = 20.0 * x[0] ** 2 + 20.0 * x[1] ** 2
    att = 3.0 * x[2] ** 2
    vel = 2.0 * x[3] ** 2 + 2.0 * x[4] ** 2 + 1.0 * x[5] ** 2
    ctrl = 0.05 * (u[0] - 4.905) ** 2 + 0.05 * (u[1] - 4.905) ** 2
    return pos + att + vel + ctrl
