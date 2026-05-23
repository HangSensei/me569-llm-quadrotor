"""P E2 paraphrase 1. Saved by run_prompt_robustness.py."""
# --- raw LLM response ---
# ```python
# def stage_cost(x, u):
#     return (10.0 * (x[0]**2 + x[1]**2) + 5.0 * x[2]**2 + 1.0 * (x[3]**2 + x[4]**2) + 0.5 * x[5]**2 + 0.01 * ((u[0] - 4.905)**2 + (u[1] - 4.905)**2))
# ```
# --- extracted code ---
def stage_cost(x, u):
    return (10.0 * (x[0]**2 + x[1]**2) + 5.0 * x[2]**2 + 1.0 * (x[3]**2 + x[4]**2) + 0.5 * x[5]**2 + 0.01 * ((u[0] - 4.905)**2 + (u[1] - 4.905)**2))
