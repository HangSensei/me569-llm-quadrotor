"""Saved LLM-generated basis functions from past E1 runs.

Each module in this directory corresponds to a single (condition, run_index)
pair. The ``basis`` function in each module is the raw code produced by
the LLM and accepted by the sandbox, preserved verbatim so that later
diagnostic runs can replay the same basis against different seeds /
thresholds without re-calling the LLM.

Naming convention:
    <condition>_<model>_run_<NN>.py
where condition is ``p`` or ``q`` and NN is a zero-padded counter.
"""
