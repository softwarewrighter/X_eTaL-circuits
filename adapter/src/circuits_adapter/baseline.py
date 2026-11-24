"""The baseline of an experiment: the metric on the clean prompt and on its twin.

Every intervention is measured against these two numbers.
"""

from circuits_adapter import experiment, metric, model, prompts, run, twins


def compute(m, exp: experiment.Experiment, seed: int) -> dict:
    """Per-position logit differences for the clean prompt and the twin (arrays, float64)."""
    clean, corrupt = prompts.twins(seed, exp.length, m.bos)
    twins.check(clean, corrupt, exp.length)
    out = {}
    for name, ids in (("clean_ld", clean), ("corrupt_ld", corrupt)):
        out[name] = metric.logit_difference(model.logits(m, ids), clean, corrupt, exp.length)
    return {"clean": clean, "corrupt": corrupt, **out}


def write(out, m, exp: experiment.Experiment, seed: int):
    b = compute(m, exp, seed)
    info = model.run_info(m, str(exp.path), seed, exp.length, b["clean"], b["corrupt"])
    axes = ("position",)
    arrays = {"clean_ld": (axes, b["clean_ld"]), "corrupt_ld": (axes, b["corrupt_ld"])}
    return run.write(out, info, arrays)
