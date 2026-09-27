"""Optivex Track 4 entry built on the official strong-RAG scaffold."""
from __future__ import annotations

import sys
from pathlib import Path

# Local development keeps the participant agent beside the pinned official
# checkout. The container copies ``baselines`` next to this script, so no path
# adjustment is needed there.
_upstream = Path(__file__).resolve().parent.parent / "track4-analysis-public"
if _upstream.is_dir():
    sys.path.insert(0, str(_upstream))

from baselines.strong_rag_baseline import agent as rag_agent
from baselines.strong_rag_baseline import cli as rag_cli
from baselines.strong_rag_baseline.prompts import build_user_prompt as upstream_prompt

_upstream_run_entity = rag_agent.run_entity


SYSTEM_PROMPT = """\
You are the chair of a small evidence council for financial prediction. Work only from the
entity fields and frozen evidence excerpts supplied in the user message. Do not use memory,
outside facts, or events after the cutoff date.

Internally perform four checks before answering:
1. Predictor: infer the requested label, value, or ranking metric from the row and evidence.
2. Evidence reviewer: identify passages that directly support the prediction itself.
3. Calibration reviewer: choose a 90% interval that reflects uncertainty in the target.
4. Admission guard: reject unsupported reasoning, post-cutoff information, and citations that
   do not entail the submitted prediction.

If evidence is weak, remain conservative and widen the interval; never invent support. Return
one JSON object only. Every quote must be copied verbatim from a provided excerpt, and every
claim must state what that quote supports about the submitted prediction."""


def optivex_prompt(task: dict, entity: dict, retrieved: list) -> str:
    base = upstream_prompt(task, entity, retrieved)
    return base + """

OPTIVEX ADMISSION CHECK BEFORE OUTPUT:
- Use no fact absent from ENTITY or EVIDENCE EXCERPTS.
- Prefer evidence about the predicted target over generic company description.
- A citation must support the label/value/ranking, not merely be topically related.
- If excerpts conflict, favor the latest pre-cutoff passage and reflect conflict in uncertainty.
- Do not mention this checklist, the council, or hidden reasoning in the JSON response.
"""


def optivex_run_entity(*args, **kwargs):
    """Use the upstream pipeline but omit optional fields when they are unused.

    The public schema declares ``label`` optional for regression and ranking,
    but it is not nullable. The upstream strong-RAG scaffold currently emits
    ``"label": null`` for those families, which fails g1. Omitting it follows
    the published contract.
    """
    result = _upstream_run_entity(*args, **kwargs)
    if result.prediction.get("label") is None:
        result.prediction.pop("label", None)
    return result


def main() -> int:
    # run_entity imports these names into its own module, so patch that module's
    # prompt hooks before the official CLI constructs any prediction.
    rag_agent.SYSTEM_PROMPT = SYSTEM_PROMPT
    rag_agent.build_user_prompt = optivex_prompt
    rag_agent.run_entity = optivex_run_entity
    rag_cli.run_entity = optivex_run_entity
    return rag_cli.main()


if __name__ == "__main__":
    sys.exit(main())
