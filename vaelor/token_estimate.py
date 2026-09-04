"""One calibrated characters-per-token figure, with the measurement behind it.

There is no tokeniser in this process. Every budget that speaks in tokens is
therefore converting characters, and the conversion factor was written out by
hand in three separate modules - ``assistant_machine_brief``,
``provider_runtime``, and ``model_calibration`` - each carrying its own copy of
the number ``3`` and its own comment explaining that three was "conservative".

It was not conservative. It was wrong, in a direction nothing tested for:

- Counting *tokens from characters*, three over-counted by 38%. That is how a
  717-token brief was recorded in VD-008 as 920, and how the live brief drifted
  to 991 with every test still green.
- Counting *characters from tokens* - which is what a context budget does -
  three under-filled the window by the same proportion. Every managed-local
  request was handed roughly 28% less context than the model could hold. It
  never overflowed, so nothing complained; the appliance simply paid for a
  window and used three quarters of it. Under VD-001 the NPU window is
  deliberately 8192, and FLM caches no prefix, so that shortfall was re-paid on
  every single request.

The measurement is one brief through the tokeniser that actually bills for it:
2,972 characters, 717 tokens - **4.145 characters per token**. Four is used
rather than 4.145 because four is safe in both directions at once: estimating
tokens it rounds up (743 against 717, +3.6%), and sizing a context in
characters it rounds down (2,868 against 2,972, -3.5%). A single integer that
errs towards "smaller budget" whichever way it is applied.

Anything that has an exact count must report the exact count. This module
exists for the cases that genuinely have none.
"""

from __future__ import annotations

import math

#: The sample this factor is calibrated against: the Z2 machine brief, measured
#: on the tokeniser that charges for it. Kept here, beside the constant, so a
#: future change to ``CHARS_PER_TOKEN`` has to argue with a measurement rather
#: than with an adjective.
MEASURED_SAMPLE_CHARS = 2972
MEASURED_SAMPLE_TOKENS = 717

#: Characters per token. See the module docstring for why this is 4 and not 3.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text``. An estimate, never a count."""
    return math.ceil(len(str(text)) / CHARS_PER_TOKEN)


def context_chars(tokens: int) -> int:
    """How many characters fit in ``tokens`` worth of window.

    The inverse of :func:`estimate_tokens`, and the direction the context
    budgets use. Negative room is zero room, not a negative budget.
    """
    return max(0, int(tokens)) * CHARS_PER_TOKEN
