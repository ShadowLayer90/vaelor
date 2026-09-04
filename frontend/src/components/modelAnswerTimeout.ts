/**
 * How long the browser waits for a request a model answers.
 *
 * The server budget is `remote_inference_budget()` — `VAELOR_INFERENCE_TIMEOUT_SECONDS`,
 * defaulting to 240 seconds and configurable up to `MAX_INFERENCE_SECONDS = 900`
 * (`vaelor/inference_client.py:61-76`). Any client budget below that ceiling
 * abandons a request the appliance is still working on and reports it as a
 * failure, while the run behind it carries on and completes.
 *
 * That is exactly what the appliance check did: it inherited `apiRequest`'s
 * 60-second default for a POST and cut off at roughly 63 seconds. 930 seconds
 * clears the highest budget the server can be configured with, plus headroom,
 * and is the value the AI Chat path already uses for the same reason.
 *
 * Use this for every request that waits on a completion. It is deliberately not
 * a per-surface number: a second opinion about how long a model may think is
 * how one surface ends up truncating what another tolerates.
 */
export const MODEL_ANSWER_TIMEOUT_MS = 930_000;
