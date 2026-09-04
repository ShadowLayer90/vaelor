/**
 * The difference between a model's identifier and a model's name.
 *
 * A live test on the Z2 found the AI Chat toolbar offering
 * `/models/Qwen3-1.7B-Q4_K_M.gguf` in the select and repeating it in the
 * sentence beneath — *"This chat uses /models/Qwen3-1.7B-Q4_K_M.gguf."* That
 * string is how the managed llama.cpp server names the file it loaded; it is
 * not the thing the owner chose, and showing it as one is the same shape as
 * the phantom NPU model names: an internal token that outlived the surface it
 * belonged to.
 *
 * **Deliberately conservative.** Only an identifier that is plainly a path or
 * a GGUF file is rewritten. `gpt-4o-mini`, `qwen3-30b` and
 * `meta-llama/Llama-3-8B-Instruct` are names a provider chose and a reader may
 * have to quote back to it, so they are passed through untouched — rewriting
 * them would replace one unreadable string with a *different* unreadable
 * string, which is worse than leaving it alone.
 *
 * The identifier is never thrown away. Whoever is debugging a model server
 * needs the exact path, so it travels beside the name rather than instead of
 * it.
 */

/** Quantisation tokens llama.cpp files carry, as a whole trailing segment. */
const QUANTISATION = /^(?:IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32|FP16|FP32)$/i;

/** `-00001-of-00003`, the shard suffix a split download carries. */
const SHARD = /-\d+-of-\d+$/i;

/** A path, or a file this appliance downloaded. Nothing else is rewritten. */
const LOOKS_LIKE_A_FILE = /^(?:[a-zA-Z]:[\\/]|[\\/]|\.{1,2}[\\/])|\.gguf$/i;

export interface ModelIdentity {
  /** What a reader is shown. Never a path. */
  name: string;
  /** The quantisation, when the identifier carried one. Otherwise `""`. */
  quantization: string;
  /** Exactly what the server called it, kept for whoever is debugging. */
  identifier: string;
  /** True when the name and the identifier are not the same string. */
  renamed: boolean;
}

/**
 * Read a served model identifier as a name, a quantisation and the original.
 *
 * The quantisation is separated rather than dropped: on a machine that can
 * hold several builds of one model it is the difference between them, so it is
 * part of the choice and not noise.
 */
export function modelIdentity(identifier: string): ModelIdentity {
  const raw = String(identifier ?? "").trim();
  const blank = { name: raw, quantization: "", identifier: raw, renamed: false };
  if (!raw || !LOOKS_LIKE_A_FILE.test(raw)) return blank;
  const base = raw.split(/[\\/]/).filter(Boolean).at(-1) ?? "";
  // The shard marker sits *after* the quantisation, so it comes off first or
  // the quantisation is never the last token to be examined.
  const stem = base.replace(/\.gguf$/i, "").replace(SHARD, "");
  if (!stem) return blank;
  const parts = stem.split("-").filter(Boolean);
  let quantization = "";
  if (parts.length > 1 && QUANTISATION.test(parts[parts.length - 1])) {
    quantization = parts.pop() as string;
  }
  const name = parts.join(" ") || stem;
  return { name, quantization, identifier: raw, renamed: name !== raw };
}

/**
 * One line naming a model, with the quantisation only where there is one.
 *
 * `Qwen3 1.7B · Q4_K_M` rather than `/models/Qwen3-1.7B-Q4_K_M.gguf`, and
 * `gpt-4o-mini` unchanged.
 */
export function modelDisplayName(identifier: string): string {
  const model = modelIdentity(identifier);
  if (!model.name) return model.identifier;
  return model.quantization ? `${model.name} · ${model.quantization}` : model.name;
}
