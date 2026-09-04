import { useState } from "react";
import { formatQuantity } from "../lib/format";
import type { CopilotSetupData, LocalModelChoice } from "./CopilotSetup";
import { Icon } from "./Icon";
import { Button, Input, Notice } from "./ui";

/**
 * The models this appliance can install — read from the appliance.
 *
 * This screen used to hold its own `MODEL_OPTIONS`: four models with
 * hand-written sizes, memory advice and licences, and the recommendation
 * matched against them by name. That is the same defect as the phantom 32B —
 * a table of *names* with no artifact behind them — one layer up. Two of the
 * four were not in the served catalog at all, so a reader could press "Review
 * installation" on a model this appliance had no repository or file for; and
 * the sizes were prose, which is how *"Install Qwen3 32B · about 1.1 GB"* was
 * written beside a 20 GB model.
 *
 * Everything shown here now comes from `recommendation.catalog`, whose entries
 * are the ones `vaelor/model_catalog.py` can resolve to a repository and a
 * file. A machine that can hold more than the catalog offers is told that
 * about its hardware — `catalog_note` — and pointed at the search box, which
 * can resolve a real file. Naming a model the appliance does not stock is no
 * longer something this screen is able to do, because it has no names of its
 * own to say.
 */

/**
 * The card borders, which are decoration and carry no meaning.
 *
 * They used to be a property of a hand-written entry — `accent: "cyan"` beside
 * `license: "Apache 2.0"` — so a colour looked like a fact about the model.
 * They are assigned by position instead, the way the telemetry tiles are: the
 * caption on Home already had to say that colours group readings rather than
 * indicate health, and the same is true here.
 */
const CARD_ACCENTS = ["cyan", "orange", "violet", "green"] as const;

function ModelCard({
  accent,
  busy,
  disabled,
  gpu,
  model,
  onChoose,
  recommended,
}: {
  accent: string;
  busy: boolean;
  disabled: boolean;
  gpu: boolean;
  model: LocalModelChoice;
  onChoose: (query: string) => void;
  recommended: boolean;
}) {
  return (
    <article className={`model-choice model-choice--${accent}`}>
      {/*
        * The badge row. Both the recommended flag and the GPU marker (added for
        * the ROCmFP4 27B) live here, in one in-flow flex row at the top of the
        * card, so they wrap and space beside each other instead of overlapping.
        * Each is still driven by its own condition — recommended by the pick,
        * GPU by the recommendation state so it shows only where the override
        * fired — and the row is absent when the card carries neither.
        */}
      {(recommended || gpu) && (
        <div className="model-choice__badges">
          {recommended && (
            <span className="model-choice__recommended">Recommended for this hardware</span>
          )}
          {gpu && (
            <span className="model-choice__gpu">Runs on the graphics processor</span>
          )}
        </div>
      )}
      <div className="model-choice__heading">
        <span><Icon name="cpu" /></span>
        <small>{model.parameter_size ? `${model.parameter_size} parameters` : "Local model"}</small>
      </div>
      <h3>{model.name}</h3>
      <p>{model.experience}</p>
      <dl>
        {/* The size sentence the appliance derived from the byte count its own
            fit check divides by, rather than a figure written beside a name. */}
        <div><dt>Download</dt><dd>{model.size_note}</dd></div>
        {model.quantization && (
          <div><dt>Quantisation</dt><dd>{model.quantization}</dd></div>
        )}
        {/* The artifact. An entry without these two is not in the catalog, and
            showing them is what makes that checkable from the screen. */}
        {model.repo && <div><dt>Repository</dt><dd>{model.repo}</dd></div>}
        {model.file && <div><dt>File</dt><dd>{model.file}</dd></div>}
      </dl>
      <Button
        disabled={busy || disabled}
        onClick={() => onChoose(model.search_query)}
        variant={recommended ? "primary" : undefined}
      >
        Review installation
      </Button>
    </article>
  );
}

export function ModelCatalog({
  busy,
  disabled,
  setup,
  onChoose,
  onClose,
}: {
  busy: boolean;
  disabled: boolean;
  setup?: CopilotSetupData | null;
  onChoose: (query: string) => void;
  onClose: () => void;
}) {
  const [search, setSearch] = useState("");
  // The CATALOG's recommendation, not the Assistant's. These are two separate
  // flows on this appliance: the Assistant is set up on its own screen, while
  // this catalog installs GPU/CPU GGUFs for local/AI-Chat use.
  // `setup.recommendation` is the Assistant's pick; reading it here was the bug
  // that showed the Assistant's model as the catalog pick. `catalog_recommendation`
  // is the installable pick — the GPU 27B on a gfx1151 box, the base GGUF
  // elsewhere — so the catalog recommends independently of the Assistant.
  const recommendation = setup?.catalog_recommendation;
  const recommendedId = recommendation?.primary?.id ?? "";
  // Whether the Assistant is actually served on a neural processor on THIS
  // machine — the NPU override sets `served_on_npu` on the Assistant's
  // `recommendation`, and it is absent on a Pi (where the Assistant's model is
  // a downloadable GGUF of the same class this catalog lists). The scope line
  // below only claims a neural processor when this is true; asserting one
  // universally is the VD-108 class of hardware-specific copy stated as fact.
  const assistantOnNpu = setup?.recommendation?.served_on_npu === true;
  // The recommended pick is served on the GPU when the override marked it so.
  // Tied to the recommendation, not to any catalog entry's engine, so the GPU
  // marker only appears on the machine where the pick can actually run there.
  const gpuServed =
    recommendation?.served_on_gpu === true ||
    recommendation?.primary?.backend === "rocmfpx";
  const served = recommendation?.catalog ?? [];
  // Recommended first, catalog order after it. The order is the appliance's;
  // this only lifts the one entry it chose.
  const models = [
    ...served.filter((model) => model.id === recommendedId),
    ...served.filter((model) => model.id !== recommendedId),
  ];
  const memory = setup
    ? `${formatQuantity(setup.hardware.memory_total_bytes, "capacity")} RAM`
    : "this device";
  return (
    <section className="model-catalog" aria-labelledby="model-catalog-title">
      <div className="model-catalog__header">
        <div>
          <span className="page-eyebrow">Local AI catalog</span>
          <h2 id="model-catalog-title">Choose what matters most</h2>
          <p>
            {recommendation?.primary
              ? `Vaelor recommends ${recommendation.primary.name} for ${memory}.`
              : "Vaelor has not read this machine's recommendation yet."}
            {" "}Every exact file must pass RAM, storage, and format checks before download.
          </p>
          {/*
            * Clarify NPU vs GPU for a beginner (test finding #2). The
            * hardware-independent part is true on every machine: these are
            * installable local models for AI Chat, a GPU-served card is marked,
            * and the Assistant is a separate flow. The neural-processor clause
            * is added ONLY where the Assistant is actually NPU-served
            * (`assistantOnNpu`); on a Pi that claim would be false, so it is
            * omitted rather than stated universally (the VD-108 class).
            */}
          <p className="model-catalog__scope">
            These are local models you install for AI Chat and other on-device use — a card that
            runs on the graphics processor is marked. The Assistant is set up separately
            {assistantOnNpu
              ? " and runs its own model on this appliance's neural processor"
              : ""}, not from this list.
          </p>
          {/*
            * What is offered, and what the hardware could hold — two figures,
            * kept apart. Conflating them is the whole defect: a Z2 whose
            * budget reaches the 32B class was told "Vaelor recommends 32B"
            * against a catalog whose largest entry is 4B. Where the machine
            * *does* outrun the catalog this line stays off, because
            * `catalog_note` above already states the same fact and adds what
            * to do about it, and one screen does not need to say it twice.
            */}
          {recommendation?.hardware_tier && !recommendation.exceeds_catalog && (
            <p className="model-catalog__tiers">
              Recommended size {recommendation.parameter_range} · this machine's memory budget
              reaches the {recommendation.hardware_tier} class.
            </p>
          )}
        </div>
        <Button onClick={onClose} variant="quiet">Close</Button>
      </div>
      {/*
        * A machine that outruns the catalog is told so about its *hardware*.
        * The sentence names no model, because naming one the catalog does not
        * stock is the whole defect this screen was carrying.
        */}
      {recommendation?.exceeds_catalog && recommendation.catalog_note && (
        <Notice severity="info">{recommendation.catalog_note}</Notice>
      )}
      {models.length ? (
        <div className="model-catalog__grid">
          {models.map((model, index) => (
            <ModelCard
              accent={CARD_ACCENTS[index % CARD_ACCENTS.length]}
              busy={busy}
              disabled={disabled}
              gpu={gpuServed && model.id === recommendedId}
              key={model.id}
              model={model}
              onChoose={onChoose}
              recommended={model.id === recommendedId}
            />
          ))}
        </div>
      ) : (
        /*
         * No fallback list. A hand-written stand-in is exactly what was here
         * before, and offering a model this appliance has not confirmed it can
         * fetch is worse than saying nothing and leaving the search box, which
         * resolves a real repository and file every time.
         */
        <Notice severity="info">
          Vaelor has not read this appliance's reviewed model list yet. Search Hugging Face
          below to check a specific model, or reopen this screen once the appliance has answered.
        </Notice>
      )}
      <form className="model-search" onSubmit={(event) => { event.preventDefault(); if (search.trim()) onChoose(search); }}>
        <div>
          <label htmlFor="model-search">Looking for another model?</label>
          <span>Enter a Hugging Face model name. Vaelor will check format, memory, and storage first.</span>
        </div>
        <Input
          aria-label="Looking for another model?"
          id="model-search"
          label={<span className="sr-only">Looking for another model?</span>}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Example: organization/model-name"
          value={search}
        />
        <Button disabled={busy || disabled || !search.trim()} type="submit">Check this model</Button>
      </form>
    </section>
  );
}
