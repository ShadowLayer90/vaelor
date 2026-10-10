# Third-party notices

Vaelor is derived in part from open-source software published by SunFounder.
The five SunFounder projects listed below declare the GNU General Public
License, version 2. A copy of that license is retained with the source and at
[LICENSE](LICENSE). The OCI base image is not GPL-2.0-only; its Python and
Debian components retain their respective upstream licenses and notices.

The derivation is confined to the Python control plane (hardware, enclosure,
and telemetry support). The React web interface under `frontend/` is an
original Vaelor work and is not derived from the legacy Pironman dashboard,
whose compiled bundle ships in no Vaelor release artifact (see the legacy
dashboard row below).

## Included in Vaelor's source or release artifacts

| Component | Upstream project | Role in this repository |
| --- | --- | --- |
| `pm_dashboard` | [sunfounder/pm_dashboard](https://github.com/sunfounder/pm_dashboard) | Original dashboard package, API, packaging, and legacy web interface |
| `pironman5` | [sunfounder/pironman5](https://github.com/sunfounder/pironman5) | Pironman enclosure installation and hardware-control services |
| `pm_auto` | [sunfounder/pm_auto](https://github.com/sunfounder/pm_auto) | Pironman automation and hardware-support dependency |
| `sf_rpi_status` | [sunfounder/sf_rpi_status](https://github.com/sunfounder/sf_rpi_status) | Raspberry Pi status and telemetry dependency |
| Legacy dashboard web project | [sunfounder/pm_dashboard_www](https://github.com/sunfounder/pm_dashboard_www) | Provenance for the former interface; its compiled bundle is not included in Vaelor release artifacts |
| Python 3.12 slim Bookworm OCI base | [Docker Official Image: python](https://hub.docker.com/_/python) | Pinned multi-architecture runtime base for the restricted OCI core; includes Python Software Foundation and Debian components with their own notices |
| React and React DOM | [facebook/react](https://github.com/facebook/react) | Compiled into the web interface bundle (`vaelor/www_v2`). MIT License. |
| Hugeicons Free icons | [Hugeicons](https://hugeicons.com) (`@hugeicons/core-free-icons` 4.3.5) | 46 stroke-rounded glyphs copied into `frontend/src/components/hugeicons.ts` and drawn by the web interface. MIT License; see the section below. |
| On-device Assistant model (`Qwen3.5-4B-NPU2`) | Derived from the Qwen3.5-4B model, in FastFlowLM's NPU model format | Published as split assets on the Vaelor GitHub release and fetched by `deploy/fetch-npu-model.sh`. Not part of the wheel or the source. See the section below. |

Python dependencies (Flask, cryptography, paramiko, PyYAML, pypdf, xlrd, and the
others pinned in `requirements-release.txt`) are installed from PyPI at install
time and are not bundled in the wheel; each keeps its own licence.

## Downloaded at install or deploy time, not redistributed

Vaelor's installer and console download these components from their upstream
sources onto the machine they run on. No Vaelor artifact contains them. Each is
pinned by version, checksum, or image digest in the code that fetches it, and
each is governed by its own licence, which applies to whoever runs it.

| Component | Source | When it is fetched | Licence |
| --- | --- | --- | --- |
| FastFlowLM runtime 1.0.2 | [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) release, SHA-256 verified | install, on a machine with a neural accelerator | split: MIT orchestration code, proprietary NPU kernels — see below |
| InfluxDB 1.12.4 | InfluxData's package pool, SHA-256 verified | install | MIT |
| Telegraf 1.32.3 | InfluxData's downloads, SHA-256 verified | staged at install on the controller; copied by the controller to each cluster worker | MIT |
| AMD `amd-smi` and the ROCm runtime | AMD's package repository | install on an AMD GPU or NPU host; `amd-smi` alone on a cluster worker | AMD's licences for those packages |
| Docker Engine | the host's or Docker's repositories | install, only with your approval when Docker is absent | Apache-2.0 |
| `nginx:stable-alpine` | Docker Hub | install (pre-pull) | BSD-2-Clause (nginx) and Alpine's package licences |
| `arizephoenix/phoenix` 20.9.0 | Docker Hub | install (pre-pull) | Elastic License 2.0 (Arize Phoenix) |
| `kyuz0/amd-strix-halo-toolboxes` (llama.cpp on ROCm) | Docker Hub | install (pre-pull) on a gfx1151/gfx1150 GPU | MIT (llama.cpp) and the image's other components' licences |
| `julianmb/q38rocm` (llama.cpp fork for the FP4 model) | GitHub Container Registry | install (pre-pull) on a gfx1151/gfx1150 GPU | MIT (llama.cpp) and the image's other components' licences |
| `rocm/vllm` (vLLM 0.27 on ROCm 10) | Docker Hub | per machine, when a cluster model is served | Apache-2.0 (vLLM) and AMD's licences for ROCm |
| `ryai-vllm` (vLLM 0.22) | `oci-registry.ryai.dev` | per machine, only when a deployment selects it | Apache-2.0 (vLLM) and the image's other components' licences |
| SearXNG | Docker Hub (`searxng/searxng`, by digest) | when web research is enabled | AGPL-3.0 |
| Model weights | Hugging Face or the source you choose | when you download or serve a model | each model's own licence |

Where a model or image's licence restricts use, the restriction is between you
and its publisher; Vaelor neither grants nor narrows it.

## FastFlowLM

The Assistant's NPU tier runs on FastFlowLM's `flm` runtime, which Vaelor
supervises directly. The installer downloads the pinned upstream release
(v1.0.2) from [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM), verifies
its SHA-256, and installs it under `/var/lib/vaelor/flm`. **No Vaelor artifact
contains any FastFlowLM code or binary.**

### Attribution

FastFlowLM asks to be acknowledged in a README, project page, or product. Vaelor
carries the requested line in [README.md](README.md):

```text
Powered by [FastFlowLM](https://github.com/ROCm/FastFlowLM)
```

Keep it there. Removing it while driving the runtime withdraws the
acknowledgement the upstream project asks for.

### The licence is split, and the two upstream sources disagree

Two different sets of terms cover two different parts of FastFlowLM, and they
must not be summarised as one.

- **Orchestration code and CLI tools — MIT.** `LICENSE_RUNTIME.txt` is a
  standard MIT licence, `Copyright (c) 2025 FastFlowLM`. Redistribution requires
  preserving that copyright notice and the full licence text. MIT is compatible
  with Vaelor's GPL-2.0-only distribution. This half is not in dispute.
- **NPU binary kernels — the two sources conflict.**
  - The project README describes them as free for any use including commercial
    use, with no further condition.
  - `TERMS.md`, in the same repository, states the binary components are
    **"NOT open source"**, says they are covered by pending patents, and caps
    free commercial use by revenue: above **USD 10 million** annual revenue an
    explicit commercial licence is required (`info@fastflowlm.com`).

`TERMS.md` is titled *Terms of Use for Proprietary Binaries* and is the more
specific document, so it is the one to rely on. Its section headings are
*Open-Source Code (MIT License)* and *Proprietary Binaries (NPU Kernels)* — the
split is deliberate upstream, not an artefact of how it is being read here.

### Why Vaelor downloads the runtime instead of shipping it

**`TERMS.md` grants no redistribution or bundling right for the proprietary
kernels.** It sets out a usage model for whoever runs them and is silent on
shipping them inside another product. Silence is not permission, and a
proprietary, patent-pending binary shipped alongside GPL-2.0-only work would
raise a combination question this notice cannot settle. So Vaelor ships none of
it: the installer fetches the runtime from upstream's own release onto the
machine that runs it, and the operator who runs it is bound by its terms.
Release rule 5 below applies if that ever changes. This notice records
provenance and is not a substitute for legal advice.

## The on-device Assistant model

The model the Assistant runs on the NPU, `Qwen3.5-4B-NPU2`, is a fine-tune of
the Qwen3.5-4B model converted to FastFlowLM's NPU model format. It is published
as split assets on the Vaelor GitHub release (a release asset is capped at
2 GB) and is verified against a SHA-256 pinned in the Vaelor wheel before it is
unpacked. It is not part of the wheel or the source tree.

**Licence: Apache License, Version 2.0.** It is a modified version of
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Copyright 2026 Alibaba
Cloud, Apache-2.0), built on FastFlowLM's NPU conversion of that model,
[FastFlowLM/Qwen3.5-4B-NPU2](https://huggingface.co/FastFlowLM/Qwen3.5-4B-NPU2),
which is also Apache-2.0. What Vaelor changed:

- `model.q4nx`: the language weights were fine-tuned (full-parameter) on
  examples of Vaelor's tool calls, then converted to FastFlowLM's q4nx format.
- `tokenizer.json` differs from FastFlowLM's published file, and `config.json`
  is FastFlowLM's file with its `flm_version` set to 1.0.2.
- `vision_weight.q4nx`, `tokenizer_config.json` and `chat_template.jinja` are
  FastFlowLM's files, unchanged.

The release that carries the model parts also carries
`qwen35-4b-npu2.LICENSE.txt`: the full licence text, the copyright notice and
this list of changes. Keep that file with the model if you redistribute it.

The model's licence covers the weights only. Running them on the NPU uses
FastFlowLM's runtime and its proprietary NPU kernels, whose separate terms are
described above.

## Hugeicons

The web interface draws 46 icons copied from Hugeicons Free, stroke-rounded
(`@hugeicons/core-free-icons` 4.3.5): only those glyphs' path data, in
`frontend/src/components/hugeicons.ts`, with no Hugeicons package installed.
They are distributed under the MIT License, Copyright (c) 2025 Hugeicons. The
full licence text is at the top of that file, in a comment marked `@license`
so the build keeps it in the compiled bundle. The glyphs are Hugeicons'
work, not Vaelor's.

## Product illustrations

Vaelor does not ship SunFounder product photographs or derivatives. The nine
unlicensed raster files found during release review were removed. The current
overview and enclosure selector use original code-native Vaelor SVG technical
illustrations based on factual product features. Their creation and the hashes
of the removed files are kept in the project's asset-provenance record.

SunFounder product names and factual specifications remain SunFounder
references, as do HP's product names for the machines Vaelor was tested on.
Attribution does not imply SunFounder's or HP's endorsement of Vaelor.

Vaelor adds a new control-plane interface and expanded services for guarded
workload deployment, AI models, assistant and agent workflows, remote access,
auditing, recovery, hardware abstraction, and clustered node management.

When preparing a public release:

1. Preserve this notice, all upstream license files, and copyright notices.
2. Publish the complete corresponding source for the distributed GPL-covered
   build, including local modifications and the scripts used to produce it.
3. Mark modified files or releases clearly and retain upstream Git history
   wherever practical.
4. Produce a source manifest for compiled frontend assets and container images.
5. Review every newly added dependency and asset, record its license and source,
   and remove anything whose redistribution terms are unclear.

This notice records project provenance; it is not a substitute for legal
advice or a release-specific license review.
