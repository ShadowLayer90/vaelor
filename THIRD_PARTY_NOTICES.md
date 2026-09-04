# Third-party notices

Vaelor is derived in part from open-source software published by SunFounder.
The five SunFounder projects listed below declare the GNU General Public
License, version 2. A copy of that license is retained with the source and at
[LICENSE](LICENSE). The OCI base image is not GPL-2.0-only; its Python and
Debian components retain their respective upstream licenses and notices.

| Component | Upstream project | Role in this repository |
| --- | --- | --- |
| `pm_dashboard` | [sunfounder/pm_dashboard](https://github.com/sunfounder/pm_dashboard) | Original dashboard package, API, packaging, and legacy web interface |
| `pironman5` | [sunfounder/pironman5](https://github.com/sunfounder/pironman5) | Pironman enclosure installation and hardware-control services |
| `pm_auto` | [sunfounder/pm_auto](https://github.com/sunfounder/pm_auto) | Pironman automation and hardware-support dependency |
| `sf_rpi_status` | [sunfounder/sf_rpi_status](https://github.com/sunfounder/sf_rpi_status) | Raspberry Pi status and telemetry dependency |
| Legacy dashboard web project | [sunfounder/pm_dashboard_www](https://github.com/sunfounder/pm_dashboard_www) | Provenance for the former interface; its compiled bundle is not included in Vaelor release artifacts |
| Python 3.12 slim Bookworm OCI base | [Docker Official Image: python](https://hub.docker.com/_/python) | Pinned multi-architecture runtime base for the restricted OCI core; includes Python Software Foundation and Debian components with their own notices |
| FastFlowLM (`flm-real`) | [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) | NPU inference runtime Vaelor is to supervise directly for the Assistant NPU tier (VD-001, VD-002). **Split licensing — see the section below. Not currently redistributed in any Vaelor artifact.** |

## FastFlowLM

Vaelor has decided to drive FastFlowLM's `flm-real` directly for the Assistant's
NPU tier and to remove Lemonade from the machine (VD-001, VD-002). Both rows are
`Built: no` at the time of writing, and **no Vaelor artifact currently contains
any FastFlowLM code or binary.** This section is recorded ahead of that work,
because an attribution or licence term satisfied *after* a release ships is one
that was breached in the interim.

### Attribution

FastFlowLM asks to be acknowledged in a README, project page, or product. Vaelor
carries the requested line verbatim in [README.md](README.md):

```text
Powered by [FastFlowLM](https://github.com/ROCm/FastFlowLM)
```

Keep it there. Removing it while shipping or driving the runtime withdraws the
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
    use, with no further condition. This is the wording quoted in VD-002a.
  - `TERMS.md`, in the same repository, states the binary components are
    **"NOT open source"**, says they are covered by pending patents, and caps
    free commercial use by revenue: above **USD 10 million** annual revenue an
    explicit commercial licence is required (`info@fastflowlm.com`).

`TERMS.md` is titled *Terms of Use for Proprietary Binaries* and is the more
specific document, so it is the one to rely on. Its section headings are
*Open-Source Code (MIT License)* and *Proprietary Binaries (NPU Kernels)* — the
split is deliberate upstream, not an artefact of how it is being read here.

### The redistribution grant is not established

**`TERMS.md` grants no redistribution or bundling right for the proprietary
kernels.** It sets out a usage model for whoever runs them and is silent on
shipping them inside another product. Silence is not permission.

**This conflicts with a later maintainer review**, which recorded the licence
question as resolved and concluded that redistribution is permitted and that the
earlier "do not bundle, terms unclear" position is lifted. That conclusion rests
on the README wording above; `TERMS.md` was evidently not read alongside it. The
ledger is authoritative for decisions and this notice does not overrule it — but
a licence position is a fact about someone else's document, not a decision
Vaelor gets to make, so the discrepancy is recorded here rather than resolved.
**Reconcile VD-002a against `TERMS.md` before any artifact carries the binary.**

Two questions still need answering in writing:

1. **Is redistribution granted at all?** If not, Vaelor may supervise a runtime
   the operator installed — `sudo apt install ./fastflowlm*.deb` from upstream's
   own Linux packages, per VD-002a — but must not package one. Release rule 5
   below already covers this: remove anything whose redistribution terms are
   unclear.
2. **Is a proprietary, patent-pending binary compatible with distributing
   Vaelor under GPL-2.0-only?** Shipping a non-free component alongside GPL v2
   work raises a combination question this notice cannot settle.

Until both are answered, the conservative position costs Vaelor almost nothing:
install `flm-real` from upstream's published `.deb` on the appliance and ship
none of it. VD-002a itself notes that upstream packaging makes bundling a
*choice* rather than a requirement. This notice records provenance and is not a
substitute for legal advice.

## Product illustrations

Vaelor does not ship SunFounder product photographs or derivatives. The nine
unlicensed raster files found during release review were removed. The current
overview and enclosure selector use original code-native Vaelor SVG technical
illustrations based on factual product features.

SunFounder product names and factual specifications remain SunFounder
references. Attribution does not imply SunFounder endorsement of Vaelor.

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
