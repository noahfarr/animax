from __future__ import annotations

import dataclasses

from animax.trace_patterning import TracePatterning, TracePatterningParams


class NoisyPatterning(TracePatterning):
    @property
    def default_params(self) -> TracePatterningParams:
        base = TracePatterningParams()
        return dataclasses.replace(
            base, isi_min=base.cs_activation_length, isi_max=base.cs_activation_length
        )

    @property
    def name(self) -> str:
        return "NoisyPatterning-v0"
