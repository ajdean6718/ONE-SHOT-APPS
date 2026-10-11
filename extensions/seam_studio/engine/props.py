"""Engine settings property group (owner: integrator)."""

import bpy
from bpy.props import BoolProperty, FloatProperty


def _live_changed(self, context):
    from . import core
    if self.live_enabled:
        core.request_revalidate()


class SEAM_PG_EngineSettings(bpy.types.PropertyGroup):
    live_enabled: BoolProperty(name="Live Updates", default=True, update=_live_changed,
                               description="Globally enable live 2D->3D and derived-geometry updates")
    debounce_ms: FloatProperty(name="Debounce (ms)", min=0.0, max=1000.0, default=0.0,
                               description="Extra delay before rebuilding after an edit (0 = next tick)")
    async_threshold_ms: FloatProperty(name="Async Threshold (ms)", min=0.0, default=40.0,
                                      description="Builds expected to take longer than this run off the UI thread")


classes = (SEAM_PG_EngineSettings,)
