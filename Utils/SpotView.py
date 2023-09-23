from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import XFormPrimView,GeometryPrimView


class SpotView(ArticulationView):
    def __init__(
            self,
            prim_paths_expr: str,
            name: Optional[str] = "SpotView_rod",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        self._base = XFormPrimView(prim_paths_expr="/World/envs/.*/Robot/base_link", name="base_view", # the position of ee
                                    reset_xform_properties=False)

        self._ee = GeometryPrimView(prim_paths_expr="/World/envs/.*/Robot/arm0_link_ee", name="arm_ee", # the position of base
                                     reset_xform_properties=False)


        if name =='SpotView_rod':
            self._def = GeometryPrimView(prim_paths_expr="/World/envs/.*/Robot/rod/Cylinder", name="rod_view",
                                         # the position of middle point
                                         reset_xform_properties=False)



    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
