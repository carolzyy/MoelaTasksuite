from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import XFormPrimView,GeometryPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema


class FrankaView(ArticulationView):
    def __init__(
            self,
            prim_paths_expr: str,
            name: Optional[str] = "FrankaView_rod",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        self._ee = XFormPrimView(prim_paths_expr="/World/envs/.*/Robot/panda_rightfinger", name="ee_view", # the position of ee
                                    reset_xform_properties=False)

        self._base = GeometryPrimView(prim_paths_expr="/World/envs/.*/Robot/panda_link0", name="arm_base", # the position of base
                                     reset_xform_properties=False)


        if name =='FrankaView_rod':
            self._def = GeometryPrimView(prim_paths_expr="/World/envs/.*/Robot/rod/Cylinder", name="rod_view",
                                         # the position of middle point
                                         reset_xform_properties=False)



    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        
    ''' 
    def get_deformApis(self,):
        Api_list=[]
        prims = self._def.prims
        for prim in prims:
            Api_list.append(PhysxSchema.PhysxDeformableBodyAPI(prim))

        return Api_list
    '''