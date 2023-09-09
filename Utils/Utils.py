import carb
import typing
from pxr import Usd, UsdGeom, Sdf, Gf, UsdShade
from pxr import UsdPhysics, PhysicsSchemaTools, PhysxSchema
from omni.physx.bindings._physx import SimulationEvent
import omni.usd
import math



def create_mesh(stage, path, points, normals, indices, vertexCounts):
    mesh = UsdGeom.Mesh.Define(stage, path)

    # Fill in VtArrays
    mesh.CreateFaceVertexCountsAttr().Set(vertexCounts)
    mesh.CreateFaceVertexIndicesAttr().Set(indices)
    mesh.CreatePointsAttr().Set(points)
    mesh.CreateDoubleSidedAttr().Set(False)
    mesh.CreateNormalsAttr().Set(normals)

    return mesh

def create_rod(stage, path, halfSize,length):
    points = [
        Gf.Vec3f(length, -halfSize, 0),
        Gf.Vec3f(length, halfSize, 0),
        Gf.Vec3f(length, halfSize, 2 * halfSize),
        Gf.Vec3f(length, -halfSize, halfSize),
        Gf.Vec3f(0, -halfSize, 0),
        Gf.Vec3f(0, halfSize, 0),
        Gf.Vec3f(0, halfSize, 2 *halfSize),
        Gf.Vec3f(0, -halfSize, 2 *halfSize),
    ]
    normals = [
        Gf.Vec3f(1, 0, 0),
        Gf.Vec3f(1, 0, 0),
        Gf.Vec3f(1, 0, 0),
        Gf.Vec3f(1, 0, 0),
        Gf.Vec3f(-1, 0, 0),
        Gf.Vec3f(-1, 0, 0),
        Gf.Vec3f(-1, 0, 0),
        Gf.Vec3f(-1, 0, 0),
    ]
    indices = [0, 1, 2, 3, 1, 5, 6, 2, 3, 2, 6, 7, 0, 3, 7, 4, 1, 0, 4, 5, 5, 4, 7, 6]
    vertexCounts = [4, 4, 4, 4, 4, 4]

    # Create the mesh
    mesh = create_mesh(stage, path, points, normals, indices, vertexCounts)
    add_physx_deformable_body(stage,
                              path,
                              simulation_hexahedral_resolution=5,
                              self_collision=False, )
    return mesh

def add_deformable_body_material(
    stage,
    path,
    damping_scale=None,
    density=None,
    dynamic_friction=None,
    elasticity_damping=None,
    poissons_ratio=None,
    youngs_modulus=None,
):
    """Applies the PhysxSchema.PhysxDeformableSurfaceMaterialAPI to the prim at path on stage.

    Args:
        stage:                          The stage
        path:                           Path to UsdShade.Material to which the material API should be applied to.
        ... schema attributes:          See USD schema for documentation

    Returns:
        True if the API apply succeeded.
    """
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        if not prim.IsA(UsdShade.Material):
            carb.log_warn(f"AddMaterial: Prim at path {path} is already defined and not a Material")
            return False
        return True

    UsdShade.Material.Define(stage, path)
    material = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(path))

    if damping_scale is not None:
        material.CreateDampingScaleAttr().Set(damping_scale)
    if density is not None:
        material.CreateDensityAttr().Set(density)
    if dynamic_friction is not None:
        material.CreateDynamicFrictionAttr().Set(dynamic_friction)
    if elasticity_damping is not None:
        material.CreateElasticityDampingAttr().Set(elasticity_damping)
    if poissons_ratio is not None:
        material.CreatePoissonsRatioAttr().Set(poissons_ratio)
    if youngs_modulus is not None:
        material.CreateYoungsModulusAttr().Set(youngs_modulus)

    return True

def add_physx_deformable_body(  # TODO PREIST: Get defaults from schema metadata instead of hardcoding here
    stage,
    prim_path: Sdf.Path,
    collision_rest_points: typing.List[Gf.Vec3f] = None,
    collision_indices: typing.List[int] = None,
    collision_simplification : bool = True,
    collision_simplification_remeshing: bool = True,
    collision_simplification_remeshing_resolution: int = 0,
    collision_simplification_target_triangle_count: int = 0,
    collision_simplification_force_conforming: bool = False,
    simulation_rest_points: typing.List[Gf.Vec3f] = None,
    simulation_indices: typing.List[int] = None,
    embedding: typing.List[int] = None,
    simulation_hexahedral_resolution: int = 10,
    solver_position_iteration_count: int = None,
    vertex_velocity_damping: float = None,
    sleep_damping: float = None,
    sleep_threshold: float = None,
    settling_threshold: float = None,
    self_collision: bool = None,
    self_collision_filter_distance: float = None,
) -> bool:

    """Applies the soft body API to an Xform at prim_path on stage.

    Args:
        stage:                                          The stage
        prim_path:                                      Path to UsdGeom.Mesh 'skin mesh' to which the PhysxSchema.PhysXDeformableBodyAPI is applied to.
        collision_rest_points:                          List of vertices of the collision tetrahedral mesh at rest.
                                                        If a collision mesh is provided, the simulation mesh needs to be provided too.
                                                        If no collision mesh is provided, it will be computed implicitly based on the simplification parameter.
        collision_indices:                              List of indices of the collision tetrahedral mesh.
        collision_simplification:                       Boolean flag indicating if simplification should be applied to the mesh before creating a
                                                        softbody out of it. Is ignored if simulation mesh has been provided.
        collision_simplification_remeshing:             Boolean flag indicating if the simplification should be based on remeshing.
                                                        Ignored if collision_simplification equals False.
        collision_simplification_remeshing_resolution:  The resolution used for remeshing. A value of 0 indicates that a heuristic is used to determine
                                                        the resolution. Ignored if collision_simplification_remeshing is False.
        collision_simplification_target_triangle_count: The target triangle count used for the simplification. A value of 0 indicates
                                                        that a heuristic based on the simulation_hexahedral_resolution is to determine the target count.
                                                        Ignored if collision_simplification equals False.
        collision_simplification_force_conforming:      Boolean flag indicating that the tretrahedralizer used to generate the collision mesh should produce
                                                        tetrahedra that conform to the triangle mesh. If False the implementation chooses the tretrahedralizer
                                                        used.
        simulation_rest_points:                         List of vertices of the simulation tetrahedral mesh at rest.
                                                        If a simulation mesh is provided, the collision mesh needs to be provided too.
                                                        If no simulation mesh is provided it will be computed implicitly based on simulation_hexahedral_resolution.
        simulation_indices:                             List of indices of the simulation tetrahedral mesh.
        embedding:                                      Optional embedding information mapping collision points to containing simulation tetrahedra.
        simulation_hexahedral_resolution:               Target resolution of voxel simulation mesh. Is ignored if simulation mesh has been provided.
        ...:                                            See USD schema for documentation

    Returns:
        True / False that indicates success of schema application
    """

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        carb.log_warn("No valid primitive prim_path provided")
        return False

    # check if it is a rigid body:
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        carb.log_warn(
            "PhysxSchema.PhysxDeformableBodyAPI cannot be applied to a primitive with UsdPhysics.RigidBodyAPI"
        )
        return False

    # check if it is a UsdGeom.Mesh
    if not prim.IsA(UsdGeom.Mesh):
        carb.log_warn("PhysxSchema.PhysxDeformableBodyAPI can only be applied to a UsdGeom.Mesh")

    # check collision mesh
    if collision_rest_points:
        if len(collision_rest_points) < 4:
            carb.log_warn("collision_rest_points is invalid")
            return False
        if not collision_indices:
            carb.log_warn("collision mesh invalid")
            return False
        if not simulation_rest_points:
            carb.log_warn("collision mesh is invalid without simulation mesh")
            return False

    if collision_indices:
        if len(collision_indices) < 4 or len(collision_indices) % 4 != 0:
            carb.log_warn("collision_indices is invalid")
            return False
        if not collision_rest_points:
            carb.log_warn("collision mesh invalid")
            return False

    # check simulation mesh
    if simulation_rest_points:
        if len(simulation_rest_points) < 4:
            carb.log_warn("simulation_rest_points is invalid")
            return False
        if not simulation_indices:
            carb.log_warn("simulation mesh invalid")
            return False
        if not collision_rest_points:
            carb.log_warn("simulation mesh is invalid without collision mesh")
            return False

    if simulation_indices:
        if len(simulation_indices) < 4 or len(simulation_indices) % 4 != 0:
            carb.log_warn("simulation_indices is invalid")
            return False
        if not simulation_rest_points:
            carb.log_warn("simulation mesh invalid")
            return False

    if embedding:
        if len(embedding) != len(collision_rest_points):
            carb.log_warn("embedding is invalid")
            return False
        if not simulation_rest_points:
            carb.log_warn("embedding is invalid without simulation mesh")
            return False

    # apply APIs and create attributes
    deformable_body_api = PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)

    if solver_position_iteration_count is not None:
        deformable_body_api.CreateSolverPositionIterationCountAttr().Set(solver_position_iteration_count)
    if vertex_velocity_damping is not None:
        deformable_body_api.CreateVertexVelocityDampingAttr().Set(vertex_velocity_damping)
    if sleep_damping is not None:
        deformable_body_api.CreateSleepDampingAttr().Set(sleep_damping)
    if sleep_threshold is not None:
        deformable_body_api.CreateSleepThresholdAttr().Set(sleep_threshold)
    if settling_threshold is not None:
        deformable_body_api.CreateSettlingThresholdAttr().Set(settling_threshold)
    if self_collision is not None:
        deformable_body_api.CreateSelfCollisionAttr().Set(self_collision)
    if self_collision_filter_distance is not None:
        deformable_body_api.CreateSelfCollisionFilterDistanceAttr().Set(self_collision_filter_distance)

    if collision_indices:
        deformable_body_api.CreateCollisionIndicesAttr().Set(collision_indices)
    if collision_rest_points:
        deformable_body_api.CreateCollisionRestPointsAttr().Set(collision_rest_points)
    if simulation_indices:
        deformable_body_api.CreateSimulationIndicesAttr().Set(simulation_indices)
    if simulation_rest_points:
        deformable_body_api.CreateSimulationRestPointsAttr().Set(simulation_rest_points)

    # Custom attributes
    if not simulation_rest_points:
        prim.CreateAttribute("physxDeformable:simulationHexahedralResolution", Sdf.ValueTypeNames.UInt).Set(
            simulation_hexahedral_resolution
        )

    if not collision_rest_points:
        prim.CreateAttribute("physxDeformable:collisionSimplification", Sdf.ValueTypeNames.Bool).Set(
            collision_simplification
        )
        prim.CreateAttribute("physxDeformable:collisionSimplificationRemeshing", Sdf.ValueTypeNames.Bool).Set(
            collision_simplification_remeshing
        )
        prim.CreateAttribute("physxDeformable:collisionSimplificationRemeshingResolution", Sdf.ValueTypeNames.UInt).Set(
            collision_simplification_remeshing_resolution
        )
        prim.CreateAttribute("physxDeformable:collisionSimplificationTargetTriangleCount", Sdf.ValueTypeNames.UInt).Set(
            collision_simplification_target_triangle_count
        )
        prim.CreateAttribute("physxDeformable:collisionSimplificationForceConforming", Sdf.ValueTypeNames.Bool).Set(
            collision_simplification_force_conforming
        )

    if embedding:
        prim.CreateAttribute("physxDeformable:collisionToSimulationEmbedding", Sdf.ValueTypeNames.IntArray).Set(
            embedding
        )

    # turn on ccd (In the schema, it is off by default)
    deformable_body_api.CreateEnableCCDAttr().Set(True)

    PhysxSchema.PhysxCollisionAPI.Apply(prim)

    return

def add_physics_material_to_prim(stage, prim, materialPath):
    bindingAPI = UsdShade.MaterialBindingAPI.Apply(prim)
    materialPrim = UsdShade.Material(stage.GetPrimAtPath(materialPath))
    bindingAPI.Bind(materialPrim, UsdShade.Tokens.weakerThanDescendants, "physics")