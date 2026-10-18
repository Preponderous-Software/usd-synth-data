"""Write a sampled :class:`~usd_synth.scenegen.tabletop.Scene` as one flattened ``.usda`` layer.

Layout (ARCHITECTURE.md, "USD stage layout")::

    /World (defaultPrim, upAxis Z, metersPerUnit 1)
      Env/Support, Env/Background, Objects/inst_NNNN, Distractors/dis_NNNN,
      Lights/Key_N + Lights/Dome, Camera, Looks/mat_NNNN

Every float is rounded to a fixed number of decimals before it is authored, so the text is
byte-stable across platforms. User assets are added by reference and then flattened, because
Blender's USD importer does not handle composition arcs.
"""

from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from .primitives import MeshData
from .tabletop import FLOOR_SIZE_M, FLOOR_Z_M, Instance, Material, Scene

DECIMALS = 6
NS = "userProperties:usd_synth:"
DOC = "usd-synth-data scene"


def _r(value) -> float:
    return round(float(value), DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0


def _vec3(values) -> Gf.Vec3f:
    return Gf.Vec3f(*(_r(v) for v in values))


def _matrix(rows) -> Gf.Matrix4d:
    return Gf.Matrix4d(*[[_r(v) for v in row] for row in rows])


def _set(prim: Usd.Prim, name: str, type_name, value) -> None:
    prim.CreateAttribute(NS + name, type_name, custom=True).Set(value)


def _transform(xformable: UsdGeom.Xformable, rows) -> None:
    xformable.AddTransformOp().Set(_matrix(rows))


def _material(stage: Usd.Stage, mat: Material) -> UsdShade.Material:
    path = f"/World/Looks/{mat.name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(_r(mat.roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(_r(mat.metallic))
    color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if mat.texture:
        reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(mat.texture))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        color.ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        color.Set(_vec3(mat.base_color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mesh(stage: Usd.Stage, path: str, data: MeshData, material, offset=None, uv=False):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = data.points if offset is None else data.points + offset
    mesh.CreatePointsAttr(Vt.Vec3fArray([_vec3(p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(data.face_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(data.face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([_vec3(lo), _vec3(hi)]))
    if uv:  # planar projection, enough for random texture assignment
        span = np.maximum(hi[:2] - lo[:2], 1e-9)
        st = [(pts[i, :2] - lo[:2]) / span for i in data.face_indices]
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        primvar.Set(Vt.Vec2fArray([Gf.Vec2f(_r(s), _r(t)) for s, t in st]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _instance(stage: Usd.Stage, scene: Scene, inst: Instance) -> None:
    name = f"{'inst' if inst.kind == 'object' else 'dis'}_{inst.instance_id:04d}"
    path = f"/World/{'Objects' if inst.kind == 'object' else 'Distractors'}/{name}"
    xform = UsdGeom.Xform.Define(stage, path)
    _transform(xform, inst.matrix())
    prim = xform.GetPrim()
    _set(prim, "instance_id", Sdf.ValueTypeNames.Int, inst.instance_id)
    _set(prim, "category_id", Sdf.ValueTypeNames.Int, inst.category_id)
    _set(prim, "category_name", Sdf.ValueTypeNames.String, inst.category_name)
    _set(prim, "asset", Sdf.ValueTypeNames.String, inst.asset)
    _set(prim, "kind", Sdf.ValueTypeNames.String, inst.kind)
    if inst.projected_bbox is not None:
        box = Gf.Vec4d(*(round(v, 3) + 0.0 for v in inst.projected_bbox))
        _set(prim, "projected_bbox", Sdf.ValueTypeNames.Double4, box)
        _set(prim, "projected_area_px", Sdf.ValueTypeNames.Double, round(inst.projected_area_px, 3))
    material = _material(stage, inst.material)
    if inst.mesh is not None:
        _mesh(
            stage, f"{path}/{name}_geom", inst.mesh, material, uv=inst.material.texture is not None
        )
    else:
        asset = stage.DefinePrim(f"{path}/asset")
        asset.GetReferences().AddReference(inst.asset_path)
        UsdShade.MaterialBindingAPI.Apply(asset).Bind(material)


def build_stage(scene: Scene) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    _set(world, "scene_index", Sdf.ValueTypeNames.Int, scene.index)
    _set(world, "dataset_seed", Sdf.ValueTypeNames.Int64, scene.dataset_seed)
    _set(world, "image_width", Sdf.ValueTypeNames.Int, scene.width)
    _set(world, "image_height", Sdf.ValueTypeNames.Int, scene.height)
    _set(world, "support_kind", Sdf.ValueTypeNames.String, scene.support_kind)
    _set(world, "categories", Sdf.ValueTypeNames.StringArray, Vt.StringArray(scene.categories))

    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Xform.Define(stage, "/World/Env")
    support = UsdGeom.Xform.Define(stage, "/World/Env/Support").GetPrim()
    _set(support, "category_name", Sdf.ValueTypeNames.String, "support")
    support_mat = _material(stage, scene.support_material)
    for name, mesh, offset in scene.support_parts:
        _mesh(stage, f"/World/Env/Support/{name}", mesh, support_mat, offset=offset)
    h = FLOOR_SIZE_M / 2
    floor = MeshData(
        np.array([[-h, -h, FLOOR_Z_M], [h, -h, FLOOR_Z_M], [h, h, FLOOR_Z_M], [-h, h, FLOOR_Z_M]]),
        [4],
        [0, 1, 2, 3],
    )
    _mesh(stage, "/World/Env/Background", floor, _material(stage, scene.floor_material))

    UsdGeom.Xform.Define(stage, "/World/Objects")
    UsdGeom.Xform.Define(stage, "/World/Distractors")
    for inst in scene.instances:
        _instance(stage, scene, inst)

    UsdGeom.Xform.Define(stage, "/World/Lights")
    for light in scene.lights:
        path = f"/World/Lights/{light.name}"
        if light.kind == "dome":
            lux = UsdLux.DomeLight.Define(stage, path)
            if light.texture:
                lux.CreateTextureFileAttr(Sdf.AssetPath(light.texture))
        else:
            lux = UsdLux.SphereLight.Define(stage, path)
            lux.CreateRadiusAttr(_r(light.radius))
            lux.AddTranslateOp().Set(Gf.Vec3d(*(_r(v) for v in light.position)))
        lux.CreateIntensityAttr(_r(light.intensity))
        lux.CreateColorAttr(_vec3(light.color))

    cam = scene.camera
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    _transform(camera, cam.usd_matrix())
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(_r(cam.focal_length_mm))
    camera.CreateHorizontalApertureAttr(_r(cam.sensor_width_mm))
    camera.CreateVerticalApertureAttr(_r(cam.vertical_aperture_mm))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    cprim = camera.GetPrim()
    _set(cprim, "focal_length_mm", Sdf.ValueTypeNames.Double, _r(cam.focal_length_mm))
    _set(cprim, "sensor_width_mm", Sdf.ValueTypeNames.Double, _r(cam.sensor_width_mm))
    return stage


def scene_to_usda(scene: Scene) -> str:
    """Flatten the scene's stage into one layer and return its ``.usda`` text."""
    layer = build_stage(scene).Flatten()
    layer.documentation = DOC
    return layer.ExportToString()
