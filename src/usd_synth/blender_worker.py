"""Render worker, executed by Blender (see ``usd_synth.render`` for the command line).

Command: ``blender -b --factory-startup --python blender_worker.py -- job.json``.

Runs inside Blender's bundled Python, so it imports only ``bpy`` and the standard library and
never imports the ``usd_synth`` package. The job description (JSON) lists the scenes of one
batch; each scene is imported into an empty file, rendered with Cycles, and written as:

* ``rgb.png`` — 8-bit RGB with the scene's view transform;
* ``aux.exr`` — multilayer EXR holding the ``Depth`` and ``IndexOB`` passes (and ``Combined``);
* ``render.json`` — per-scene timings and environment facts.

Instance IDs come from the ``usd_synth:instance_id`` custom property that Blender's USD importer
creates from ``userProperties:usd_synth:instance_id``; if the property is missing, the ID is
parsed from the object name (``inst_0007`` / ``dis_0007`` / ``inst_0007_geom``).
"""

import json
import os
import re
import sys
import time

import bpy

ID_PROP = "usd_synth:instance_id"
NAME_RE = re.compile(r"^(?:inst|dis)_(\d+)")
GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")


def log(event, **fields):
    print("USD_SYNTH " + json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def instance_id(obj):
    if ID_PROP in obj.keys():
        return int(obj[ID_PROP])
    match = NAME_RE.match(obj.name)
    return int(match.group(1)) if match else None


def assign_pass_indices():
    """Set ``pass_index`` of every instance (labelled object or distractor) and its descendants
    to its instance ID; everything else gets 0. Distractors keep their IDs so the label stage
    can make them ``iscrowd`` regions or omit them without re-rendering."""
    for obj in bpy.data.objects:
        obj.pass_index = 0
    assigned = {}
    roots = [o for o in bpy.data.objects if instance_id(o) is not None]
    for obj in roots:
        iid = instance_id(obj)
        for target in [obj, *obj.children_recursive]:
            target.pass_index = iid
            if target.type == "MESH":
                assigned[target.name] = iid
    return assigned


def configure_device(scene, job):
    scene.cycles.device = "CPU"
    if job["device"] != "gpu":
        return "cpu"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in GPU_BACKENDS:
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return f"gpu:{backend.lower()}"
    log("warning", message="render.device is gpu but no GPU backend is available; using CPU")
    return "cpu"


def configure_world(scene, dome):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    background = nodes.get("Background")
    if dome is None:
        background.inputs["Strength"].default_value = 0.0
        return
    background.inputs["Strength"].default_value = float(dome["intensity"])
    background.inputs["Color"].default_value = (*dome["color"], 1.0)
    if dome.get("texture"):
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(dome["texture"])
        world.node_tree.links.new(env.outputs["Color"], background.inputs["Color"])


def configure_scene(scene, job, spec):
    scene.render.engine = "CYCLES"
    device = configure_device(scene, job)
    scene.cycles.samples = int(job["samples"])
    scene.cycles.use_denoising = bool(job["denoise"])
    scene.cycles.seed = int(spec["seed"])
    if job.get("threads"):
        scene.render.threads_mode = "FIXED"
        scene.render.threads = int(job["threads"])
    scene.render.resolution_x = int(job["width"])
    scene.render.resolution_y = int(job["height"])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    layer = scene.view_layers[0]
    layer.use_pass_combined = True
    layer.use_pass_z = True
    layer.use_pass_object_index = True

    cameras = [o for o in scene.objects if o.type == "CAMERA"]
    if len(cameras) != 1:
        raise RuntimeError(f"expected exactly one camera, found {len(cameras)}")
    camera = cameras[0]
    scene.camera = camera
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = float(spec["camera"]["sensor_width_mm"])
    camera.data.lens = float(spec["camera"]["focal_length_mm"])
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0
    configure_world(scene, spec.get("dome"))
    return device


def write_outputs(scene, out_dir):
    image = bpy.data.images["Render Result"]
    settings = scene.render.image_settings
    settings.media_type = "MULTI_LAYER_IMAGE"
    settings.file_format = "OPEN_EXR_MULTILAYER"
    settings.color_depth = "32"
    settings.exr_codec = "ZIP"
    image.save_render(os.path.join(out_dir, "aux.exr"), scene=scene)
    settings.media_type = "IMAGE"
    settings.file_format = "PNG"
    settings.color_mode = "RGB"
    settings.color_depth = "8"
    settings.compression = 15
    image.save_render(os.path.join(out_dir, "rgb.png"), scene=scene)


def render_one(job, spec):
    timings = {}
    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(
        filepath=spec["usd"],
        import_cameras=True,
        import_lights=True,
        import_materials=True,
        import_usd_preview=True,
        property_import_mode="ALL",
    )
    scene = bpy.context.scene
    ids = assign_pass_indices()
    device = configure_scene(scene, job, spec)
    timings["import"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    bpy.ops.render.render(write_still=False)
    timings["render"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    os.makedirs(spec["out_dir"], exist_ok=True)
    write_outputs(scene, spec["out_dir"])
    timings["write"] = time.perf_counter() - t2
    info = {
        "usd": spec["usd"],
        "blender": bpy.app.version_string,
        "device": device,
        "threads": scene.render.threads,
        "samples": scene.cycles.samples,
        "seed": scene.cycles.seed,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "pass_indices": ids,
        "timings": timings,
    }
    with open(os.path.join(spec["out_dir"], "render.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, sort_keys=True)
    return info


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) != 1:
        raise SystemExit("usage: blender -b --python blender_worker.py -- job.json")
    with open(argv[0], encoding="utf-8") as fh:
        job = json.load(fh)
    log("ready", blender=bpy.app.version_string, time=time.time())
    failures = 0
    for spec in job["scenes"]:
        try:
            info = render_one(job, spec)
            log("done", out_dir=spec["out_dir"], timings=info["timings"])
        except Exception as err:  # keep rendering the rest of the batch
            failures += 1
            log("error", out_dir=spec["out_dir"], message=f"{type(err).__name__}: {err}")
    sys.stdout.flush()
    os._exit(1 if failures else 0)  # Blender's own exit can mask the status


if __name__ == "__main__":
    main()
