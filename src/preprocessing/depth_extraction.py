from typing import Dict, Optional

import pyrender
import trimesh
from PIL import Image

from pygarment.meshgen.render.pythonrender import create_camera, create_lights
from pygarment.meshgen.sim_config import PathCofig


def render(
    paths: PathCofig,
    pyrender_garm_mesh: Optional[pyrender.Mesh] = None,
    side: str = "front",
    render_props: Optional[Dict] = None,
    save_color: bool = True,
    save_depth: bool = True,
):
    if render_props and "resolution" in render_props:
        view_width, view_height = render_props["resolution"]
    else:
        view_width, view_height = 1080, 1080
    # Create a pyrender scene
    scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 0.0))  # Transparent!

    # Create a pyrender mesh object from the trimesh object
    # Add the mesh to the scene
    assert pyrender_garm_mesh is not None, "Garment mesh should be provided"
    if pyrender_garm_mesh is not None:
        scene.add(pyrender_garm_mesh)

    camera_location = None
    # camera_location = (
    #     render_props["front_camera_location"]
    #     if render_props and "front_camera_location" in render_props
    #     else None
    # )

    breakpoint()
    create_camera(
        pyrender,
        pyrender_garm_mesh,
        scene,
        side,
        camera_location=camera_location,
    )

    create_lights(scene, intensity=80.0)

    # Create a renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=view_width, viewport_height=view_height
    )

    # Render the scene
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    if save_color:
        image = Image.fromarray(color)
        image.save(paths.render_path(side + "_garment"), "PNG")

    if save_depth:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.imshow(depth)

        image = Image.fromarray((depth * 255).astype("uint8"))
        image.save(paths.render_path("depth"), "PNG")

        path = str(paths.render_path("depth"))
        path = path.replace(".png", ".npy")
        np.save(path, depth)


def load_garment_meshes(paths: PathCofig, texture_on_garment: bool = True):
    # Load garment mesh
    # NOTE: Includes the texture
    garm_mesh = trimesh.load_mesh(str(paths.g_sim_compressed))
    garm_mesh.vertices = garm_mesh.vertices / 100  # scale to m

    # Material adjustments
    if texture_on_garment:
        material = garm_mesh.visual.material.to_pbr()
        material.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
        material.doubleSided = True  # color both face sides
        # NOTE remove transparency -- add white background just in case
        white_back = Image.new(
            "RGBA", material.baseColorTexture.size, color=(255, 255, 255, 255)
        )
        white_back.paste(material.baseColorTexture)
        material.baseColorTexture = white_back.convert("RGB")

        garm_mesh.visual.material = material

    pyrender_garm_mesh = pyrender.Mesh.from_trimesh(garm_mesh, smooth=True)

    return pyrender_garm_mesh


def render_images(
    paths: PathCofig,
    render_props,
    texture_on_garment=True,
    save_color=True,
    save_depth=True,
):
    pyrender_garm_mesh = load_garment_meshes(paths, texture_on_garment)

    for side in render_props["sides"]:
        render(
            paths,
            pyrender_garm_mesh=pyrender_garm_mesh,
            side=side,
            render_props=render_props,
            save_color=save_color,
            save_depth=save_depth,
        )
