from pathlib import Path
from typing import List, Tuple, Union

import imagesize
import numpy as np
import trimesh


def read_obj_mesh_manually(
    obj_file_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read an OBJ file and extract UV coordinates.
    Args:
        obj_file_path: Path to the OBJ file.
    Returns:
        uv_coords: UV coordinates.
        vertex_coords: Vertex coordinates.
        tid_faces: List of texture faces respective to ids of uv coords.
        vid_faces: List of faces respective to vertex 3D coords.
    """
    assert Path(obj_file_path).suffix == ".obj", "File must be an OBJ file."

    uv_coords = []
    vertex_coords = []
    # vertex ids of face
    vid_faces = []
    # texture ids of face
    tid_faces = []

    with open(obj_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == "vt":  # Texture coordinate line
                u, v = map(float, parts[1:3])  # Extract u and v
                uv_coords.append((u, v))
            elif parts[0] == "f":  # Extract the vertex index
                # obj file index start from 1
                vertex_ids = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                texture_ids = [int(p.split("/")[1]) - 1 for p in parts[1:]]

                vid_faces.append(vertex_ids)
                tid_faces.append(texture_ids)
            elif parts[0] == "v":
                vertex_coord = list(map(float, parts[1:]))
                vertex_coords.append(vertex_coord)
    return (
        np.array(uv_coords),
        np.array(vertex_coords),
        np.array(tid_faces),
        np.array(vid_faces),
    )


def v_id_map(vertices: np.ndarray) -> Tuple[List[int], np.ndarray]:
    """Map vertices to new ids.

    Args:
        vertices: List of vertices.
    Returns:
        v_map: Mapping of old vertex ids to new vertex ids.
        deduplicated_vertices: List of deduplicated vertices
        following the order of new vertex ids.
    """
    # Temporarily resolve the issue of duplicate vertices by trimesh
    # See Loading segmentation and vertex labels for ply mesh in:
    # https://www.research-collection.ethz.ch/handle/20.500.11850/690432
    v_map: List[int] = [0] * len(vertices)
    deduplicated_vertices = [vertices[0]]
    for i in range(1, len(vertices)):
        if all(vertices[i - 1] == vertices[i]):
            v_map[i] = v_map[i - 1]
        else:
            v_map[i] = v_map[i - 1] + 1
            deduplicated_vertices.append(vertices[i])
    return v_map, np.array(deduplicated_vertices)


def read_mesh(
    file_path: Union[str, Path],
    texture_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert Path(file_path).suffix in [
        ".obj",
        ".ply",
    ], "File must be an OBJ or PLY file."

    mesh = trimesh.load(file_path)

    # Texture
    # UV coordinates
    uv_coords = np.array(mesh.visual.uv)
    width, height = imagesize.get(texture_path)
    uv_coords = uv_coords * [width, height]
    # Create texture faces
    tid_faces = np.array(mesh.faces)

    # 3D vertice
    # Vertex coordinates
    vertex_coords = np.array(mesh.vertices)
    # Vertex faces
    v_map, dedup_vertex_coords = v_id_map(vertex_coords)
    vid_faces = np.array([[v_map[i] for i in face] for face in tid_faces])

    return (
        uv_coords,
        dedup_vertex_coords,
        tid_faces,
        vid_faces,
    )
