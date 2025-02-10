from pathlib import Path
from time import time
from typing import List, Tuple, Union

import igl
import numpy as np

from src.preprocessing.object_loading import read_mesh

_MAX_SIZE = 1 << 16


class _Rectangle:
    """Rectangle class for packing UV islands in unwarp_UV function"""

    def __init__(
        self,
        rect_id: int,
        width,
        height,
        x=0,
        y=0,
    ):
        self.rect_id = rect_id
        self.x = x
        self.y = y
        self.w = width
        self.h = height

    @staticmethod
    def pack_rows(
        rects: List["_Rectangle"], container_size: int = 1024
    ) -> bool:
        rects = sorted(rects, key=lambda r: r.h, reverse=True)

        cur_y = 0
        cur_x = 0
        max_row_height = 0
        for r in rects:
            if cur_x + r.w > container_size:
                cur_x = 0
                cur_y += max_row_height
                max_row_height = 0

            # Cannot pack this rectangle
            if cur_y + r.h > container_size:
                return False

            r.x = cur_x
            r.y = cur_y

            cur_x += r.w
            max_row_height = max(max_row_height, r.h)

        return True


def _uv_connected_components(face_texture_coords):
    # Find connected components of face and vertex texture coords
    face_components = igl.facet_components(face_texture_coords)
    num_ccs = max(face_components) + 1

    # Get vertex components
    # igl.vertex_components returns mismatched components' id
    # (igl.adjacency_matrix != igl.facet_adjacency_matrix)
    vert_components = np.zeros(face_texture_coords.max() + 1, dtype=int)
    for i in range(num_ccs):
        verts_in_cc = np.unique(face_texture_coords[face_components == i])
        vert_components[verts_in_cc] = i

    return vert_components, face_components, num_ccs


def unwarp_UV(
    texture_coords,
    face_texture_coords,
    padding: float,
) -> Tuple[np.ndarray, List]:
    # Unwrap uvs for each connected component------------------------
    vert_components, face_components, num_ccs = _uv_connected_components(
        face_texture_coords
    )

    # transform all UVs to update obj file
    all_uvs = np.zeros((texture_coords.shape[0], 2))
    boundary_uv_to_draw = []  # only draw the boundary UVs

    translate_Y = 0
    translate_X = 0

    half_padding = padding / 2

    # Loop through each connected component
    list_rects = []
    for i in range(num_ccs):

        # Get faces and vertices of connected component
        faces_in_cc = np.where(face_components == i)[0]
        face_vts_in_cc = face_texture_coords[faces_in_cc]

        # Find boundary loop
        bound_verts = igl.boundary_loop(face_vts_in_cc)
        bound_vert_pos = texture_coords[bound_verts]

        # get all vertices of connected component
        verts_in_cc = np.where(vert_components == i)[0]
        all_vert_pos = texture_coords[verts_in_cc]

        # Shift component by bounding box
        bbox = bound_vert_pos.min(axis=0), bound_vert_pos.max(axis=0)
        _bbox = all_vert_pos.min(axis=0), all_vert_pos.max(axis=0)
        assert np.allclose(bbox, _bbox), f"bbox: {bbox} != _bbox: {_bbox}"

        bbox_len_Y = (bbox[1][1] - bbox[0][1]) + padding
        bbox_len_X = (bbox[1][0] - bbox[0][0]) + padding

        list_rects.append(_Rectangle(i, bbox_len_X, bbox_len_Y))

    # Container size search
    lower, upper = 0, _MAX_SIZE
    container_size = 0
    while lower < upper:
        mid = (lower + upper) // 2
        if _Rectangle.pack_rows(list_rects, mid):
            container_size = mid
            upper = mid
        else:
            lower = mid + 1

    assert container_size > 0, "Cannot pack UV islands into a texture"
    _Rectangle.pack_rows(list_rects, container_size)

    print(f"[DEBUG] Container size: {container_size}")

    if not container_size:
        raise ValueError("Cannot pack UV islands into a texture")

    for i in range(num_ccs):
        ccs_id = list_rects[i].rect_id
        translate_X = list_rects[i].x
        translate_Y = list_rects[i].y

        # Get faces and vertices of connected component
        faces_in_cc = np.where(face_components == ccs_id)[0]
        face_vts_in_cc = face_texture_coords[faces_in_cc]

        # get all vertices of connected component
        verts_in_cc = np.where(vert_components == ccs_id)[0]
        all_vert_pos = texture_coords[verts_in_cc]

        # Find boundary loop
        bound_verts = igl.boundary_loop(face_vts_in_cc)
        bound_vert_pos = texture_coords[bound_verts]
        # Translation to center
        _min_X, _min_Y = bound_vert_pos.min(axis=0)

        # translate boundary positions
        verts_translated_bound = [
            (
                x + translate_X + half_padding - _min_X,
                y + translate_Y + half_padding - _min_Y,
            )
            for x, y in bound_vert_pos
        ]
        boundary_uv_to_draw.append(verts_translated_bound)

        # translate all positions
        verts_translated = np.array(
            [
                (
                    x + translate_X + half_padding - _min_X,
                    y + translate_Y + half_padding - _min_Y,
                )
                for x, y in all_vert_pos
            ]
        )
        all_uvs[verts_in_cc] = verts_translated

    return all_uvs, boundary_uv_to_draw


def normalize_squared_UVs(
    all_uvs,
    padding: float,
) -> Tuple[np.ndarray, float]:
    """Normalize UVs and turn into a square"""
    # Padding to both sides of each UVs, so the padding is halved
    # Max coordinates already include first half of padding
    # --hpd|UVs|hpd--
    half_padding = padding / 2

    # normalize all_uvs
    uv_list_raw = np.array(all_uvs)
    uv_list = uv_list_raw

    norm_x = max(uv_list_raw[:, 0]) + half_padding
    norm_y = max(uv_list_raw[:, 1]) + half_padding
    norm = max(norm_x, norm_y)

    uv_list[:, 0] = uv_list_raw[:, 0] / norm
    uv_list[:, 1] = uv_list_raw[:, 1] / norm

    return uv_list, norm


def _in_triangle(p, a, b, c) -> bool:
    """Check if point p is inside triangle abc"""

    def _sign(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    d1 = _sign(p, a, b)
    d2 = _sign(p, b, c)
    d3 = _sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def interpolate_uv(
    uv_coords: np.ndarray,
    norm_vertex_coords: np.ndarray,
    vertex_faces: np.ndarray,
    texture_faces: np.ndarray,
    min_tolerate: float = 0,
    max_tolerate: float = 0,
    max_expected_ratio: float = 100,
    tolerate_eps: float = 1e-6,
    min_uv_size: int = 2048,
):
    """Interpolate UV coordinates to create a geometry image"""
    # in case read from yaml
    tolerate_eps = float(tolerate_eps)

    def _size_binary_search(tol):
        left, right = min_uv_size, _MAX_SIZE
        ans = -1
        while left < right:
            mid = left + ((right - left) >> 1)
            scaled_coords = np.round(uv_coords * mid, 0).astype(int)
            if (
                np.unique(scaled_coords, axis=0).shape[0] / uv_coords.shape[0]
            ) >= 1 - tol - 1e-9:
                # if (unique.shape[0] == uv_coords.shape[0]):
                ans = mid
                right = mid
            else:
                left = mid + 1
        # Always found a valid size with that _MAX_SIZE (with current data)
        assert ans != -1, "Cannot find a valid size for UV"

        return ans

    actual_size = _size_binary_search(min_tolerate)
    found_tol = min_tolerate
    if actual_size**2 / uv_coords.shape[0] > max_expected_ratio:
        # Only for last attempt
        actual_size = _size_binary_search(max_tolerate)
        found_tol = max_tolerate

        left_tol, right_tol = min_tolerate, max_tolerate
        while right_tol - left_tol > tolerate_eps:
            mid_tol = (left_tol + right_tol) / 2
            found_size = _size_binary_search(mid_tol)
            if found_size**2 / uv_coords.shape[0] <= max_expected_ratio:
                actual_size = found_size
                found_tol = mid_tol
                right_tol = mid_tol
            else:
                left_tol = mid_tol

    width, height = actual_size, actual_size

    print(f"[DEBUG] Actual size: {actual_size}")
    print(f"[DEBUG] Found tolerance: {found_tol}")

    img_array = np.zeros((height, width, 3), dtype=float)

    # scale to original uv coordinates and flip the V axis
    scaled_coords = uv_coords * [width, height]
    scaled_coords[:, 1] = height - scaled_coords[:, 1]

    print("[DEBUG] Number of original vertices:", scaled_coords.shape[0])
    print(
        "[DEBUG] Number of overlapped vertices: ",
        scaled_coords.shape[0]
        - np.unique(scaled_coords.astype(int), axis=0).shape[0],
    )

    def _interpolate(face_id, cnt):
        a_id, b_id, c_id = texture_faces[face_id]
        a = scaled_coords[a_id]
        b = scaled_coords[b_id]
        c = scaled_coords[c_id]

        # bounding box for the triangle
        min_x = np.floor(max(0, min(a[0], b[0], c[0]))).astype(int)
        max_x = np.ceil(min(width, max(a[0], b[0], c[0]))).astype(int)
        min_y = np.floor(max(0, min(a[1], b[1], c[1]))).astype(int)
        max_y = np.ceil(min(height, max(a[1], b[1], c[1]))).astype(int)

        # vertex colors = 3d coordinate
        vertex_coords = norm_vertex_coords[vertex_faces[face_id]]
        a_color, b_color, c_color = vertex_coords

        # barycentric interpolation
        v0 = b - a
        v1 = c - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        denom = d00 * d11 - d01 * d01

        assert not np.isclose(denom, 0), "Denominator is zero"

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                p = np.array([x, y])
                if _in_triangle(p, a, b, c):
                    cnt += 1
                    v2 = p - a
                    d20 = np.dot(v2, v0)
                    d21 = np.dot(v2, v1)

                    v = (d11 * d20 - d01 * d21) / denom

                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1 - v - w

                    # Compute interpolated color
                    color = a_color * u + b_color * v + c_color * w

                    img_array[y, x] = color

        return cnt

    all_time = time()
    cnt = 0
    for face_id in range(texture_faces.shape[0]):
        cnt = _interpolate(face_id, cnt)

    print("[DEBUG] Number of interpolated vertices (origin inclusive): ", cnt)
    print("[DEBUG] Time: ", time() - all_time)
    return img_array


def create_geometry_uv(
    obj_file_path: Union[str, Path],
    texture_file_path: Union[str, Path],
    geometry_uv_path: Union[str, Path],
    max_dim: int = 300,
    island_padding_percent: int = 2,
    plot: bool = False,
    **kwargs,
):
    """Create geometry UV image from obj & texture file"""
    (
        uv_coords,
        vertex_coords,
        texture_faces,
        vertex_faces,
    ) = read_mesh(obj_file_path, texture_file_path)

    assert 0 < island_padding_percent <= 100, "Invalid padding percentage"
    _max_ori_uv_size = max(uv_coords.max(axis=0) - uv_coords.min(axis=0))
    padding = _max_ori_uv_size * (island_padding_percent / 100)

    all_uvs, _ = unwarp_UV(
        uv_coords,
        texture_faces,
        padding=padding,
    )

    uv_list, _ = normalize_squared_UVs(
        all_uvs,
        padding=padding,
    )

    # Create geometry uv file
    norm_vertex_coords = (vertex_coords + max_dim) / (2 * max_dim)
    img_array = interpolate_uv(
        uv_list,
        norm_vertex_coords,
        vertex_faces,
        texture_faces,
        **kwargs,
    )

    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(img_array)
        plt.show()

    np.savez_compressed(geometry_uv_path, img_array)
    return img_array
