"""
    To be used in Python 3.6+ due to dependencies
"""

import os
import random
import string
from copy import copy

import numpy as np
from scipy.spatial.transform import Rotation as R

# Correct dependencies on Win
# https://stackoverflow.com/questions/46265677/get-cairosvg-working-in-windows
if "Windows" in os.environ.get("OS", ""):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.environ["path"] += f';{os.path.abspath(dir_path + "/cairo_dlls/")}'

import cairosvg
import matplotlib.pyplot as plt
import svgpathtools as svgpath
import svgwrite as sw

# my
from pygarment import data_config

from . import core
from .utils import *


class VisPattern(core.ParametrizedPattern):
    """
    "Visualizible" pattern wrapper of pattern specification in custom JSON format.
    Input:
        * Pattern template in custom JSON format
    Output representations:
        * Pattern instance in custom JSON format
            * In the current state
        * SVG (stitching info is lost)
        * PNG for visualization

    Not implemented:
        * Support for patterns with darts

    NOTE: Visualization assumes the pattern uses cm as units
    """

    # ------------ Interface -------------

    def __init__(self, pattern_file=None):
        super().__init__(pattern_file)

        self.px_per_unit = 3

    def serialize(
        self,
        path,
        to_subfolder=True,
        tag="",
        with_3d=True,
        with_text=True,
        view_ids=True,
        with_printable=False,
        empty_ok=False,
    ):

        log_dir = super().serialize(
            path, to_subfolder, tag=tag, empty_ok=empty_ok
        )
        # If we are still here, but pattern is empty, don't generate an image
        if len(self.panel_order()) == 0:
            return log_dir

        if tag:
            tag = "_" + tag
        svg_file = os.path.join(log_dir, (self.name + tag + "_pattern.svg"))
        svg_printable_file = os.path.join(
            log_dir, (self.name + tag + "_print_pattern.svg")
        )
        png_file = os.path.join(log_dir, (self.name + tag + "_pattern.png"))
        pdf_file = os.path.join(
            log_dir, (self.name + tag + "_print_pattern.pdf")
        )
        png_3d_file = os.path.join(
            log_dir, (self.name + tag + "_3d_pattern.png")
        )

        # save visualtisation
        self._save_as_image(svg_file, png_file, with_text, view_ids)
        if with_3d:
            self._save_as_image_3D(png_3d_file)
        if with_printable:
            self._save_as_pdf(
                svg_printable_file, pdf_file, with_text, view_ids
            )

        return log_dir

    # -------- Drawing ---------

    def _verts_to_px_coords(self, vertices, translation_2d):
        """Convert given vertices and panel (2D) translation to px coordinate frame & units"""
        # Flip Y coordinate (in SVG Y looks down)
        vertices[:, 1] *= -1
        translation_2d[1] *= -1
        # Put upper left corner of the bounding box at zero
        offset = np.min(vertices, axis=0)
        vertices = vertices - offset
        translation_2d = translation_2d + offset
        return vertices, translation_2d

    def _flip_y(self, point):
        """
        To get to image coordinates one might need to flip Y axis
        """
        flipped_point = list(point)  # top-level copy
        flipped_point[1] *= -1
        return flipped_point

    def _draw_a_panel(self, panel_name, apply_transform=True, fill=True):
        """
        Adds a requested panel to the svg drawing with given offset and scaling
        Assumes (!!)
            that edges are correctly oriented to form a closed loop
        Returns
            the lower-right vertex coordinate for the convenice of future offsetting.
        """
        attributes = {
            # fill with white
            "fill": "rgb(227,175,186)" if fill else "rgb(255,255,255)",
            "stroke": "rgb(51,51,51)",
            "stroke-width": "0.2",
        }

        panel = self.pattern["panels"][panel_name]
        vertices = np.asarray(panel["vertices"])
        vertices, translation = self._verts_to_px_coords(
            vertices, np.array(panel["translation"][:2])
        )  # Only XY

        # draw edges
        segs = [self._edge_as_curve(vertices, edge) for edge in panel["edges"]]
        path = svgpath.Path(*segs)
        if apply_transform:
            # Placement and rotation according to the 3D location
            # But flatterened on 2D
            # Z-fist rotation to only reflect rotation visible in XY plane
            # NOTE: Heuristic, might be bug-prone
            rotation = R.from_euler(
                "XYZ", panel["rotation"], degrees=True
            )  # XYZ

            # Estimate degree of rotation of Y axis
            # NOTE: Ox sometimes gets flipped because of
            # Gimbal locks of this Euler angle representation
            res = rotation.apply([0, 1, 0])
            flat_rot_angle = np.rad2deg(vector_angle([0, 1], res[:2]))
            path = path.rotated(
                degs=-flat_rot_angle, origin=list_to_c(vertices[0])
            )
            # NOTE: rot/transl order is important!
            path = path.translated(list_to_c(translation))

        return path, attributes, panel["translation"][-1] >= 0

    def _add_panel_annotations(
        self,
        drawing,
        panel_name,
        path: svgpath.Path,
        with_text=True,
        view_ids=True,
    ):
        """Adds a annotations for requested panel to the svg drawing with given offset and scaling
        Assumes (!!)
            that edges are correctly oriented to form a closed loop
        Returns
            the lower-right vertex coordinate for the convenice of future offsetting.
        """
        bbox = path.bbox()
        panel_center = np.array(
            [(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2]
        )

        if with_text:
            # + np.array([-len(panel_name) * 12 / 2, 3])
            text_insert = panel_center
            drawing.add(
                drawing.text(
                    panel_name,
                    insert=text_insert,
                    fill="rgb(31,31,31)",
                    font_size="7",
                    text_anchor="middle",
                    dominant_baseline="middle",
                )
            )

        if view_ids:
            # name vertices
            for idx in range(len(path)):
                seg = path[idx]
                ver = c_to_np(seg.start)
                drawing.add(
                    drawing.text(
                        str(idx),
                        insert=ver,
                        fill="rgb(245,96,66)",
                        font_size="7",
                    )
                )
            # name edges
            for idx in range(len(path)):
                seg = path[idx]
                middle = c_to_np(
                    seg.point(seg.ilength(seg.length() / 2, s_tol=1e-3))
                )
                middle[1] -= 3  # slightly above the line
                # name
                drawing.add(
                    drawing.text(
                        idx,
                        insert=middle,
                        fill="rgb(44,131,68)",
                        font_size="7",
                        text_anchor="middle",
                    )
                )

    def get_svg(
        self,
        svg_filename,
        with_text=True,
        view_ids=True,
        flat=False,
        fill_panels=True,
        margin=2,
    ) -> sw.Drawing:
        """Convert pattern to writable svg representation"""

        # If we are still here, but pattern is empty, don't generate an image
        if len(self.panel_order()) == 0:
            raise core.EmptyPatternError()

        # Get svg representation per panel
        # Order by depth (=> most front panels render in front)
        # TODOLOW Even smarter way is needed for prettier allignment
        panel_order = self.panel_order()

        # MYNOTE: Extract z dimension of pattern
        panel_z = [
            self.pattern["panels"][pn]["translation"][-1] for pn in panel_order
        ]
        z_sorted_panels = [p for _, p in sorted(zip(panel_z, panel_order))]

        # Get panel paths
        paths_front, paths_back = [], []
        attributes_f, attributes_b = [], []
        names_f, names_b = [], []
        shift_x_front, shift_x_back = margin, margin
        for panel in z_sorted_panels:
            if panel is not None:
                path, attr, front = self._draw_a_panel(
                    panel, apply_transform=not flat, fill=fill_panels
                )
                if flat:
                    path = path.translated(
                        list_to_c(
                            [shift_x_front if front else shift_x_back, 0]
                        )
                    )
                    bbox = path.bbox()
                    diff = (bbox[1] - bbox[0]) + margin
                    if front:
                        shift_x_front += diff
                    else:
                        shift_x_back += diff
                if front:
                    paths_front.append(path)
                    attributes_f.append(attr)
                    names_f.append(panel)
                else:
                    paths_back.append(path)
                    attributes_b.append(attr)
                    names_b.append(panel)

        # Shift back panels if both front and back exist
        if len(paths_front) > 0 and len(paths_back) > 0:
            front_max_x = max([path.bbox()[1] for path in paths_front])
            back_min_x = min([path.bbox()[0] for path in paths_back])
            shift_x = front_max_x - back_min_x + 10  # A little spacing
            if flat:
                front_max_y = max([path.bbox()[3] for path in paths_front])
                back_min_y = min([path.bbox()[2] for path in paths_back])
                shift_y = front_max_y - back_min_y + 10  # A little spacing
                shift_x = 0
            else:
                shift_y = 0
            paths_back = [
                path.translated(list_to_c([shift_x, shift_y]))
                for path in paths_back
            ]

        # SVG convert
        paths = paths_front + paths_back
        arrdims = np.array([path.bbox() for path in paths])
        dims = np.max(arrdims[:, 1]) - np.min(arrdims[:, 0]), np.max(
            arrdims[:, 3]
        ) - np.min(arrdims[:, 2])

        viewbox = (
            np.min(arrdims[:, 0]) - margin,
            np.min(arrdims[:, 2]) - margin,
            dims[0] + 2 * margin,
            dims[1] + 2 * margin,
        )

        # Pattern info for correct placement
        self.svg_bbox = [
            np.min(arrdims[:, 0]),
            np.max(arrdims[:, 1]),
            np.min(arrdims[:, 2]),
            np.max(arrdims[:, 3]),
        ]
        self.svg_bbox_size = [viewbox[2], viewbox[3]]

        # Save
        attributes = attributes_f + attributes_b

        dwg = svgpath.wsvg(
            paths,
            attributes=attributes,
            margin_size=0,
            filename=svg_filename,
            viewbox=viewbox,
            dimensions=[str(viewbox[2]) + "cm", str(viewbox[3]) + "cm"],
            paths2Drawing=True,
        )

        # text annotations
        panel_names = names_f + names_b
        if with_text or view_ids:
            for i, panel in enumerate(panel_names):
                if panel is not None:
                    self._add_panel_annotations(
                        dwg, panel, paths[i], with_text, view_ids
                    )

        return dwg

    def _save_as_image(
        self,
        svg_filename,
        png_filename,
        with_text=True,
        view_ids=True,
        margin=2,
    ):
        """
        Saves current pattern in svg and png format for visualization

        * with_text: include panel names
        * view_ids: include ids of vertices and edges in the output image
        * margin: small amount of free space around the svg drawing (to correctly display the line width)

        """

        dwg = self.get_svg(
            svg_filename,
            with_text=with_text,
            view_ids=view_ids,
            flat=False,
            margin=margin,
        )

        dwg.save(pretty=True)

        # to png
        # NOTE: Assuming the pattern uses cm
        # 3 px == 1 cm
        # DPI = 96 (default) px/inch == 96/2.54 px/cm
        cairosvg.svg2png(
            url=svg_filename,
            write_to=png_filename,
            dpi=2.54 * self.px_per_unit,
        )

    def _save_as_image_3D(self, png_filename):
        """Save the patterns with 3D positioning using matplotlib visualization"""

        # NOTE: this routine is mostly needed for debugging

        fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
        ax = fig.add_subplot(projection="3d")

        # TODOLOW Support arcs / curves (use linearization)
        for panel in self.pattern["panels"]:
            p = self.pattern["panels"][panel]
            rot = p["rotation"]
            tr = p["translation"]
            verts_2d = p["vertices"]

            verts_to_plot = copy(verts_2d)
            verts_to_plot.append(verts_to_plot[0])

            verts3d = np.vstack(
                tuple([self._point_in_3D(v, rot, tr) for v in verts_to_plot])
            )
            x = np.squeeze(np.asarray(verts3d[:, 0]))
            y = np.squeeze(np.asarray(verts3d[:, 1]))
            z = np.squeeze(np.asarray(verts3d[:, 2]))

            ax.plot(x, y, z)

        ax.view_init(elev=115, azim=-59, roll=30)
        ax.set_aspect("equal")
        fig.savefig(png_filename, dpi=300, transparent=False)

        plt.close(fig)  # Cleanup

    def _save_as_pdf(
        self,
        svg_filename,
        pdf_filename,
        with_text=True,
        view_ids=True,
        margin=2,
    ):
        """Save a pattern as a pdf with non-overlapping panels and no filling
        Suitable for printing
        """
        dwg = self.get_svg(
            svg_filename,
            with_text=with_text,
            view_ids=view_ids,
            flat=True,
            fill_panels=False,
            margin=margin,
        )
        dwg.save(pretty=True)

        # to pdf
        # NOTE: Assuming the pattern uses cm
        # 3 px == 1 cm
        # DPI = 96 (default) px/inch == 96/2.54 px/cm
        cairosvg.svg2pdf(
            url=svg_filename,
            write_to=pdf_filename,
            dpi=2.54 * self.px_per_unit,
        )


class RandomPattern(VisPattern):
    """
    Parameter randomization of a pattern template in custom JSON format.
    Input:
        * Pattern template in custom JSON format
    Output representations:
        * Pattern instance in custom JSON format
            (with updated parameter values and vertex positions)
        * SVG (stitching info is lost)
        * PNG for visualization

    Implementation limitations:
        * Parameter randomization is only performed once on loading
        * Only accepts unchanged template files (all parameter values = 1)
        otherwise, parameter values will go out of control and outside of the original range
        (with no way to recognise it)
    """

    # ------------ Interface -------------
    def __init__(self, template_file):
        """Note that this class requires some input file:
        there is not point of creating this object with empty pattern"""
        super().__init__(
            template_file, view_ids=False
        )  # don't show ids for datasets

        # update name for a random pattern
        self.name = self.name + "_" + self._id_generator()

        # randomization setup
        self._randomize_pattern()

    # -------- Other Utils ---------
    def _id_generator(
        self, size=10, chars=string.ascii_uppercase + string.digits
    ):
        """Generated a random string of a given size, see
        https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
        """
        return "".join(random.choices(chars, k=size))
