from copy import deepcopy

import numpy as np
import pygarment as pyg
from assets.garment_programs import bands
from scipy.spatial.transform import Rotation as R


# ------  Armhole shapes ------
def ArmholeSquare(incl, width, angle, invert=True, **kwargs):
    """Simple square armhole cut-out
    Not recommended to use for sleeves, stitching in 3D might be hard

    if angle is provided, it also calculated the shape of the sleeve interface to attach

    returns edge sequence and part to be preserved  inverted
    """

    edges = pyg.EdgeSeqFactory.from_verts([0, 0], [incl, 0], [incl, width])
    if not invert:
        return edges, None

    sina, cosa = np.sin(angle), np.cos(angle)
    l = edges[0].length()
    sleeve_edges = pyg.EdgeSeqFactory.from_verts(
        [incl + l * sina, -l * cosa], [incl, 0], [incl, width]
    )

    # TODOLOW Bend instead of rotating to avoid sharp connection
    sleeve_edges.rotate(angle=-angle)

    return edges, sleeve_edges


def ArmholeAngle(
    incl, width, angle, incl_coeff=0.2, w_coeff=0.2, invert=True, **kwargs
):
    """Piece-wise smooth armhole shape"""
    diff_incl = incl * (1 - incl_coeff)
    edges = pyg.EdgeSeqFactory.from_verts(
        [0, 0], [diff_incl, w_coeff * width], [incl, width]
    )
    if not invert:
        return edges, None

    sina, cosa = np.sin(angle), np.cos(angle)
    l = edges[0].length()
    sleeve_edges = pyg.EdgeSeqFactory.from_verts(
        [diff_incl + l * sina, w_coeff * width - l * cosa],
        [diff_incl, w_coeff * width],
        [incl, width],
    )
    # TODOLOW Bend instead of rotating to avoid sharp connection
    sleeve_edges.rotate(angle=-angle)

    return edges, sleeve_edges


def ArmholeCurve(
    incl,
    width,
    angle,
    bottom_angle_mix=0,
    invert=True,
    verbose=False,
    **kwargs,
):
    """Classic sleeve opening on Cubic Bezier curves"""
    # Curvature as parameters?
    cps = [[0.5, 0.2], [0.8, 0.35]]
    edge = pyg.CurveEdge([incl, width], [0, 0], cps)
    edge_as_seq = pyg.EdgeSequence(edge.reverse())

    if not invert:
        return edge_as_seq, None

    # Initialize inverse (initial guess)
    # Agle == 0
    down_direction = np.array([0, -1])  # Full opening is vertically aligned
    inv_cps = deepcopy(cps)
    inv_cps[-1][1] *= -1  # Invert the last
    inv_edge = pyg.CurveEdge(
        start=[incl, width],
        end=(
            np.array([incl, width]) + down_direction * edge._straight_len()
        ).tolist(),
        control_points=inv_cps,
    )

    # Rotate by desired angle (usually desired sleeve rest angle)
    inv_edge.rotate(angle=-angle)

    # Optimize the inverse shape to be nice
    shortcut = inv_edge.shortcut()
    rotated_direction = shortcut[-1] - shortcut[0]
    rotated_direction /= np.linalg.norm(rotated_direction)
    left_direction = np.array([-1, 0])
    mix_factor = bottom_angle_mix

    dir = (1 - mix_factor) * rotated_direction + (
        mix_factor * down_direction
        if mix_factor > 0
        else (-mix_factor * left_direction)
    )

    # TODOLOW Remember relative curvature results and reuse them? (speed)
    fin_inv_edge = pyg.ops.curve_match_tangents(
        inv_edge.as_curve(),
        down_direction,  # Full opening is vertically aligned
        dir,
        target_len=edge.length(),
        return_as_edge=True,
        verbose=verbose,
    )

    return edge_as_seq, pyg.EdgeSequence(fin_inv_edge.reverse())


# -------- New sleeve definitions -------


class SleevePanel(pyg.Panel):
    """Trying proper sleeve panel"""

    def __init__(
        self,
        name,
        body,
        design,
        open_shape,
        length_shift=0,
        _standing_margin=5,
    ):
        """Define a standard sleeve panel (half a sleeve)
        * length_shift -- force upd sleeve length by this amount.
            Can be used to adjust length evaluation to fit the cuff
        """
        super().__init__(name)
        MIN_LENGTH = 5  # Minimum sleeve length

        shoulder_angle = np.deg2rad(body["_shoulder_incl"])
        rest_angle = max(
            np.deg2rad(design["sleeve_angle"]["v"]), shoulder_angle
        )
        standing = design["standing_shoulder"]["v"]

        # Calculating extension size & end size before applying ruffles
        # Since ruffles add to pattern length & width, but not to de-facto
        # sleeve length in 3D
        end_width = design["end_width"]["v"] * abs(
            open_shape[0].start[1] - open_shape[-1].end[1]
        )
        # Ensure it fits regardless of parameters
        end_width = max(end_width, body["wrist"] / 2)

        # Ruffles at opening
        if not pyg.utils.close_enough(design["connect_ruffle"]["v"], 1):
            open_shape.extend(design["connect_ruffle"]["v"])

        # -- Main body of a sleeve --
        opening_length = abs(open_shape[0].start[0] - open_shape[-1].end[0])
        arm_width = abs(open_shape[0].start[1] - open_shape[-1].end[1])
        # Length from the border of the opening to the end of the sleeve
        length = design["length"]["v"] * (body["arm_length"] - opening_length)
        # NOTE: Asked to reduce by too much: reduce as much as possible
        length = max(length + length_shift, MIN_LENGTH)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0], [0, -end_width], [length, -arm_width]
        )

        # Align the opening
        open_shape.snap_to(self.edges[-1].end)
        open_shape[0].start = self.edges[-1].end  # chain
        self.edges.append(open_shape)
        # Fin
        self.edges.close_loop()

        if standing:
            # Add a "shelve" to create square shoulder appearance
            if rest_angle > (shoulder_angle + np.deg2rad(_standing_margin)):
                top_edge = self.edges[-1]
                start = top_edge.start
                len = design["standing_shoulder_len"]["v"]

                x_shift = len * np.cos(rest_angle - shoulder_angle)
                y_shift = len * np.sin(rest_angle - shoulder_angle)

                standing_edge = pyg.Edge(
                    start=start, end=[start[0] - x_shift, start[1] + y_shift]
                )
                top_edge.start = standing_edge.end

                self.edges.substitute(top_edge, [standing_edge, top_edge])
            else:
                if self.verbose:
                    print(
                        f"{self.__class__.__name__}::WARNING::"
                        f"Sleeve rest angle {np.rad2deg(rest_angle):.3f} should be "
                        f'larger than shoulder angle {body["_shoulder_incl"]} by '
                        f"at least {_standing_margin} deg to enable "
                        "standing shoulder. Standing shoulder ignored"
                    )
                standing = False

        # Interfaces
        self.interfaces = {
            # NOTE: interface needs reversing because the open_shape was reversed for construction
            "in": pyg.Interface(
                self, open_shape, ruffle=design["connect_ruffle"]["v"]
            ),
            "out": pyg.Interface(
                self, self.edges[0], ruffle=design["cuff"]["top_ruffle"]["v"]
            ),
            "top": pyg.Interface(
                self, self.edges[-2:] if standing else self.edges[-1]
            ),
            "bottom": pyg.Interface(self, self.edges[1]),
        }

        # Default placement
        self.set_pivot(self.edges[-1].start)
        self.translate_to(
            [
                -body["shoulder_w"] / 2,
                body["height"] - body["head_l"],
                0,
            ]
        )

    def length(self, longest_dim=False):
        return self.interfaces["bottom"].edges.length()


class Sleeve(pyg.Component):
    """Trying to do a proper sleeve"""

    def __init__(self, tag, body, design, front_w, back_w):
        """Defintion of a sleeve:
        * front_w, back_w: the width front and the back of the top
        the sleeve will attach to -- needed for correct share calculations
            They may be
            * Specified as scalar numbers
            * Specified as functions w.r.t. the requested vertical level (=>
                calculated width of a horizontal slice)
        """
        super().__init__(f"{self.__class__.__name__}_{tag}")

        design = design["sleeve"]
        self.design = design
        self.body = body

        sleeve_balance = body["_base_sleeve_balance"] / 2

        rest_angle = max(
            np.deg2rad(design["sleeve_angle"]["v"]),
            np.deg2rad(body["_shoulder_incl"]),
        )

        connecting_width = design["connecting_width"]["v"]
        smoothing_coeff = design["smoothing_coeff"]["v"]

        front_w = front_w(connecting_width) if callable(front_w) else front_w
        back_w = back_w(connecting_width) if callable(back_w) else back_w

        # --- Define sleeve opening shapes ----
        # NOTE: Non-trad armholes only for sleeveless styles due to
        # unclear inversion and stitching errors (see below)
        armhole = (
            globals()[design["armhole_shape"]["v"]]
            if design["sleeveless"]["v"]
            else ArmholeCurve
        )
        front_project, front_opening = armhole(
            front_w - sleeve_balance,
            connecting_width,
            angle=rest_angle,
            incl_coeff=smoothing_coeff,
            w_coeff=smoothing_coeff,
            invert=not design["sleeveless"]["v"],
            bottom_angle_mix=design["opening_dir_mix"]["v"],
            verbose=self.verbose,
        )
        back_project, back_opening = armhole(
            back_w - sleeve_balance,
            connecting_width,
            angle=rest_angle,
            incl_coeff=smoothing_coeff,
            w_coeff=smoothing_coeff,
            invert=not design["sleeveless"]["v"],
            bottom_angle_mix=design["opening_dir_mix"]["v"],
        )

        self.interfaces = {
            "in_front_shape": pyg.Interface(self, front_project),
            "in_back_shape": pyg.Interface(self, back_project),
        }

        if design["sleeveless"]["v"]:
            # The rest is not needed!
            return

        if front_w != back_w:
            front_opening, back_opening = pyg.ops.even_armhole_openings(
                front_opening,
                back_opening,
                tol=0.2
                / front_opening.length(),  # ~2mm tolerance as a fraction of length
                verbose=self.verbose,
            )

        # --- Eval length adjustment for cuffs (if any) ----
        cuff_len_adj = self._cuff_len_adj()

        # ----- Get sleeve panels -------
        self.f_sleeve = SleevePanel(
            f"{tag}_sleeve_f",
            body,
            design,
            front_opening,
            length_shift=-cuff_len_adj,
        ).translate_by([0, 0, 15])
        self.b_sleeve = SleevePanel(
            f"{tag}_sleeve_b",
            body,
            design,
            back_opening,
            length_shift=-cuff_len_adj,
        ).translate_by([0, 0, -15])

        # Connect panels
        self.stitching_rules = pyg.Stitches(
            (self.f_sleeve.interfaces["top"], self.b_sleeve.interfaces["top"]),
            (
                self.f_sleeve.interfaces["bottom"],
                self.b_sleeve.interfaces["bottom"],
            ),
        )

        # Interfaces
        self.interfaces.update(
            {
                "in": pyg.Interface.from_multiple(
                    self.f_sleeve.interfaces["in"],
                    self.b_sleeve.interfaces["in"].reverse(
                        with_edge_dir_reverse=True
                    ),
                ),
                "out": pyg.Interface.from_multiple(
                    self.f_sleeve.interfaces["out"],
                    self.b_sleeve.interfaces["out"],
                ),
            }
        )

        # Cuff
        if design["cuff"]["type"]["v"]:
            # Class
            # Copy to avoid editing original design dict
            cdesign = deepcopy(design)
            cuff_circ = (
                self.interfaces["out"].edges.length()
                / design["cuff"]["top_ruffle"]["v"]
            )
            # Ensure it fits regardless of parameters
            cuff_circ = max(cuff_circ, body["wrist"])
            cdesign["cuff"]["b_width"] = dict(v=cuff_circ)
            cdesign["cuff"]["cuff_len"]["v"] = cuff_len_adj

            cuff_class = getattr(bands, cdesign["cuff"]["type"]["v"])
            self.cuff = cuff_class(f"sl_{tag}", cdesign)

            # Position
            self.cuff.rotate_by(
                R.from_euler(
                    "XYZ", [0, 0, -90], degrees=True  # from -Ox direction
                )
            )
            self.cuff.place_by_interface(
                self.cuff.interfaces["top"],
                self.interfaces["out"],
                gap=2,
                alignment="top",
            )

            self.stitching_rules.append(
                (self.cuff.interfaces["top"], self.interfaces["out"])
            )

            # UPD out interface!
            self.interfaces["out"] = self.cuff.interfaces["bottom"]

        # Final rotation of sleeve piece
        self.rotate_by(
            R.from_euler("XYZ", [0, 0, body["arm_pose_angle"]], degrees=True)
        )

        # Set label
        self.set_panel_label("arm")

    def _cuff_len_adj(self):
        """Eval sleeve length adjustment due to cuffs (if any)"""
        if not self.design["cuff"]["type"]["v"]:
            return 0

        cuff_len_adj = (
            self.design["cuff"]["cuff_len"]["v"] * self.body["arm_length"]
        )
        max_len = self.design["length"]["v"] * self.body["arm_length"]
        if cuff_len_adj > max_len * 0.7:
            cuff_len_adj = max_len * 0.7

        return cuff_len_adj

    def length(self):
        if self.design["sleeveless"]["v"]:
            return 0

        if self.design["cuff"]["type"]["v"]:
            return self.f_sleeve.length() + self.cuff.length()

        return self.f_sleeve.length()
