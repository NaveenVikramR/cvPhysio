"""Test for src.feature_extraction.compute_angle() with the depth (z)
coordinate dropped -- added for the elbow-accuracy branch's read-only
analysis (scripts/compare_elbow_variants.py), which compares elbow
angle computed with full 3D world landmarks against the same angle
with z forced to 0. compute_angle() itself is NOT modified or
reimplemented anywhere -- these tests only confirm its existing,
already-general (x, y, z) formula behaves exactly like a standard 2D
angle calculation when z happens to be the same constant (0) for all
three points, and that genuinely dropping a non-zero z changes the
result -- which is the whole premise the comparison script tests on
real data.
"""

import math

from src.feature_extraction import compute_angle


def independent_2d_angle(a, b, c) -> float:
    """A second, independent implementation (via atan2 of each vector's
    polar angle) of "the angle at b between b->a and b->c" in 2D, used
    only to cross-check compute_angle()'s behavior when z=0 -- not a
    reimplementation this project depends on anywhere else."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    angle_a = math.atan2(ay, ax)
    angle_c = math.atan2(cy, cx)
    diff = abs(math.degrees(angle_a - angle_c)) % 360
    return 360 - diff if diff > 180 else diff


def test_compute_angle_with_z_zero_matches_independent_2d_formula():
    """A right angle laid out in the x/y plane, z=0 for all three
    points -- compute_angle() must agree with an independently
    implemented 2D angle formula, not just with itself."""
    a = (1.0, 0.0, 0.0, 1.0)
    b = (0.0, 0.0, 0.0, 1.0)
    c = (0.0, 1.0, 0.0, 1.0)
    assert abs(compute_angle(a, b, c) - independent_2d_angle(a, b, c)) < 1e-9
    assert abs(compute_angle(a, b, c) - 90.0) < 1e-9


def test_compute_angle_with_z_zero_matches_independent_2d_formula_for_an_oblique_triangle():
    """A non-right-angle, non-axis-aligned case, still z=0 -- rules out
    the right-angle test above being a coincidence of symmetric axes."""
    a = (3.0, 1.0, 0.0, 1.0)
    b = (0.5, -0.5, 0.0, 1.0)
    c = (-1.0, 2.0, 0.0, 1.0)
    assert abs(compute_angle(a, b, c) - independent_2d_angle(a, b, c)) < 1e-9


def test_dropping_a_nonzero_z_changes_the_angle():
    """The premise scripts/compare_elbow_variants.py tests on real data:
    the same (a, b, c) triple gives a DIFFERENT, still well-defined
    angle once a genuinely non-zero z is forced to 0 -- z-dropping is
    not a no-op in general. b->a=(2,0,1), b->c=(1,1,0): the 3D angle
    (z kept) and the 2D angle (z dropped) are both finite, neither
    degenerate (0/180/NaN), and they differ from each other.
    """
    a_3d = (2.0, 0.0, 1.0, 1.0)
    b_3d = (0.0, 0.0, 0.0, 1.0)
    c_3d = (1.0, 1.0, 0.0, 1.0)
    angle_3d = compute_angle(a_3d, b_3d, c_3d)

    a_xy = (a_3d[0], a_3d[1], 0.0, a_3d[3])
    b_xy = (b_3d[0], b_3d[1], 0.0, b_3d[3])
    c_xy = (c_3d[0], c_3d[1], 0.0, c_3d[3])
    angle_xy = compute_angle(a_xy, b_xy, c_xy)

    assert abs(angle_3d - 50.768) < 1e-2
    assert abs(angle_xy - 45.0) < 1e-9
    assert abs(angle_3d - angle_xy) > 1.0  # genuinely different, not a rounding artifact


def test_dropping_z_onto_a_shared_point_can_produce_nan():
    """A different, degenerate way z-dropping can change the result:
    two points that differ only in z become the SAME point once z is
    dropped, turning a well-defined 3D angle into an undefined (NaN)
    one -- compute_angle()'s existing coincident-point guard, exercised
    here via z-dropping specifically (not a new behavior, see
    tests/test_feature_extraction.py for the guard's own direct tests)."""
    a = (0.0, 0.0, 0.0, 1.0)
    b = (0.0, 0.0, 1.0, 1.0)
    c = (0.0, 0.0, 2.0, 1.0)
    assert abs(compute_angle(a, b, c) - 180.0) < 1e-9  # collinear along z in 3D

    a_xy = (a[0], a[1], 0.0, a[3])
    b_xy = (b[0], b[1], 0.0, b[3])
    c_xy = (c[0], c[1], 0.0, c[3])
    assert math.isnan(compute_angle(a_xy, b_xy, c_xy))  # a, b, c all collapse to (0, 0) -- zero-length vectors
