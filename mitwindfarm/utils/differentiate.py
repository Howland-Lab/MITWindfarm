"""
Differentiation module

Kirby Heck
2025 May 05
"""

import numpy as np
import warnings


def second_der(
    arr,
    dxi=None,
    axis=-1,
):
    """
    Computes the second derivative using second-order finite differences.

    Uses central differences for the interior and second-order forward/backward
    differences for the boundaries.

    [Written by Gemini 2.5 Pro]

    Args:
        arr (np.ndarray): Input array.
        axis (int): The axis along which the derivative is taken. Default is -1.
        dxi (float or array-like, optional): Spacing between points.
            If None, spacing is assumed to be 1.0.
            If a single float, it's used for the specified axis.
            If an array-like, its length must be 1 or match arr.ndim. If it
            matches arr.ndim, the spacing for the specified axis is used.

    Returns:
        np.ndarray: The computed second derivative, same shape as arr.

    Raises:
        ValueError: If dxi is array-like and its length doesn't match arr.ndim.
        ValueError: If the array size along the specified axis is less than 2.
                    (Second-order boundaries require >= 4 points).
    """
    arr = np.asarray(arr)
    ndim = arr.ndim
    # Ensure axis is positive for easier handling
    positive_axis = axis if axis >= 0 else ndim + axis
    if not (0 <= positive_axis < ndim):
        raise np.AxisError(f"Invalid axis {axis} for array with {ndim} dimensions.")

    n = arr.shape[positive_axis]

    if dxi is None:
        dx = 1.0
    else:
        dxi = np.atleast_1d(dxi)
        if len(dxi) == 1:
            dx = float(dxi[0])  # Ensure float division
        elif len(dxi) == ndim:
            dx = float(dxi[positive_axis])
        else:
            raise ValueError(
                f"Length of dxi ({len(dxi)}) must be 1 or match the number of dimensions in arr ({ndim})."
            )

    if dx == 0:
        raise ValueError("Spacing dx cannot be zero.")

    dx_sq = dx * dx

    # Initialize the result array
    d2f_dx2 = np.zeros_like(arr, dtype=np.result_type(arr, dx))  # Match type

    # Handle edge cases for small arrays where second-order boundaries are not possible
    if n < 2:
        raise ValueError(
            f"Array size {n} along axis {positive_axis} is too small to compute second derivative."
        )
    elif n == 2:
        # Cannot compute 2nd derivative meaningfully with only 2 points using these methods.
        # np.gradient would give 0 for the 2nd derivative. Let's return 0.
        warnings.warn(
            f"Array size {n} along axis {positive_axis} is small; returning zeros for second derivative.",
            stacklevel=2,
        )
        return d2f_dx2  # Already zeros
    elif n == 3:
        # Can only compute central difference for the middle point
        # Boundaries are problematic for 2nd order.
        # Let's compute the middle point and leave boundaries as 0, with a warning.
        warnings.warn(
            f"Array size {n} along axis {positive_axis} is small; using central difference for interior point only, boundaries set to zero.",
            stacklevel=2,
        )
        f_im1 = np.take(arr, 0, axis=positive_axis)
        f_i = np.take(arr, 1, axis=positive_axis)
        f_ip1 = np.take(arr, 2, axis=positive_axis)
        center_val = (f_ip1 - 2 * f_i + f_im1) / dx_sq
        # Place the result in the correct slice of the output array
        result_slice = [slice(None)] * ndim
        result_slice[positive_axis] = 1
        d2f_dx2[tuple(result_slice)] = center_val
        return d2f_dx2

    # --- Main computation for n >= 4 ---

    # Interior points (central difference)
    f_im1 = np.take(arr, np.arange(0, n - 2), axis=positive_axis)
    f_i = np.take(arr, np.arange(1, n - 1), axis=positive_axis)
    f_ip1 = np.take(arr, np.arange(2, n - 0), axis=positive_axis)
    interior_val = (f_ip1 - 2 * f_i + f_im1) / dx_sq

    # Place the result in the correct slice of the output array
    result_slice_interior = [slice(None)] * ndim
    result_slice_interior[positive_axis] = slice(1, n - 1)
    d2f_dx2[tuple(result_slice_interior)] = interior_val

    # Boundary point 0 (forward difference O(dx^2))
    # [2*f0 - 5*f1 + 4*f2 - f3] / dx^2
    f0 = np.take(arr, 0, axis=positive_axis)
    f1 = np.take(arr, 1, axis=positive_axis)
    f2 = np.take(arr, 2, axis=positive_axis)
    f3 = np.take(arr, 3, axis=positive_axis)
    start_val = (2 * f0 - 5 * f1 + 4 * f2 - f3) / dx_sq

    result_slice_start = [slice(None)] * ndim
    result_slice_start[positive_axis] = 0
    d2f_dx2[tuple(result_slice_start)] = start_val

    # Boundary point N-1 (backward difference O(dx^2))
    # [2*f_{N-1} - 5*f_{N-2} + 4*f_{N-3} - f_{N-4}] / dx^2
    f_n1 = np.take(arr, n - 1, axis=positive_axis)
    f_n2 = np.take(arr, n - 2, axis=positive_axis)
    f_n3 = np.take(arr, n - 3, axis=positive_axis)
    f_n4 = np.take(arr, n - 4, axis=positive_axis)
    end_val = (2 * f_n1 - 5 * f_n2 + 4 * f_n3 - f_n4) / dx_sq

    result_slice_end = [slice(None)] * ndim
    result_slice_end[positive_axis] = n - 1
    d2f_dx2[tuple(result_slice_end)] = end_val

    return d2f_dx2


if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    y = np.sin(x) * x
    d2y_analytical = 2 * np.cos(x) - x * np.sin(x)
    d2y = second_der(y, dxi=np.diff(x)[0], axis=0)
    d2y_numpy = np.gradient(np.gradient(y, x), x)
    print("Error in numerical vs analytical second derivative:")
    print(np.max(np.abs(d2y - d2y_analytical)))
    print("Numpy second derivative error:")
    print(np.max(np.abs(d2y_numpy - d2y_analytical)))
