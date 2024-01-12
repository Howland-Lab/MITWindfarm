from typing import Iterable, Literal
import numpy as np


class Layout:
    """
    Represents a wind farm layout with x, y, and optional z coordinates.
    """

    def __init__(self, xs: list[float], ys: list[float], zs: list[float] = None):
        """
        Initialize the Layout instance with x, y, and optional z coordinates.

        Parameters:
        - xs: List of x-coordinates.
        - ys: List of y-coordinates.
        - zs: List of z-coordinates (optional, default is zero for all points).
        """
        self.x = np.array(xs)
        self.y = np.array(ys)
        if zs is None:
            self.z = np.zeros_like(self.x)
        else:
            self.z = np.array(zs)

        # Calculate the centroid for later use
        self.centroid = np.vstack([self.x, self.y]).mean(axis=1).reshape([-1, 1])

    def rotate(
        self, angle: float, center: Literal["origin", "centroid"] = "centroid"
    ) -> "Layout":
        """
        Rotate the wind farm layout clockwise about origin or centroid by an angle (in radians).

        Parameters:
        - angle: The angle of rotation in radians.
        - center: Center of rotation, can be 'origin' or 'centroid' (default is 'centroid').

        Returns:
        A new Layout instance after rotation.
        """
        if center == "origin":
            X0 = np.array([[0], [0]])
        elif center == "centroid":
            X0 = self.centroid

        X = np.vstack([self.x, self.y])
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        x_new, y_new = rot_mat @ (X - X0) + X0

        return Layout(x_new, y_new, self.z)

    def __iter__(self):
        """
        Iterate over the x, y, z coordinates of the layout.
        """
        return zip(self.x, self.y, self.z)

    def iter_downstream(self) -> Iterable:
        """
        Iterate over the layout elements in a downstream order (sorted by x-coordinate).

        Returns:
        An iterable of tuples representing the sorted layout.
        """
        for i in np.argsort(self.x):
            yield self.x[i], self.y[i], self.z[i]
