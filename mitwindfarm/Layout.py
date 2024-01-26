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
        self,
        angle: float,
        units: Literal["deg", "rad"] = "deg",
        center: Literal["origin", "centroid"] = "centroid",
    ) -> "Layout":
        """
        Rotate the wind farm layout clockwise about origin or centroid by an angle (in radians).

        Parameters:
        - angle: The angle of rotation.
        - units: the units of the angle of rotation (default is 'deg')
        - center: Center of rotation, can be 'origin' or 'centroid' (default is 'centroid').

        Returns:
        A new Layout instance after rotation.
        """
        if units == "deg":
            angle = np.deg2rad(angle)
        elif units != "rad":
            raise ValueError("units is not 'deg' or 'rad'")

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

    def __len__(self):
        return len(self.x)

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
            yield i, (self.x[i], self.y[i], self.z[i])


class GridLayout(Layout):
    def __init__(self, Sx: float, Sy: float, Nx: int, Ny: int, offset: float = 0):
        """
        Sx, Sy: streamwise and spanwise spacing respectively
        Nx, Ny: streamwise and spanwise number of turbines respectively
        offset: 0.0 is no offset, 1.0 is fully offset layout
        """
        self.Sx = Sx
        self.Sy = Sy
        self.Nx = Nx
        self.Ny = Ny
        self.offset = offset
        # Define grid layout
        xdim = np.linspace(0, (Nx - 1) * Sx, Nx)
        ydim = np.linspace(0, (Ny - 1) * Sy, Ny)
        xs = np.array([])
        ys = np.array([])
        for i, x in enumerate(xdim):
            curr_off = 0.5 * offset * Sy * i
            for j, y in enumerate(ydim):
                xs = np.append(xs, [x])
                ys = np.append(ys, [y + curr_off])

        super().__init__(xs, ys)
class Square(GridLayout):
    def __init__(self, S: float, N: int):
        super().__init__(S, S, N, N, offset=0)
