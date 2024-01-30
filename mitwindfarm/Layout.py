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
    def __init__(
        self,
        Sx: float,
        Sy: float,
        Nx: int,
        Ny: int,
        offset: float = 0,
        shape: Literal["stag", "trap"] = "stag",
    ):
        """
        Initialize a grid layout for turbines.

        Parameters:
        - Sx (float): Streamwise spacing between turbines.
        - Sy (float): Spanwise spacing between turbines.
        - Nx (int): Number of turbines in the streamwise direction.
        - Ny (int): Number of turbines in the spanwise direction.
        - offset (float, optional): Y offset. 0.0 is no offset, 1.0 is fully offset layout.
          Default is 0.
        - shape (Literal["stag", "trap"], optional): Grid shape. Either staggered ("stag") or
          trapezoidal ("trap"). Default is "stag".

        Attributes:
        - Sx (float): Streamwise spacing between turbines.
        - Sy (float): Spanwise spacing between turbines.
        - Nx (int): Number of turbines in the streamwise direction.
        - Ny (int): Number of turbines in the spanwise direction.
        - offset (float): Y offset for the layout.
        - xs (numpy.ndarray): Flattened array of turbine x-coordinates.
        - ys (numpy.ndarray): Flattened array of turbine y-coordinates.

        Raises:
        - ValueError: If the specified shape is not "stag" or "trap".

        This class generates a grid layout of turbines based on the provided parameters.
        The layout can be either staggered ("stag") or trapezoidal ("trap"), and turbines
        can be offset in the y-direction.

        Example:
        ```python
        grid_layout = GridLayout(Sx=2.0, Sy=1.5, Nx=4, Ny=3, offset=0.5, shape="trap")
        ```
        """
        Sx, Sy = float(Sx), float(Sy)
        self.Sx = Sx
        self.Sy = Sy
        self.Nx = Nx
        self.Ny = Ny
        self.offset = offset

        x = Sx * np.arange(0, Nx)
        y = Sy * np.arange(0, Ny)
        xmesh, ymesh = np.meshgrid(x, y)
        if shape == "trap":
            y_offset = 0.5 * Sy * offset * np.arange(0, Nx)

        elif shape == "stag":
            y_offset = 0.5 * Sy * offset * np.array([x % 2 for x in range(Nx)])
        else:
            raise ValueError(f"shape {shape} not defined.")
        ymesh += y_offset

        xs = xmesh.flatten()
        ys = ymesh.flatten()
        super().__init__(xs, ys)


class Square(GridLayout):
    def __init__(self, S: float, N: int):
        super().__init__(S, S, N, N, offset=0)
