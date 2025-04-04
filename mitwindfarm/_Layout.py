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
        self.x = np.array(xs) if isinstance(xs, list) else xs
        self.y = np.array(ys) if isinstance(ys, list) else ys

        if zs is None:
            self.z = np.zeros_like(self.x)
        else:
            self.z = zs

        # Calculate the centroid for later use
        self.x_centroid = np.mean(self.x)
        self.y_centroid = np.mean(self.y)

    def __repr__(self):
        x_repr = np.array2string(self.x, separator=", ")
        y_repr = np.array2string(self.y, separator=", ")
        return f"Layout(x={x_repr}, y={y_repr})"

    def rotate(
        self,
        angle: float,
        units: Literal["deg", "rad"] = "deg",
        center: Literal["origin", "centroid"] = "centroid",
    ) -> "Layout":
        """
        Rotate the wind farm layout clockwise about origin or centroid by an angle in degrees or radians.

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
            x_0 = 0
            y_0 = 0
        elif center == "centroid":
            x_0 = self.x_centroid
            y_0 = self.y_centroid

        x_new = (self.x - x_0) * np.cos(angle) + (self.y - y_0) * (-np.sin(angle)) + x_0
        y_new = (self.x - x_0) * np.sin(angle) + (self.y - y_0) * (np.cos(angle)) + y_0

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
        self.Sx = Sx
        self.Sy = Sy
        self.Nx = Nx
        self.Ny = Ny
        
        if offset < 0 or offset > 1:
            raise ValueError(f"offset ({offset} should be between 0 and 1.)")
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
