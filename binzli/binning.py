# Copyright (C) 2025-2026  Fabian L. Seidler
#
# This file is part of binzli.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Utility for binning data.
"""

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class BinningScheme:
    """
    A binning schematic. Assumes continous bins.

           _____
     ____ |     |
    |     |     |          _________
    |     |     |_________|         |
    |     |     |         |         |
    i0 x0 i1 x1 i2   x2   i3   x3   i4
    <=b0=> <=b1=> <==b2==> <===b3==>

    Each `i` in the illustration above identifies an interface, each `x` a bin center,
    and each `b` a bin width.

    """
    interfaces: npt.NDArray[np.float64]
    """ Interfaces between bins """

    centers: npt.NDArray[np.float64]
    """ Center of a bin """

    bin_widths: npt.NDArray[np.float64]
    """ width of a bin (from interface to interface) """

    def __post_init__(self) -> None:
        """
        Checks if array lengths are consistent.
        """

        # TODO: implement

    def __len__(self) -> int:
        return len(self.centers)

    @classmethod
    def new_from_interfaces(
        cls,
        interfaces: npt.NDArray[np.float64],
    ) -> Self:
        """
        A binning scheme based on interfaces. NOTE: assumes that bins are continous.
             ___
         ___|   |
        |   |   |        _______
        |   |   |_______|       |
        |   |   |       |       |
        * x * x *   x   *   x   *

        Each `*` in the illustration above identifies an interface, each `x` a bin center.

        Parameters:
        -----------
        interfaces : npt.NDArray[np.float64]
            Interfaces between bins. Array must be monotonically increasing and have at least two elements.
        Returns:
        --------
            The interfaces as binned object; includes the position of bin centers.

        Raises:
        -------
        AssertionError
            If the lengths of x and y do not match. TODO: implement
        ValueError
            If `x` is not monotonically increasing. TODO: implement
        """
        centers = interfaces[:-1] + np.diff(interfaces) / 2
        bin_widths = np.diff(interfaces)

        return cls(centers=centers, interfaces=interfaces, bin_widths=bin_widths)

    @classmethod
    def new_from_centers(
        cls,
        centers: npt.NDArray[np.float64],
        edge_method: Literal["dismiss", "extrapolate", "half-bin"] = "half-bin",
    ) -> Self:
        """
        A binning scheme based on the centers of a bin. NOTE: calculates the position of
        the interfaces under the assumption that they are half-way between centers.
        NOTE: assumes that bins are continous.
             ___
         ___|   |
        |   |   |        _______
        |   |   |_______|       |
        |   |   |       |       |
        * x * x *   x   *   x   *

        Each `*` in the illustration above identifies an interface, each `x` a bin center.

        Parameters:
        -----------
        centers : npt.NDArray[np.float64]
            Bin centers. Must be monotonically increasing and have at least two elements.
        edge_method : Literal["dismiss", "extrapolate", "half-bin"], optional
            A method for handling the edges of the bins.
            - "dismiss": Excludes the edges of the bin and only includes the inner values.
            - "half-bin": Places the last interface half the distance to x_i-1 to the right; same for the leftmost bin (default).

        Returns:
        --------
        BinnedData
            A new instance of BinnedData containing the binned data, with the bin
            centers, interfaces, and corresponding values.

        Raises:
        -------
        AssertionError
            If the lengths of x and y do not match.
        ValueError
            If `x` is not monotonically increasing.
        """

        x = centers

        match edge_method:
            case "dismiss":
                interfaces = (x[:-1] + x[1:]) / 2
                centers = x[1:-1]
            case "half-bin":
                inner_interfaces = (x[:-1] + x[1:]) / 2
                first_interface = x[0] - (inner_interfaces[0] - x[0])
                last_interface = x[len(x) - 1] + (
                    x[len(x) - 1] - inner_interfaces[len(inner_interfaces) - 1]
                )
                interfaces = np.append(
                    np.append(first_interface, inner_interfaces), last_interface
                )
                centers = x
            case _:
                raise NotImplementedError()
                # TODO: implement other methods!

        bin_widths = np.diff(interfaces)

        return cls(centers=centers, interfaces=interfaces, bin_widths=bin_widths)


@dataclass(frozen=True)
class BinnedData:
    """
    Data, binned to a CONTINOUS binning scheme as in `Bins`.

           _y1__
     _y0_ |     |
    |     |     |          ___y3____
    |     |     |___y2____|         |
    |     |     |         |         |
    i0 x0 i1 x1 i2   x2   i3   x3   i4
    <=b0=> <=b1=> <==b2==> <===b3==>

    Each `i` in the illustration above identifies an interface, each `x` a bin center, each `b` a
    bin width, and each `y` a data point (value, bin height).

    """

    binning_scheme: BinningScheme
    """ Bins """
    values: npt.NDArray[np.float64]
    """ Bin heights. For each bin, there is one height in the array. """

    def __post_init__(self) -> None:
        """
        Checks if array lengths are consistent.
        """

        if len(self.binning_scheme) != len(self.values):
            raise ValueError("There must be as many values as bins!")

    def __len__(self) -> int:
        return len(self.binning_scheme)

    @property
    def centers(self) -> npt.NDArray[np.float64]:
        return self.binning_scheme.centers

    @property
    def interfaces(self) -> npt.NDArray[np.float64]:
        return self.binning_scheme.interfaces

    @property
    def bin_widths(self) -> npt.NDArray[np.float64]:
        return self.binning_scheme.bin_widths

    @classmethod
    def new_from_interfaces(
        cls,
        interfaces: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
    ) -> Self:
        """
        Creates a new instance of BinnedData from interfaces and the values in-between.
        NOTE: assumes that bins are continous.

              _y1_
         _y0_|    |
        |    |    |         ___y3___
        |    |    |___y2___|        |
        |    |    |        |        |
        i0 * i1 * i2   *   i3   *   i4

        Each `i` in the illustration above identifies an interface, each `*` a bin center,
        and each `y` a bin value.

        Parameters:
        -----------
        interfaces : npt.NDArray[np.float64]
            Interfaces between bins. Array must be monotonically increasing and have at least two elements.
        values : npt.NDArray[np.float64]
            The y-values (measurements or values) corresponding to the x-values.
            This array must have the same len(x) - 1.
        Returns:
        --------
        BinnedData
            A new instance of BinnedData containing the binned data, with the bin
            centers, interfaces, and corresponding values.

        Raises:
        -------
        AssertionError
            If the lengths of x and y do not match. NOTE: not yet implemented
        ValueError
            If `x` is not monotonically increasing. NOTE: not yet implemented
        """

        assert len(interfaces) - 1 == len(values), "TODO: descriptive error message"

        bins = BinningScheme.new_from_interfaces(interfaces)

        return cls(binning_scheme=bins, values=values)

    @classmethod
    def new_from_xy(
        cls,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        edge_method: Literal["dismiss", "extrapolate", "half-bin"] = "half-bin",
    ) -> Self:
        """
        Creates a new instance of BinnedData from two equally long arrays, x and y.
        NOTE: assumes that bins are continous.

              _y1_
         _y0_|    |
        |    |    |         ___y3___
        |    |    |___y2___|        |
        |    |    |        |        |
        * x0 * x1 *   x2   *   x3   *

        Each `*` in the illustration above identifies an interface, each `x` a bin center,
        and each `y` a bin value.

        Parameters:
        -----------
        x : npt.NDArray[np.float64]
            Bin centers. Must be monotonically increasing and have at least two elements.
        y : npt.NDArray[np.float64]
            The y-values (measurements or values) corresponding to the x-values.
            This array must have the same length as x.
        edge_method : Literal["dismiss", "extrapolate", "half-bin"], optional
            A method for handling the edges of the bins:
            - "dismiss": Excludes the edges of the bin and only includes the inner values.
            - "half-bin": Places the last interface half the distance to x_i-1 to the right; same for the leftmost bin (default).

        Returns:
        --------
        BinnedData
            A new instance of BinnedData containing the binned data, with the bin
            centers, interfaces, and corresponding values.

        Raises:
        -------
        AssertionError
            If the lengths of x and y do not match. NOTE: not yet implemented
        ValueError
            If `x` is not monotonically increasing. NOTE: not yet implemented
        """

        assert len(y) == len(x), "TODO: descriptive error message"

        match edge_method:
            case "dismiss":
                interfaces = (x[:-1] + x[1:]) / 2
                centers = x[1:-1]
                values = y[1:-1]
            case "half-bin":
                inner_interfaces = (x[:-1] + x[1:]) / 2
                first_interface = x[0] - (inner_interfaces[0] - x[0])
                last_interface = x[len(x) - 1] + (
                    x[len(x) - 1] - inner_interfaces[len(inner_interfaces) - 1]
                )
                interfaces = np.append(
                    np.append(first_interface, inner_interfaces), last_interface
                )
                centers = x
                values = y
            case _:
                raise NotImplementedError()
                # TODO: implement other methods!

        bin_widths = np.diff(interfaces)

        bins = BinningScheme(interfaces=interfaces, centers=centers, bin_widths=bin_widths)

        return cls(binning_scheme=bins, values=values)

    def rebin_to_scheme(self, scheme: BinningScheme | Self) -> Self:
        """
        Rebins the current `BinnetData` to a new target binning (of either type `Bins` or `BinnedData`)

        Parameters
        ----------
            target : Union[Bins, BinnedData]
                `Bins` or `BinnedData` to which the current data should be binned.
        """

        # Generate intermediate binnings, which as an overlap over the bins before
        intermediate_interfaces = np.sort(
            np.unique(np.concatenate([self.interfaces, scheme.interfaces]))
        )

        # interpolate values to the (temporary) stacked building scheme
        interp = interp1d(
            self.centers, self.values, bounds_error=False, fill_value=0.0
        )
        intermediate_values = interp(intermediate_interfaces)

        # integrate each bin to generate the averaged value of sub-bins
        target_interfaces = np.sort(scheme.interfaces)
        target_values = trapz_numba(
            target_interfaces, intermediate_interfaces, intermediate_values
        )

        return type(self)(
            binning_scheme=BinningScheme.new_from_interfaces(target_interfaces),
            values=target_values / np.diff(target_interfaces),
        )


@njit
def trapz_numba(target_interfaces, intermediate_interfaces, intermediate_values):
    """
    Calculates the re-binned bin values from two overlapping binning schemes.
    """
    i = 0
    target_values = np.zeros(len(target_interfaces) - 1)

    for j in range(len(target_interfaces) - 1):
        left = target_interfaces[j]
        right = target_interfaces[j + 1]
        area = 0.0

        # Move i forward if we're behind the larger (super) bin
        while i < len(intermediate_interfaces) - 1 and intermediate_interfaces[i + 1] <= left:
            i += 1

        k = i
        while k < len(intermediate_interfaces) - 1 and intermediate_interfaces[k] < right:
            seg_left = max(left, intermediate_interfaces[k])
            seg_right = min(right, intermediate_interfaces[k + 1])

            if seg_right > seg_left:
                dx = seg_right - seg_left

                # linear interpolation
                y0 = intermediate_values[k]
                y1 = intermediate_values[k + 1]

                area += 0.5 * dx * (y0 + y1)

            k += 1

        target_values[j] = area

    return target_values

