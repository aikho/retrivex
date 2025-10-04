# Retrivex -- Retrieval Models Explainability Toolkit.
# Copyright (C) 2025 Andrei Khobnia

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
import pytest
import numpy as np
from retrivex.visualization import plot_connections


def test_plot_connection_no_errors(mocker):
        # Mock plt.show() to prevent the plot window from appearing
        mock_show = mocker.patch("matplotlib.pyplot.show")

        scores = np.array([
                [0.1, 0.9, 0.6],
                [0.1, 0.4, 0.5],
                [0.2, 0.8, 0.2]])
        
        plot_connections(
            scores, 
            ["T1", "T2", "T3"],
            ["T4", "T5", "T6"])

        # Assert that plt.show() was called
        mock_show.assert_called_once()


def test_plot_connection_lenght_errors():
        with pytest.raises(Exception):
                scores = np.array([
                        [0.1, 0.9, 0.6],
                        [0.1, 0.4, 0.5],
                        [0.2, 0.8, 0.2]])
        
                plot_connections(
                        scores, 
                        ["T1", "T2", "T3"],
                        ["T4", "T5"])

