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
"""
Visualization Utils Module.

This module provides visualization tools to understand
the interactions between tokens in explanations.
"""
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.path import Path


# Color scheme for relevance visualization
RELEVANCE_COLORS = {
    'positive': [1.0, 0.0, 0.0],  # Red for positive relevance
    'negative': [0.0, 0.0, 1.0],  # Blue for negative relevance
}


def __create_token_positions(
        tokens: List[str],
        base_width: int = 50,
        position_coef: float = 0.15) -> np.ndarray:
    """
    Create a mapping of tokens to their positions based on character length.
    """
    # Calculate total character count and canvas width
    total_chars = len(''.join(tokens))
    total_canvas_width = base_width * total_chars
    average_width = int(total_canvas_width / total_chars)

    # Initialize canvas and tracking variables
    current_position = 0
    word_centers = []

    # Position each word on the canvas
    for word in tokens:
        # Calculate word width ((1-coef) * average + coef * proportional to character count)
        proportional_width = int((len(word) / total_chars) * total_canvas_width)
        word_width = int((1 - position_coef) * average_width + position_coef * proportional_width)

        # Store center position for this word
        word_centers.append(current_position + word_width // 2)
        current_position += word_width

    return np.array(word_centers).reshape(-1, 1)


def __create_relevance_color(
        relevance_score: float,
        alpha: float,
        threshold: float = 0.0) -> str:
    """
    Create a color with alpha blending based on relevance score.
    """
    color_type = 'positive' if relevance_score > threshold else 'negative'
    base_color = RELEVANCE_COLORS[color_type]
    return matplotlib.colors.to_hex(base_color + [alpha], keep_alpha=True)


def plot_connections(
                    scores: Union[np.array, torch.tensor],
                    query_tokens: List[str],
                    candidate_tokens: List[str],
                    details: Dict = None,
                    norm_scores: bool = True,
                    figsize: Tuple[int, int] = (5,5),
                    plot_title: Optional[str] = None,
                    query_label: str = "query",
                    candidate_label: str = "candidate",
                    token_canvas_width: int = 50,
                    token_fontsize: int = 10,
                    title_fontsize: int = 24,
                    connection_line_thick: float = 3.0,
                    title_pad: float = 10,
                    plot_pad: float = 0.05) -> None:
    """
    Create a parallel coordinate plot showing interactions between token pairs.
    
    This visualization shows how features from one sequence relate to features
    in another sequence through curved connections colored by relevance scores.
    
    Args:
        scores: Token to token attribution scores matrix
        query_tokens: Query tokens list
        candidate_tokens: Candidate tokens list
        details: Additional similarity terms
        figsize: Size of the plot
        norm_scores: Whether to normalize scores. Default value is `True`
        plot_title: Optional title for the plot
        query_label: Label for query sequence in a plot. Defaul value is `query`
        candidate_label: Label for candidate sequence in a plot. 
        Default value is `candidate`
        token_canvas_width: Base wisth fpr token in a plot. Default value is 50
        token_fontsize: Font size of tokens
        title_fontsize: Font size of plot title
        connection_line_thick: Thickness of the connection lines between tokens. 
        Default value is 3.0
        title_pad: Padding for title. Default value is 10.0
        plot_pad: Padding for a plot. Default value is  0.05
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()

    assert scores.shape[0] == len(query_tokens)
    assert scores.shape[1] == len(candidate_tokens)

    if norm_scores:
        scores = scores / np.linalg.norm(scores)

    if details:
        print(str(details))

    sequence_tokens = [query_tokens, candidate_tokens]
    # Create position canvases for both sentences
    sentence1_positions = __create_token_positions(
        query_tokens,
        base_width=token_canvas_width
    )

    sentence2_positions = __create_token_positions(
        candidate_tokens,
        base_width=token_canvas_width
    )

    # Normalize second sentence positions to match first sentence scale
    pos2_normalized = ((sentence2_positions - sentence2_positions.min()) /
                      (sentence2_positions.max() - sentence2_positions.min()) *
                      (sentence1_positions.max() - sentence1_positions.min()) +
                      sentence1_positions.min())

    # Create coordinate pairs for all word combinations
    coordinate_pairs = np.array(
        np.meshgrid(sentence1_positions, pos2_normalized.T), dtype=float
    ).T.reshape(-1, 2)

    # Calculate plot boundaries with padding
    y_mins = coordinate_pairs.min(axis=0)
    y_maxs = coordinate_pairs.max(axis=0)
    y_ranges = y_maxs - y_mins
    y_mins -= y_ranges * plot_pad
    y_maxs += y_ranges * plot_pad
    y_ranges = y_maxs - y_mins
    # Transform coordinates for dual-axis plotting
    transformed_coords = np.zeros_like(coordinate_pairs)
    transformed_coords[:, 0] = coordinate_pairs[:, 0]
    transformed_coords[:, 1:] = ((coordinate_pairs[:, 1:] - y_mins[1:]) / y_ranges[1:] *
                                y_ranges[0] + y_mins[0])

    # Set up the plot with dual y-axes
    fig, primary_axis = plt.subplots(figsize=figsize)

    # Create additional y-axes for each text
    all_axes = [primary_axis] + [primary_axis.twinx() for i in range(coordinate_pairs.shape[1] - 1)]

    # Configure each axis
    for axis_idx, axis in enumerate(all_axes):
        axis.set_ylim(y_mins[axis_idx], y_maxs[axis_idx])
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.set_yticks(np.unique(coordinate_pairs[:, axis_idx]))

        if axis != primary_axis:
            axis.spines['left'].set_visible(False)
            axis.yaxis.set_ticks_position('right')
            axis.spines["right"].set_position(("axes", axis_idx / (coordinate_pairs.shape[1] - 1)))

        # Set word labels as y-tick labels
        axis.set_yticklabels(sequence_tokens[axis_idx], fontsize=token_fontsize)
        axis.set_ylim(axis.get_ylim()[::-1])  # Reverse y-axis

    # Configure x-axis
    primary_axis.set_xlim(0, coordinate_pairs.shape[1] - 1)
    primary_axis.set_xticks(range(coordinate_pairs.shape[1]))
    primary_axis.set_xticklabels([query_label, candidate_label])
    primary_axis.tick_params(axis='x', which='major', pad=7)
    primary_axis.spines['right'].set_visible(False)
    primary_axis.xaxis.tick_top()

    if plot_title:
        primary_axis.set_title(plot_title, fontsize=title_fontsize, pad=title_pad)

    # Draw connections between token pairs
    for pair_idx in range(coordinate_pairs.shape[0]):
        # Create Bezier curve for smooth connections
        curve_points = list(zip(
            [x for x in np.linspace(0, len(coordinate_pairs) - 1,
                                   len(coordinate_pairs) * 3 - 2, endpoint=True)],
            np.repeat(coordinate_pairs[pair_idx, :], 2)
        ))

        # Define path codes for Bezier curve
        path_codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(curve_points) - 1)]
        connection_path = Path(curve_points, path_codes)

        # Get relevance score for this token pair
        token1_idx = np.where(sentence1_positions.squeeze() == coordinate_pairs[pair_idx, 0])
        token2_idx = np.where(pos2_normalized.squeeze() == coordinate_pairs[pair_idx, 1])
        pair_relevance = scores[token1_idx, token2_idx].squeeze().item()

        # Create colored connection based on relevance
        connection_color = __create_relevance_color(
            relevance_score=pair_relevance,
            alpha=abs(pair_relevance)
        )

        # Add the connection to the plot
        connection_patch = patches.PathPatch(
            connection_path,
            facecolor='none',
            linewidth=connection_line_thick,
            edgecolor=connection_color
        )
        primary_axis.add_patch(connection_patch)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
