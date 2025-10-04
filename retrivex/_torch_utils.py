
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
Internal torch utils module.
"""
from typing import List, Callable
import torch



def interpolate_between_input_and_reference(
    embeddings: torch.Tensor,
    num_interpolation_steps: int
) -> torch.Tensor:
    """
    Create a linear interpolation path between sentence embedding and reference embedding.
    
    This function implements the core interpolation needed for integrated gradients.
    It creates a series of intermediate embeddings that gradually transition from
    the sentence embedding to the reference embedding, which is essential for
    computing attribution scores via integrated gradients.
    
    The interpolation formula is: interpolated = reference + alpha * (sentence - reference)
    where alpha goes from 1.0 to 0.0 in equal steps.
    
    Args:
        embeddings: Tensor of shape [2, seq_len, hidden_dim] where:
                   - embeddings[0] is the sentence embedding (target)
                   - embeddings[1] is the reference embedding (baseline)
        num_interpolation_steps: Number of interpolation steps to create
        
    Returns:
        Tensor of shape [num_interpolation_steps + 1, seq_len, hidden_dim] containing:
        - First num_interpolation_steps: interpolated embeddings from sentence to reference
        - Last element: the reference embedding itself
        
    Example:
        If num_interpolation_steps = 3:
        - Step 0: reference + 1.0 * (sentence - reference) = sentence
        - Step 1: reference + 0.67 * (sentence - reference) 
        - Step 2: reference + 0.33 * (sentence - reference)
        - Step 3: reference + 0.0 * (sentence - reference) = reference
    """
    assert embeddings.shape[0] == 2, ("Expected 2 embeddings (sentence + reference), " +
        f"got {embeddings.shape[0]}")

    # Calculate step size for interpolation (from 1.0 to 0.0)
    interpolation_step_size = 1.0 / num_interpolation_steps
    device = embeddings.device

    # Extract sentence and reference embeddings
    sentence_embedding = embeddings[0]  # Shape: [seq_len, hidden_dim]
    reference_embedding = embeddings[-1].unsqueeze(0)  # Shape: [1, seq_len, hidden_dim]

    # Create interpolation coefficients: [1.0, 1-step, 1-2*step, ..., 0.0]
    # Shape: [num_interpolation_steps, 1, 1] for broadcasting
    interpolation_alphas = torch.arange(
        start=1.0,
        end=0.0,
        step=-interpolation_step_size
    ).unsqueeze(1).unsqueeze(1).to(device)

    # Compute interpolated embeddings using broadcasting
    # interpolated = reference + alpha * (sentence - reference)
    embedding_difference = sentence_embedding - reference_embedding
    interpolated_embeddings = reference_embedding + interpolation_alphas * embedding_difference

    # Concatenate interpolated steps with the final reference embedding
    # This ensures we have exactly num_interpolation_steps + 1 total embeddings
    final_interpolation_sequence = torch.cat([interpolated_embeddings, reference_embedding])

    return final_interpolation_sequence


def create_interpolation_hook(
    num_interpolation_steps: int,
    output_storage: List[torch.Tensor]
) -> Callable:
    """
    Create a forward hook for Transformer-based models to capture interpolated embeddings.
    
    Transformer-based models have a simpler structure where we only need to interpolate
    the main hidden states (first input argument). This hook modifies the
    forward pass to use interpolated embeddings instead of the original ones.
    
    Args:
        num_interpolation_steps: Number of steps for integrated gradients
        output_storage: List to store the interpolated embeddings for later use
        
    Returns:
        Hook function that can be registered with PyTorch modules
    """
    def interpolation_function(_, layer_inputs):
        """
        Hook function which intercepts model encoder layer inputs and applies interpolation.
        
        Args:
            _: The transformer layer (unused in this hook)
            layer_inputs: Tuple of inputs to the layer, where:
                         - layer_inputs[0]: hidden states tensor [2, seq_len, hidden_dim]
                         - layer_inputs[1:]: other inputs (attention masks, etc.)
        
        Returns:
            Modified layer inputs with interpolated hidden states
        """
        # Apply interpolation to the hidden states (first input)
        interpolated_hidden_states = interpolate_between_input_and_reference(
            embeddings=layer_inputs[0],
            num_interpolation_steps=num_interpolation_steps
        )

        # Store interpolated embeddings for gradient computation
        output_storage.append(interpolated_hidden_states)

        # Return modified inputs with interpolated hidden states
        return (interpolated_hidden_states,) + layer_inputs[1:]

    return interpolation_function
