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
Explainability for vector similarity models available via SentenceTransformer.
"""
from typing import Tuple, Dict, List, Optional
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


from retrivex import _torch_utils


class IntJacobianExplainableTransformer(SentenceTransformer):
    """
    Extended SentenceTransformer which provides vector similarity explainability through 
    integrated gradients based on approach described in paper "Approximate Attributions 
    for Off-the-Shelf Siamese Transformers" (Moeller et al., EACL 2024) 
    https://aclanthology.org/2024.eacl-long.125/. This approach is an approximation of 
    Integrated Jacobians (IJ) method (which is based on generalization of integrated 
    gradients) with using padding sequences as approximate references that suppose to 
    have similarity close to zero for most inputs.

    This class enables attribution analysis by computing how different tokens contribute
    to the similarity between two sentences. It extends the base SentenceTransformer
    to include reference embeddings and attribution computation capabilities.
    """

    def forward(self, input_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass that separates reference embeddings from sentence embeddings.
        
        The input is expected to contain both sentence embeddings and a reference embedding
        concatenated together. This method splits them and adds the reference as a separate
        feature for later use in attribution analysis.
        
        Args:
            input_features: Dictionary containing tokenized input with embeddings and attention masks.
            
        Returns:
            Dictionary with sentence embeddings, attention masks, and separated reference embeddings
        """
        # Extract input tensors from features dictionary
        input_ids = input_features['input_ids']
        
        # Get sequence length and device for tensor operations
        sequence_length = input_ids.shape[1]
        device = input_ids.device
        
        # Create reference token sequence using tokenizer-specific special tokens
        reference_token_ids = self._create_reference_sequence(sequence_length, device)
        
        # Concatenate original input_ids with the reference sequence
        # This adds one more sequence to the batch
        input_features['input_ids'] = torch.cat([input_ids, reference_token_ids], dim=0)
        
        # Call parent class forward method to get embeddings
        output_features = super().forward(input_features)

        sentence_embeddings = output_features['sentence_embedding']
        
        # Split the last embedding as reference, keep the rest as sentence embeddings
        output_features.update({
            'sentence_embedding': sentence_embeddings[:-1],  # All but last embedding
        })
        
        # Store reference embedding and its attention mask separately
        output_features['reference_embedding'] = sentence_embeddings[-1]
        
        return output_features
    

    def _create_reference_sequence(self, sequence_length, device) -> torch.Tensor:
        """
        Create a reference token sequence that's compatible with any tokenizer.
        
        The reference sequence follows the pattern: [CLS/BOS] + [PAD]*(seq_len-2) + [SEP/EOS]
        in accordance to the approach described in paper "Approximate Attributions for 
        Off-the-Shelf Siamese Transformers" (Moeller et al., EACL 2024) 
        https://aclanthology.org/2024.eacl-long.125/.

        This method automatically detects the appropriate special token IDs from the tokenizer.
        Standard BERT tokens are used if tokenizer special tokens could not be detected.
        
        Args:
            sequence_length (int): Length of the sequence to create
            device: PyTorch device to place the tensor on
            
        Returns:
            torch.Tensor: Reference token sequence of shape (1, sequence_length)
        """
        # Get the tokenizer from the parent transformer model
        tokenizer = self[0].auto_model.config.tokenizer if hasattr(self[0].auto_model.config, 'tokenizer') else None
        
        # If we can't access tokenizer directly, try to get it from the model
        if tokenizer is None and hasattr(self, 'tokenizer'):
            tokenizer = self.tokenizer
        elif tokenizer is None and hasattr(self[0].auto_model, 'tokenizer'):
            tokenizer = self[0].auto_model.tokenizer
        
        # Fallback: Try to infer special tokens from common tokenizer attributes
        if tokenizer is not None:
            # Get special token IDs with fallbacks for different tokenizer types
            cls_token_id = getattr(tokenizer, 'cls_token_id', None)
            sep_token_id = getattr(tokenizer, 'sep_token_id', None)
            pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            
            # Handle different tokenizer naming conventions
            if cls_token_id is None:
                cls_token_id = getattr(tokenizer, 'bos_token_id', None)  # For GPT-style models
            if sep_token_id is None:
                sep_token_id = getattr(tokenizer, 'eos_token_id', None)  # For GPT-style models
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, 'unk_token_id', 0)  # Fallback to 0 or unk_token
            
            # Use detected token IDs if available
            if sep_token_id is not None and pad_token_id is not None:
                num_pad_tokens = sequence_length - (2 if cls_token_id else 1)
                reference_sequence = [pad_token_id] * (num_pad_tokens) + [sep_token_id]
                if cls_token_id:
                    reference_sequence = [cls_token_id] + reference_sequence
                return torch.tensor([reference_sequence], dtype=torch.long).to(device)
        
        # Ultimate fallback: Use the original hardcoded values with explanation
        # This maintains backward compatibility and works for most BERT-style models
        print("Warning: Could not detect tokenizer special tokens. Using default BERT-style tokens.")
        print("Default mapping: 0=[CLS], 1=[PAD], 2=[SEP]")
        reference_sequence = [0] + [1] * (sequence_length - 2) + [2]
        return torch.tensor([reference_sequence], dtype=torch.long).to(device)
    

    def __tensors_to_device(self, tensors: dict, device: torch.device):
        """
        Move tensors to device.
        """
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                tensors[k] = v.to(device)


    def _initialize_attribution_to_layer(
            self, layer_index: int, 
            num_interpolation_steps: int,
            encoder_layers_path: Optional[str] = None):
        """
        Initialize attribution computation for a specific transformer layer.
        
        This method should be implemented by subclasses to set up hooks for
        capturing intermediate representations at the specified layer.
        
        Args:
            layer_index: Index of the transformer layer to analyze
            num_interpolation_steps: Number of steps for integrated gradients computation
            
        Set up attribution hooks for a specific encoder layer.
        
        Args:
            layer_index: Index of the encoder layer to analyze
            num_interpolation_steps: Number of interpolation steps for integrated gradients
            encoder_layers_path: Custom path to intermediate encoder layer (for example "encoder.enc_block")
        """
        if hasattr(self, 'forward_hook') and self.forward_hook is not None:
            raise AttributeError('A hook is already registered. Call reset_attribution_hooks() first.')

        if encoder_layers_path:
            parts = encoder_layers_path.split('.')
            current_obj = self[0].auto_model
            for part in parts:
                if not hasattr(current_obj, part):
                    raise AttributeError(f"Couldn't accesss encoder layers by path `{encoder_layers_path}`")
                current_obj = getattr(current_obj, part)
            encoder_layers = current_obj
        elif hasattr(self[0].auto_model, "encoder") and hasattr(self[0].auto_model.encoder, "layer"):
            encoder_layers = self[0].auto_model.encoder.layer
        elif hasattr(self[0].auto_model, "encoder") and hasattr(self[0].auto_model.encoder, "block"):
            encoder_layers = self[0].auto_model.encoder.block
        elif hasattr(self[0].auto_model, "layers"):
            encoder_layers = self[0].auto_model.layers
        else:
            raise AttributeError("Couldn't access intermediate encoders of this model. " \
                "Try using parameter `encoder_layers_path`")
        assert layer_index < len(encoder_layers), f'Model only has {len(encoder_layers)} layers, requested layer {layer_index}'
        
        try:
            self.num_interpolation_steps = num_interpolation_steps
            self.intermediate_representations = []
            
            # Register hook for the specified layer
            self.forward_hook = encoder_layers[layer_index].register_forward_pre_hook(
                _torch_utils.create_interpolation_hook(
                    num_interpolation_steps=num_interpolation_steps, 
                    output_storage=self.intermediate_representations
                )
            )
        except AttributeError:
            raise AttributeError('The encoder model is not supported for attribution analysis')

    def _reset_attribution_hooks(self):
        """
        Remove all registered hooks and clean up attribution state.
        
        This method should be implemented by subclasses to properly clean up
        any registered forward hooks and intermediate storage.
        Remove registered hooks and clean up attribution state."""
        if hasattr(self, 'forward_hook') and self.forward_hook is not None:
            self.forward_hook.remove()
            self.forward_hook = None
        else:
            print('No attribution hook has been registered.')

    def _get_readable_tokens(self, input_text: str) -> List[str]:
        """
        Tokenize text and add special tokens with proper formatting.
        
        This method tokenizes the input text, removes tokenizer-specific prefixes,
        and adds CLS and EOS tokens for proper model input formatting.
        
        Args:
            input_text: Raw text to tokenize
            
        Returns:
            List of tokens with special tokens added
        """
        tokens = self[0].tokenizer.tokenize(input_text, )
        
        # Remove tokenizer-specific prefixes (e.g., 'Ġ' for GPT-style, 'Â' for some models)
        cleaned_tokens = [
            token[1:] if token[0] in ['Ġ', 'Â'] else token 
            for token in tokens
        ]
        
        # Add classification and end-of-sequence tokens
        formatted_tokens = cleaned_tokens + ["[EOS]"]

        cls_token = getattr(self[0].tokenizer, 'cls_token', None)
        if cls_token is None:
            cls_token = getattr(self[0].tokenizer, 'bos_token', None)
        if cls_token:
            formatted_tokens = [cls_token] + formatted_tokens
        return formatted_tokens

    def _compute_integrated_gradients_jacobian(
            self, 
            final_embedding: torch.Tensor, 
            intermediate_representations: torch.Tensor,
            move_to_cpu: bool = True,
            show_progress: bool = True
    ) -> torch.Tensor:
        """
        Compute the Jacobian matrix using integrated gradients.
        
        This method computes gradients of each dimension of the final embedding with respect
        to intermediate representations, implementing the integrated gradients technique.
        
        Args:
            final_embedding: The final text embedding to compute gradients for
            intermediate_representations: Intermediate layer representations from interpolation
            move_to_cpu: Whether to move computed gradients to CPU for memory efficiency
            show_progress: Whether to show progress bar during computation
            
        Returns:
            Jacobian tensor of shape [embedding_dim, sequence_length, mbedding_dim]
        """
        embedding_dimension = self[0].get_word_embedding_dimension()
        jacobian_components = []
        
        # Compute gradients for each dimension of the final embedding
        for dimension_idx in tqdm(range(embedding_dimension), disable=not show_progress):
            # Retain graph for all but the last dimension
            should_retain_graph = (dimension_idx < embedding_dimension - 1)
            
            # Compute gradient of this embedding dimension w.r.t. intermediate representations
            dimension_gradients = torch.autograd.grad(
                outputs=list(final_embedding[:, dimension_idx]), 
                inputs=intermediate_representations, 
                retain_graph=should_retain_graph
            )[0].detach()
            
            if move_to_cpu:
                dimension_gradients = dimension_gradients.cpu()
                
            jacobian_components.append(dimension_gradients)
        
        # Stack all gradients and average over interpolation steps
        jacobian_matrix = torch.stack(jacobian_components) / self.num_interpolation_steps
        
        # Remove the reference embedding dimension and sum over interpolation steps
        # Shape: [embedding_dim, sequence_length, embedding_dim]
        jacobian_matrix = jacobian_matrix[:, :-1, :, :].sum(dim=1)
        
        return jacobian_matrix
    
    def _process_text(
            self, 
            text: str,
            representations_idx: int,
            device: torch.device, 
            move_to_cpu: bool = True, 
            show_progress: bool = True):
        """
        Process text and compute embeddings, jacobian and interpolation diff.
        """
        tokenized_input = self[0].tokenize([text])
        self.__tensors_to_device(tokenized_input, device)
        features = self.forward(tokenized_input)
        embedding = features['sentence_embedding']
        intermediate = self.intermediate_representations[representations_idx]
        
        # Compute Jacobian for text
        jacobian = self._compute_integrated_gradients_jacobian(
            embedding, intermediate, move_to_cpu=move_to_cpu, show_progress=show_progress
        )
        embedding_dim, sequence_length, hidden_dim = jacobian.shape
        jacobian = jacobian.reshape((embedding_dim, sequence_length * hidden_dim))

        # Compute interpolation difference for text
        interpolation_diff = intermediate[0] - intermediate[-1]
        interpolation_diff = interpolation_diff.reshape(sequence_length * hidden_dim, 1).detach()

        return embedding, jacobian, interpolation_diff, hidden_dim, sequence_length, features
        
    def explain(
            self, 
            query: str, 
            candidate: str, 
            similarity_metric: str = 'cosine',
            return_details: bool = False,
            move_to_cpu: bool = False,
            show_progress: bool = True,
            compress_embedding_dimension: bool = True,
            layer_index=0, 
            num_interpolation_steps=250,
            encoder_layers_path: Optional[str] = None,
            embeddings_layer_path: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[str], List[str], Optional[Dict]]:
        """
        Explain similarity between two texts using integrated gradients attribution.
        
        This method computes token-level attribution scores that explain how each token
        pair in `query` and `candidate` texts contributes to the similarity between 
        `query` and `candidate`.
        
        Args:
            query: Query text to compare
            candidate: Candidate text to compare  
            similarity_measure: 'cosine' or 'dot' product similarity
            return_details: Whether to return individual similarity components
            move_to_cpu: Whether to move computations to CPU for memory efficiency
            show_progress: Whether to show progress bars during computation
            compress_embedding_dimension: Whether to sum over embedding dimensions
            layer_index: Index of the encoder layer to use in analysis
            num_interpolation_steps: Number of interpolation steps for integrated gradients
            encoder_layers_path: Optional custom path to intermediate encoder layer 
            (for example, "encoder.enc_block") if intermediate encoders could not be found 
            automatically.
            embeddings_layer_path: Optional custom path to embeddings layer 
            (for example, "encoder.emb") if embeddings layer could not be found 
            automatically.
            
        Returns:
            Tuple containing:
            - attribution_matrix: Token-to-token attribution scores
            - tokens_query: Tokenized version of query text
            - tokens_candidate: Tokenized version of candidate text
            - (optional) similarity_score, reference_similarities if return_detailed_terms=True
        """
        assert similarity_metric in ['cosine', 'dot'], (f"Invalid similarity metric: {similarity_metric}. " +
            "Allowed metrics are `cosine` and `dot`")

        self._initialize_attribution_to_layer(layer_index, num_interpolation_steps, encoder_layers_path)

        # Clear any previous intermediate representations
        self.intermediate_representations.clear()
        # Get embeddings layer
        if embeddings_layer_path:
            parts = embeddings_layer_path.split('.')
            current_obj = self[0].auto_model
            for part in parts:
                if not hasattr(current_obj, part):
                    raise AttributeError(f"Couldn't accesss embeddings layer by path `{encoder_layers_path}`")
                current_obj = getattr(current_obj, part)
            embeddings = current_obj
        elif hasattr(self[0].auto_model, "embed_tokens"):
             embeddings = self[0].auto_model.embed_tokens
        elif (hasattr(self[0].auto_model, "embeddings") and 
            hasattr(self[0].auto_model.embeddings, "word_embeddings")):
            embeddings = self[0].auto_model.embeddings.word_embeddings
        elif (hasattr(self[0].auto_model, "encoder") and 
            hasattr(self[0].auto_model.encoder, "embed_tokens")):
            embeddings = self[0].auto_model.encoder.embed_tokens
        else:
            raise AttributeError("Couldn't access embeddings layer of this model. " \
                "Try using parameter `embeddings_layer_path`")

        device = embeddings.weight.device

        # Process `query` text
        embedding_q, jacobian_q, interpolation_diff_q, dimension_q, length_q, features_q = self._process_text(
            query,
            0,
            device,
            move_to_cpu,
            show_progress)

        # Process `candidate` text
        embedding_c, jacobian_c, interpolation_diff_c, dimension_c, length_c, features_c = self._process_text(
            candidate,
            1,
            device,
            move_to_cpu,
            show_progress)

        # Compute attribution matrix using integrated gradients formula
        jacobian_product = torch.mm(jacobian_q.T, jacobian_c)
        query_diff_expanded = interpolation_diff_q.repeat(1, length_c * dimension_c)
        cand_diff_expanded = interpolation_diff_c.repeat(1, length_q * dimension_q)
        
        if move_to_cpu:
            query_diff_expanded = query_diff_expanded.detach().cpu()
            cand_diff_expanded = cand_diff_expanded.detach().cpu()
            embedding_q = embedding_q.detach().cpu()
            embedding_c = embedding_c.detach().cpu()
        
        # Compute final attribution matrix
        attribution_matrix = query_diff_expanded * jacobian_product * cand_diff_expanded.T
        
        # Normalize by embedding norms for cosine similarity
        if similarity_metric == 'cosine':
            attribution_matrix = attribution_matrix / torch.norm(embedding_q[0]) / torch.norm(embedding_c[0])
        
        # Reshape to token-level attributions
        attribution_matrix = attribution_matrix.reshape(
            length_q, dimension_q, length_c, dimension_c
        )
        
        # Optionally compress embedding dimension by summing
        if compress_embedding_dimension:
            attribution_matrix = attribution_matrix.sum(dim=(1, 3))
        
        attribution_matrix = attribution_matrix.detach().cpu()

        # Get readable tokens for both texts
        query_tokens = self._get_readable_tokens(query)
        cand_tokens = self._get_readable_tokens(candidate)

        details = None
        # Return detailed similarity components if requested
        if return_details:
            reference_embedding_q = features_q['reference_embedding']
            reference_embedding_c = features_c['reference_embedding']
            if move_to_cpu:
                reference_embedding_q = reference_embedding_q.detach().cpu()
                reference_embedding_c = reference_embedding_c.detach().cpu()
            
            # Compute various similarity components
            if similarity_metric == 'cosine':
                metric = lambda x, y: torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()
            elif similarity_metric == 'dot':
                metric = lambda x, y: torch.dot(x, y).item()

            main_similarity = metric(embedding_q[0], embedding_c[0])
            ref_embedding_query_similarity = metric(embedding_q[0], reference_embedding_c)
            ref_embedding_cand_similarity = metric(embedding_c[0], reference_embedding_q)
            reference_similarity = metric(reference_embedding_q, reference_embedding_c)
                
            details = {
                "query_candidate_similarity": main_similarity, 
                "query_reference_similarity": ref_embedding_query_similarity, 
                "candidate_reference_similarity": ref_embedding_cand_similarity, 
                "references_similarity": reference_similarity}

        self._reset_attribution_hooks()
        return attribution_matrix, query_tokens, cand_tokens, details
