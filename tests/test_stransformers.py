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
from retrivex.stransformers import IntJacobianExplainableTransformer


class TestIntJacobianExplainableTransformer:
    @pytest.fixture
    def mini_explainer(self):
        """ Explaineable Transformer based 
        on mini model for tests.
        """
        return IntJacobianExplainableTransformer("all-MiniLM-L6-v2")

    @pytest.mark.parametrize(
    ("text_a", "text_b"),
    [("This is some text", "This is similar text"), 
     ("Word", ""),
     ("This is an example text", "Another text for example")])
    def test_explain_no_errors(self, text_a, text_b, mini_explainer):
        mini_explainer.explain(
            text_a, 
            text_b, 
            layer_index=3, 
            num_interpolation_steps=5)
        
    @pytest.mark.parametrize(
    ("text_a", "text_b"),
    [("This is some text", "This is similar text"), 
     ("Word", ""),
     ("This is an example text", "Another text for example")])
    def test_explain_dot_similarity_no_errors(self, text_a, text_b, mini_explainer):
        mini_explainer.explain(
            text_a, 
            text_b, 
            layer_index=3, 
            num_interpolation_steps=5,
            similarity_metric="dot")
    
    @pytest.mark.parametrize(
    "text",
    ["This is some text", 
     "Word",
     ""])
    def test_explain_same_text(self, text, mini_explainer):
        attributions, _, _, _ = mini_explainer.explain(
            text, 
            text, 
            layer_index=5, 
            num_interpolation_steps=5)
        
        attributions = attributions.numpy()
        # check the same token-to-token attribution is max
        # except [CLS] token
        max_a = np.max(attributions, axis=0)
        max_b = np.max(attributions, axis=1)
        for i in range(1, attributions.shape[0]):
            assert attributions[i][i] == max_a[i] 
            assert attributions[i][i] == max_b[i] 

    @pytest.mark.parametrize(
    "text",
    ["This is some text"])
    def test_explain_layer_error(self, text, mini_explainer):
        with pytest.raises(Exception):
            mini_explainer.explain(
                text, 
                text, 
                layer_index=10, 
                num_interpolation_steps=5)
            
    @pytest.mark.parametrize(
    "text",
    ["This is some text", 
     "Word",
     ""])
    def test_explain_same_text_similarity(self, text, mini_explainer):
        _, _, _, details = mini_explainer.explain(
            text, 
            text,
            return_details=True,
            layer_index=5, 
            num_interpolation_steps=5)
        
        assert details["query_candidate_similarity"] == 1.0
        assert details["references_similarity"] == 1.0

    @pytest.mark.parametrize(
    ("text_a", "text_b"),
    [("This is some text", "This is similar text"), 
     ("Word", ""),
     ("This is an example text", "Another text for example")])
    def test_explain_custom_layers(self, text_a, text_b, mini_explainer):
        mini_explainer.explain(
            text_a, 
            text_b, 
            layer_index=3, 
            num_interpolation_steps=5,
            encoder_layers_path="encoder.layer")
        
    @pytest.mark.parametrize(
    ("text_a", "text_b"),
    [("This is some text", "This is similar text"), 
     ("Word", ""),
     ("This is an example text", "Another text for example")])
    def test_explain_custom_layers_error(self, text_a, text_b, mini_explainer):
        with pytest.raises(AttributeError):
            mini_explainer.explain(
                text_a, 
                text_b, 
                layer_index=3, 
                num_interpolation_steps=5,
                encoder_layers_path="encoder.layers")
            
    @pytest.mark.parametrize(
    ("text_a", "text_b"),
    [("This is some text", "This is similar text"), 
     ("Word", ""),
     ("This is an example text", "Another text for example")])
    def test_explain_custom_embeddings(self, text_a, text_b, mini_explainer):
        mini_explainer.explain(
            text_a, 
            text_b, 
            layer_index=3, 
            num_interpolation_steps=5,
            embeddings_layer_path="embeddings.word_embeddings")
        
    @pytest.mark.parametrize(
    ("text_a", "text_b"),
    [("This is some text", "This is similar text"), 
     ("Word", ""),
     ("This is an example text", "Another text for example")])
    def test_explain_custom_embeddings_error(self, text_a, text_b, mini_explainer):
        with pytest.raises(AttributeError):
            mini_explainer.explain(
                text_a, 
                text_b, 
                layer_index=3, 
                num_interpolation_steps=5,
                embeddings_layer_path="encoder.embs")
