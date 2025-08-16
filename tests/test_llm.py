import unittest
from unittest.mock import Mock, patch, MagicMock
import torch

from mas_consensus.llm import HuggingFaceLLM


class TestHuggingFaceLLM(unittest.TestCase):

    @patch("mas_consensus.llm.AutoModelForCausalLM")
    @patch("mas_consensus.llm.AutoTokenizer")
    def test_init_causal_model(self, mock_tokenizer, mock_model):
        # Set up the mocks properly
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock the model's dtype property
        mock_model_instance.dtype = torch.float32

        llm = HuggingFaceLLM(
            model="test-model", model_type="causal", instruction_format="llama"
        )

        # Verify model was initialized correctly
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")

    @patch("mas_consensus.llm.AutoModelForSeq2SeqLM")
    @patch("mas_consensus.llm.AutoTokenizer")
    def test_init_seq2seq_model(self, mock_tokenizer, mock_model):
        # Set up the mocks properly
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        llm = HuggingFaceLLM(
            model="test-model", model_type="seq2seq", instruction_format="t5"
        )

        # Verify model was initialized correctly
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")

    def test_format_prompt_llama(self):
        # Test without actually loading a model
        with (
            patch("mas_consensus.llm.AutoModelForCausalLM"),
            patch("mas_consensus.llm.AutoTokenizer"),
        ):
            llm = HuggingFaceLLM(model="test-model", instruction_format="llama")
            instruction = "Test instruction"
            formatted = llm.format_prompt(instruction)

            self.assertIn("<|begin_of_text|>", formatted)
            self.assertIn("<|start_header_id|>user<|end_header_id|>", formatted)
            self.assertIn(instruction, formatted)
            self.assertIn("<|start_header_id|>assistant<|end_header_id|>", formatted)

    def test_format_prompt_mistral(self):
        # Test without actually loading a model
        with (
            patch("mas_consensus.llm.AutoModelForCausalLM"),
            patch("mas_consensus.llm.AutoTokenizer"),
        ):
            llm = HuggingFaceLLM(model="test-model", instruction_format="mistral")
            instruction = "Test instruction"
            formatted = llm.format_prompt(instruction)

            self.assertIn("<s>[INST]", formatted)
            self.assertIn(instruction, formatted)
            self.assertIn("[/INST]", formatted)

    def test_format_prompt_phi(self):
        # Test without actually loading a model
        with (
            patch("mas_consensus.llm.AutoModelForCausalLM"),
            patch("mas_consensus.llm.AutoTokenizer"),
        ):
            llm = HuggingFaceLLM(model="test-model", instruction_format="phi")
            instruction = "Test instruction"
            formatted = llm.format_prompt(instruction)

            self.assertIn("Instruct: ", formatted)
            self.assertIn(instruction, formatted)
            self.assertIn("Output:", formatted)

    def test_format_prompt_default(self):
        # Test without actually loading a model
        with (
            patch("mas_consensus.llm.AutoModelForCausalLM"),
            patch("mas_consensus.llm.AutoTokenizer"),
        ):
            llm = HuggingFaceLLM(model="test-model", instruction_format="unknown")
            instruction = "Test instruction"
            formatted = llm.format_prompt(instruction)

            # For unknown formats, it should return the instruction as is
            self.assertEqual(formatted, instruction)


if __name__ == "__main__":
    unittest.main()
