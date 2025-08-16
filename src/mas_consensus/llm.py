import torch
import logging
from phi.llm.base import LLM
from pydantic import Field, PrivateAttr
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typing import Optional, Any


class HuggingFaceLLM(LLM):
    """
    A wrapper for Hugging Face language models that is compatible with the phidata library.
    """

    model: str = Field(
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        description="The name/path of the model to use",
    )
    model_type: str = Field(
        default="causal", description="Type of the model, either 'causal' or 'seq2seq'."
    )
    max_tokens_response: int = Field(
        default=2048, description="Maximum number of tokens in response"
    )
    instruction_format: str = Field(
        default="llama", description="Instruction format to use: 'mistral' or 'llama'"
    )
    context_window: int = Field(
        default=16384, description="Context window for the worker/manager"
    )
    use_quantization: bool = Field(
        default=False, description="Whether to use 4-bit quantization"
    )

    _model: Any = PrivateAttr()  # Use Any to avoid type issues
    _tokenizer: AutoTokenizer = PrivateAttr()
    _logger: logging.Logger = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        if self.model_type == "causal":
            model_class = AutoModelForCausalLM
        elif self.model_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self._model = model_class.from_pretrained(
            self.model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        # Handle pad_token for models that don't have one
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore
        self._logger = logging.getLogger(__name__)

    def format_prompt(self, instruction: str) -> str:
        """Formats the prompt according to the specified instruction format."""
        instruction = instruction.strip()
        if self.instruction_format.lower() == "llama":
            return (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                f"{instruction}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        elif self.instruction_format.lower() == "mistral":
            return f"<s>[INST]{instruction}[/INST]"
        elif self.instruction_format.lower() == "phi":
            return f"Instruct: {instruction}\nOutput:"
        else:
            return instruction

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Completes the prompt using the loaded Hugging Face model.
        Note: This implementation is synchronous.
        """
        self._logger.info(f"Generating completion for prompt of length {len(prompt)}")

        # Tokenize the input
        inputs = self._tokenizer(  # type: ignore
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window,
            padding=True,
        )

        # Move inputs to the same device as the model
        device = getattr(self._model, "device", "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self._model.generate(  # type: ignore
                **inputs,
                max_new_tokens=self.max_tokens_response,
                pad_token_id=getattr(self._tokenizer, "pad_token_id", None),
                **kwargs,
            )

        # Decode the response
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore
        # Remove the prompt from the response if it's included
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()
        self._logger.info(f"Generated response of length {len(response)}")
        return response

    def invoke(self, *args, **kwargs) -> str:
        """
        Invokes the model with the given prompt.
        This method is required by the phi framework.
        """
        # Extract prompt from args or kwargs
        prompt = ""
        if args:
            prompt = args[0] if isinstance(args[0], str) else str(args[0])
        elif "prompt" in kwargs:
            prompt = kwargs["prompt"]
        elif "messages" in kwargs:
            # Handle messages format if provided
            messages = kwargs["messages"]
            if isinstance(messages, list):
                # Simple concatenation of messages
                prompt = "\n".join(
                    [
                        msg.get("content", "") if isinstance(msg, dict) else str(msg)
                        for msg in messages
                    ]
                )
            else:
                prompt = str(messages)

        return self.complete(prompt, **kwargs)

    def response(self, *args, **kwargs) -> str:
        """
        Generates a response to the given prompt.
        This method is required by the phi framework.
        """
        return self.invoke(*args, **kwargs)

    @property
    def tokenizer(self):
        return self._tokenizer
