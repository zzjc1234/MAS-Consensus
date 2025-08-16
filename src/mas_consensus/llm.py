import torch
from phi.llm.base import LLM
from pydantic import Field, PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HuggingFaceLLM(LLM):
    """
    A wrapper for Hugging Face language models that is compatible with the phidata library.
    """
    model: str = Field(
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        description="The name/path of the model to use"
    )
    max_tokens_response: int = Field(
        default=2048,
        description="Maximum number of tokens in response"
    )
    instruction_format: str = Field(
        default="llama",
        description="Instruction format to use: 'mistral' or 'llama'"
    )
    context_window: int = Field(
        default=16384,
        description="Context window for the worker/manager"
    )
    use_quantization: bool = Field(
        default=False,
        description="Whether to use 4-bit quantization"
    )

    _model: AutoModelForCausalLM = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype="auto"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

    def format_prompt(self, instruction: str) -> str:
        """Formats the prompt according to the specified instruction format."""
        instruction = instruction.strip()
        if self.instruction_format.lower() == "llama":
            return (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                f"{instruction}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        return f"<s>[INST]{instruction}[/INST]"

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Completes the prompt using the loaded Hugging Face model.
        Note: This implementation is synchronous.
        """
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_tokens_response,
            **kwargs
        )
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The response from some models includes the prompt, so we remove it.
        return response[len(prompt):]

    @property
    def tokenizer(self):
        return self._tokenizer
