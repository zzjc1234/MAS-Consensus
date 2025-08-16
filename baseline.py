!pip install PyMuPDF
!pip install -U transformers phidata

import asyncio
import gc
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import fitz
import httpx
import nest_asyncio
import numpy as np
import torch
from phi.assistant import Assistant
from phi.llm.base import LLM
from pydantic import Field, PrivateAttr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_with_pymupdf(pdf_path):
    references_pattern = re.compile(r'^references?$|^bibliography$')
    citation_pattern = re.compile(r'^\[\d+\]|^\d+\.|^[A-Z][a-z]+,\s*[A-Z]\.|^\s*[A-Z][a-z]+\s+[A-Z][a-z]+')
    number_pattern = re.compile(r'^\[\d+\]$|^\d+\.$')
    year_pattern = re.compile(r'^.*?\(\d{4}\)[.,].*?$')
    whitespace_pattern = re.compile(r'\s+')
    
    full_text = []
    in_references = False
    
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))
                
                for block in blocks:
                    text = block[4].strip() if block[4] else ''
                    if not text:
                        continue
                        
                    if references_pattern.match(text.lower()):
                        in_references = True
                        continue
                        
                    if in_references and citation_pattern.match(text):
                        continue
                    
                    if number_pattern.match(text):
                        continue
                        
                    if year_pattern.match(text) and len(text.split()) < 20:
                        continue
                    
                    text = whitespace_pattern.sub(' ', text)
                    if text.isupper() or len(text) <= 3:
                        full_text.append(f"\n## {text}\n")
                    else:
                        full_text.append(text + "\n")
    except Exception as e:
        raise ValueError(f"Error processing PDF file: {str(e)}")
    
    return "\n".join(full_text)

def extract_text(file_path):
    if file_path.lower().endswith('.txt'):
        return read_text_file(file_path)
    elif file_path.lower().endswith('.pdf'):
        return extract_text_with_pymupdf(file_path)
    else:
        raise ValueError(f"Unsupported file format. File must be either .pdf or .txt")

@dataclass
class TaskConfig:
    first_worker_instruction: str
    worker_instruction: str
    manager_instruction: str
    multi_summary_instruction: str

class TaskType(Enum):
    QA = "qa"
    SUMMARIZATION = "summarization"

class TaskFactory:
    @staticmethod
    def create_task(task_type: TaskType) -> TaskConfig:
        if task_type == TaskType.QA:
            return TaskConfig(
                first_worker_instruction="""
                [CONTEXT]
                {chunk_text}
                [QUESTION]
                {question}
                [TASK]
                List several facts from the provided context that might help to answer the question:
                - [Fact 1]
                - [Fact 2]
                ...
                Provide a [SUMMARY] by summarizing all the relevant information related to the question.
                Keep the existing structure in focus. Do not answer the question.
                """,
                worker_instruction="""
                [SUMMARY]
                {previous_summary}
                [CONTEXT]
                {chunk_text}
                [QUESTION]
                {question}
                [TASK]
                List several facts from the provided context that might help to answer the question:
                - [Fact 1]
                - [Fact 2]
                ...
                Prioritize all relevant information to the question and then refine the current summary by including the new additional information in [REVISED SUMMARY].
                Do not answer the question.
                """,
                manager_instruction="""
                [SUMMARY]
                {last_summary}
                [QUESTION]
                {question}
                [TASK]
                Using all context available, resolve any contradictions, and provide a comprehensive answer below.
                Answer format: <answer>...</answer>
                """,
                multi_summary_instruction="""
                [SUMMARIES]
                {last_summary}
                [QUESTION]
                {question}
                [TASK]
                Integrate all informations from every summary, resolve contradictions, and provide a comprehensive answer.
                Answer format: <answer>[Combined response using all relevant facts]</answer>
                """
            )
        elif task_type == TaskType.SUMMARIZATION:
            return TaskConfig(
                first_worker_instruction="""
                [CONTEXT]
                {chunk_text}
                [TASK]
                Create a lengthy summary by incorporating key information from this context.
                Focus on main ideas, findings, and conclusions. Maintain a coherent narrative.
                """,
                worker_instruction="""
                [PREVIOUS SUMMARY]
                {previous_summary}
                [CONTEXT]
                {chunk_text}
                [TASK]
                Refine the existing summary by incorporating key information from this context.
                Focus on main ideas, findings, and conclusions. Maintain a coherent narrative.
                """,
                manager_instruction="""
                [SUMMARY]
                {last_summary}
                [TASK]
                Using all context available, resolve any contradictions and only provide a single lengthy summary using simple, everyday language.
                Format: <summary>...</summary>
                """,
                multi_summary_instruction="""
                [SUMMARIES]
                {last_summary}
                [TASK]
                Integrate all information from the summaries into a single coherent summary.
                Resolve any contradictions and ensure the logical flow of ideas.
                """
            )

class ProcessingMode(Enum):
    LTR = "left_to_right"
    RTL = "right_to_left"
    RAND = "random"

@dataclass
class TextChunk:
    text: str
    chunk_id: str = "-1"
    left_child: Optional['TextChunk'] = None
    right_child: Optional['TextChunk'] = None
    depth: int = 0
    token_count: int = 0

@dataclass
class ChainOfAgentsConfig:
    worker_context_window: int = 16384
    manager_context_window: int = 16384
    max_tokens_per_chunk: int = 8192
    processing_mode: ProcessingMode = ProcessingMode.LTR
    task_type: TaskType = TaskType.QA
    split_threshold: float = 1.1  # Threshold for splitting chunks based on priority score
    sensitivity_curve: float = 0.3 
    min_tokens_to_split: int = 512 

class Model(LLM):
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
    api_url: str = Field(
        default="http://localhost:8000/v1/completions",
        description="URL of the TabbyAPI endpoint"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication with TabbyAPI"
    )
    context_window: int = Field(
        default=16384,
        description="Context window for the worker/manager"
    )
    _client: httpx.AsyncClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = httpx.AsyncClient()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            use_fast=True
        )
        
    def format_prompt(self, instruction: str) -> str:
        instruction = instruction.strip()
        if self.instruction_format.lower() == "llama":
            return (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                f"{instruction}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        return f"<s>[INST]{instruction}[/INST]"


    async def complete(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens_response,
            "temperature": 0.0,
            "temperature_last": False,
            "dynamic_temperature": False,
            "dynamic_temperature_low": 0.1,
            "do_sample": True,
            "top_p": 0.9,
            "min_p": 0.0,
            "top_k": 0,
            "typical_p": 1.0,
            "tfs": 1.0,
            "top_a": 0.0,
            "repetition_penalty": 1.2,
            "min_new_tokens": 200,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "early_stopping": False,
            "seed": 0,
            "add_bos_token": True,
            "truncation_length": self.context_window,
            "ban_eos_token": False,
            "skip_special_tokens": True
        }
        
        print(f"Making request to {self.api_url}")
        print(f"Payload size: {len(str(payload))} characters")
    
        try:
            print("Initiating API request...")
            response = await self._client.post(
                f"{self.api_url}/v1/completions",
                json=payload,
                headers=headers,
                timeout=900.0
            )
            print(f"Request completed with status code: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            print("Response received and parsed successfully")
            return result['choices'][0]['text'].strip()
            
        except httpx.ReadTimeout as e:
            print(f"Read timeout error occurred: {str(e)}")
            print(f"The server took too long to send the response")
            raise Exception(f'API read timeout: {str(e)}')
            
        except httpx.ConnectTimeout as e:
            print(f"Connection timeout error occurred: {str(e)}")
            print(f"Could not establish connection to the server")
            raise Exception(f'API connection timeout: {str(e)}')
            
        except httpx.RequestError as e:
            print(f"Request error: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Request details: {e.request.url if e.request else 'No request info'}")
            raise Exception(f'API request error: {str(e)}')
            
        except httpx.HTTPStatusError as e:
            print(f"HTTP error {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            print(f"Request URL: {e.request.url}")
            print(f"Request headers: {e.request.headers}")
            raise Exception(f'API HTTP error: {str(e)}')
            
        except KeyError as e:
            print(f"Missing key in response: {str(e)}")
            print(f"Full response: {result}")
            raise Exception(f'API response error: {str(e)}')
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise Exception(f'API error: {str(e)}')

    async def close(self):
        """Close the HTTP client explicitly"""
        await self._client.aclose()
    
    @property
    def tokenizer(self):
        return self._tokenizer

class WorkerAgent(Assistant):    
    def __init__(self, llm: Model, chunk_id: str):  # chunk_id is now a string
        super().__init__(
            name=f"worker_{chunk_id}",
            llm=llm
        )

    async def process_chunk(
        self,
        chunk: TextChunk,
        previous_summary: Optional[str], 
        query: Optional[str],
        instruction: str
    ) -> str:
        formatted_instruction = instruction.format(
            chunk_text=chunk.text,
            previous_summary=previous_summary or "No previous summary",
            question=query
        )
        
        prompt = self.llm.format_prompt(formatted_instruction)
                    
        print(f"\n=== Worker Agent {chunk.chunk_id} Processing ===")
        response = await self.llm.complete(prompt)
        print(f"Worker Response:\n{response}\n")
        
        return response

class ManagerAgent(Assistant):    
    def __init__(self, llm: Model):
        super().__init__(
            name="manager",
            llm=llm
        )
    
    @staticmethod
    def remove_answer_tags(response: str, task_type: TaskType) -> str:
        """Remove answer tags from response if they exist and if task type is QA."""
        if task_type == TaskType.QA:
            # Try to find content within <answer> tags
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                
                # Find any text after the closing </answer> tag
                after_tag_match = re.search(r'</answer>(.*)', response, re.IGNORECASE | re.DOTALL)
                result = answer_content
                if after_tag_match and after_tag_match.group(1).strip():
                    # Combine the answer content with the text after the tag
                    result = f"{answer_content} {after_tag_match.group(1).strip()}"
                
                # Remove any remaining <answer> or </answer> tags
                result = re.sub(r'</?answer>', '', result, flags=re.IGNORECASE)
                return result
        elif task_type == TaskType.SUMMARIZATION:
            # Try to find content within <summary> tags
            summary_match = re.search(r'<summary>(.*?)</summary>', response, re.IGNORECASE | re.DOTALL)
            if summary_match:
                summary_content = summary_match.group(1).strip()
                
                # Find any text after the closing </summary> tag
                after_tag_match = re.search(r'</summary>(.*)', response, re.IGNORECASE | re.DOTALL)
                result = summary_content
                if after_tag_match and after_tag_match.group(1).strip():
                    # Combine the summary content with the text after the tag
                    result = f"{summary_content} {after_tag_match.group(1).strip()}"
                
                # Remove any remaining <summary> or </summary> tags
                result = re.sub(r'</?summary>', '', result, flags=re.IGNORECASE)
                return result
                    
        # If no tags are found, remove any leftover tags and return
        result = re.sub(r'</?answer>|</?summary>', '', response, flags=re.IGNORECASE)
        return result.strip()
    
    async def generate_response(
        self,
        summary: str,
        query: Optional[str],
        instruction: str,
        task_type: TaskType
    ) -> str:
        formatted_instruction = instruction.format(
            last_summary=summary,
            question=query
        )
        
        prompt = self.llm.format_prompt(formatted_instruction)
        
        print("\n=== Manager Agent Processing ===")
        response = await self.llm.complete(prompt)
        return ManagerAgent.remove_answer_tags(response, task_type)

class ChunkProcessor:
    def __init__(self, llm: Model, config: ChainOfAgentsConfig):
        self.llm = llm
        self.config = config
        self.vectorizer = TfidfVectorizer()
        self.mean_score: float = 0.0
        self.std_score: float = 0.0

    def calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy for text diversity"""
        words = text.split()
        total_words = len(words)
        if total_words == 0:
            return 0.0
            
        _, counts = np.unique(words, return_counts=True)
        probabilities = counts / total_words
        return -np.sum(probabilities * np.log2(probabilities))

    def calculate_priority_score(self, chunk: TextChunk, query: str) -> float:
        entropy = self.calculate_entropy(chunk.text)
        if not query:
            return entropy
            
        vectors = self.vectorizer.fit_transform([chunk.text, query])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        
        return (0.7 * entropy * np.log1p(entropy)) + (0.3 * similarity ** 2)

    def subsplit_chunk(self, chunk: TextChunk) -> TextChunk:
        """Split chunk into binary tree of sub-chunks"""
        if not self._needs_split(chunk, None) or chunk.depth >= 3:  # MAX_DEPTH = 3
            return chunk
            
        sentences = chunk.text.split('. ')
        mid = len(sentences) // 2
        
        left_text = '. '.join(sentences[:mid]) + '.'
        right_text = '. '.join(sentences[mid:]) + '.'
        
        left_chunk = TextChunk(
            text=left_text,
            chunk_id=f"{chunk.chunk_id}.1",
            depth=chunk.depth + 1
        )
        right_chunk = TextChunk(
            text=right_text,
            chunk_id=f"{chunk.chunk_id}.2",
            depth=chunk.depth + 1
        )
        
        chunk.left_child = self.subsplit_chunk(left_chunk)
        chunk.right_child = self.subsplit_chunk(right_chunk)
        
        return chunk

    def get_ordered_chunks(self, root_chunk: TextChunk) -> List[TextChunk]:
        """Get chunks in processing order (deepest first, left to right)"""
        chunks = []
        
        def traverse(chunk: TextChunk):
            if not chunk:
                return
                
            if not chunk.left_child and not chunk.right_child:
                chunks.append(chunk)
                return
                
            if chunk.left_child:
                traverse(chunk.left_child)
            if chunk.right_child:
                traverse(chunk.right_child)
            chunks.append(chunk)
        
        traverse(root_chunk)
        return chunks

    def _needs_split(self, chunk: TextChunk, query: Optional[str], min_tokens: int = 512) -> bool:
        approx_tokens = chunk.token_count
        MIN_TOKENS = min_tokens
        
        if approx_tokens < MIN_TOKENS:
            print(f"Chunk {chunk.chunk_id} below minimum token threshold ({approx_tokens} < {MIN_TOKENS})")
            return False
            
        score = self.calculate_priority_score(chunk, query) if query else self.calculate_entropy(chunk.text)
        
        distribution_factor = 1 - np.exp(-self.config.sensitivity_curve * self.std_score)
        dynamic_threshold = self.mean_score + (self.config.split_threshold * distribution_factor * self.std_score)
        needs_split = score > dynamic_threshold
    
        print(f"Chunk {chunk.chunk_id} evaluation:")
        print(f"  Score: {score:.2f}")
        print(f"  Threshold: {dynamic_threshold:.2f} (mean {self.mean_score:.2f} + {self.config.split_threshold}σ)")
        
        return needs_split
    
    def _split_into_chunks(self, text: str) -> List[TextChunk]:
        # Get actual token count using the tokenizer
        tokens = self.llm.tokenizer.encode(text)
        total_tokens = len(tokens)
        del tokens  # Free memory
    
        print(f"\n=== Document Token Count ===")
        print(f"The document contains {total_tokens} tokens")
        
        if total_tokens <= self.config.max_tokens_per_chunk:
            print("Document not split because total_tokens <= max_tokens_per_chunk")
            return [TextChunk(text=text, chunk_id="0", token_count=total_tokens)]
    
        segments = text.split('\n')
        if len(segments) <= 1:
            # Split by sentences using compiled regex
            segments = SENTENCE_SPLIT_PATTERN.split(text)
            print(f"Re-split into {len(segments)} sentence segments")
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for segment in segments:
            # Use actual token count instead of approximation
            segment_tokens = len(self.llm.tokenizer.encode(segment))
            
            if current_size + segment_tokens > self.config.max_tokens_per_chunk:
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    # Calculate actual token count for the complete chunk
                    chunk_token_count = len(self.llm.tokenizer.encode(chunk_text))
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_id=str(chunk_id),
                        token_count=chunk_token_count  # Set the token count here
                    ))
                    chunk_id += 1
                    current_chunk = [segment]
                    current_size = segment_tokens
                else:
                    # Calculate actual token count for the complete chunk
                    chunk_token_count = len(self.llm.tokenizer.encode(segment))
                    chunks.append(TextChunk(
                        text=segment,
                        chunk_id=str(chunk_id),
                        token_count=chunk_token_count  # Set the token count here
                    ))
                    chunk_id += 1
            else:
                current_chunk.append(segment)
                current_size += segment_tokens
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            # Calculate actual token count for the final chunk
            chunk_token_count = len(self.llm.tokenizer.encode(chunk_text))
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_id=str(chunk_id),
                token_count=chunk_token_count  # Set the token count here
            ))
        
        print(f"Document split into {len(chunks)} chunks")
        return chunks

class ChainOfAgents:
    def __init__(
        self,
        llm: Model,
        chunks: List[TextChunk] = None,
        config: ChainOfAgentsConfig = ChainOfAgentsConfig(),
    ):
        self.llm = llm
        self.config = config
        self.chunks = chunks or []
        self.is_first_chunk: bool = True
        self.task_config = TaskFactory.create_task(config.task_type)
        self.chunk_processor = ChunkProcessor(llm, config)

    def _get_chunk_order(self, num_chunks: int) -> List[int]:
        if self.config.processing_mode == ProcessingMode.LTR:
            return list(range(num_chunks))
        elif self.config.processing_mode == ProcessingMode.RTL:
            return list(range(num_chunks - 1, -1, -1))
        else:
            return random.sample(range(num_chunks), num_chunks)

    async def process(
        self,
        query: Optional[str] = None,
    ) -> str:
        current_summary = None
        self.is_first_chunk = True
        
        initial_scores = []
        print("\n=== Initial Chunk Scores ===")
        for chunk in self.chunks:
            score = self.chunk_processor.calculate_priority_score(chunk, query) if query else self.chunk_processor.calculate_entropy(chunk.text)
            initial_scores.append(score)
            print(f"Chunk {chunk.chunk_id}: {score:.2f}")
    
        self.chunk_processor.mean_score = np.mean(initial_scores) if initial_scores else 0.0
        self.chunk_processor.std_score = np.std(initial_scores) if initial_scores else 0.0
        
        q3 = np.percentile(initial_scores, 75)
        iqr = q3 - np.percentile(initial_scores, 25)
        self.chunk_processor.std_score = min(self.chunk_processor.std_score, iqr/1.35)
        
        print(f"\nDistribution stats:")
        print(f"Mean: {self.chunk_processor.mean_score:.2f}")
        print(f"Std: {self.chunk_processor.std_score:.2f}")
    
        for chunk in self.chunks:
            current_summary = await self._process_sub_chunk(chunk, current_summary, query)
        
        return current_summary

    async def _process_sub_chunk(self, chunk: TextChunk, current_summary: str, query: Optional[str]) -> str:
        MAX_DEPTH = 3
        depth = len(chunk.chunk_id.split('.')) - 1
        
        # Get actual token count for the chunk
        chunk_tokens = len(self.llm.tokenizer.encode(chunk.text))
        chunk.token_count = chunk_tokens
        
        # Check splitting criteria
        should_not_split = (
            depth >= MAX_DEPTH or 
            chunk_tokens < self.config.min_tokens_to_split or
            not self.chunk_processor._needs_split(chunk, query, self.config.min_tokens_to_split)
        )
        
        if not should_not_split:
            # Look ahead: Check if splitting would create valid child chunks
            # Improved sentence splitting with better regex
            sentences = SENTENCE_SPLIT_PATTERN.split(chunk.text)
            
            # Make sure we have enough sentences to make splitting worthwhile
            if len(sentences) >= 4:
                mid = len(sentences) // 2
                left_text = ' '.join(sentences[:mid])
                right_text = ' '.join(sentences[mid:])
                
                # Verify that both resulting chunks have reasonable sizes using actual token counts
                left_tokens = len(self.llm.tokenizer.encode(left_text))
                right_tokens = len(self.llm.tokenizer.encode(right_text))
            
                # Only proceed with splitting if both resulting chunks are large enough
                if left_tokens >= self.config.min_tokens_to_split and right_tokens >= self.config.min_tokens_to_split:
                    left_chunk = TextChunk(
                        text=left_text,
                        chunk_id=f"{chunk.chunk_id}.1",
                        depth=depth + 1,
                        token_count=left_tokens
                    )
                    right_chunk = TextChunk(
                        text=right_text,
                        chunk_id=f"{chunk.chunk_id}.2",
                        depth=depth + 1,
                        token_count=right_tokens
                    )
                    
                    chunk.left_child = left_chunk
                    chunk.right_child = right_chunk
                    
                    # Process left chunk first, then right chunk
                    print(f"  Decision: SPLIT")
                    summary = await self._process_sub_chunk(left_chunk, current_summary, query)
                    final_summary = await self._process_sub_chunk(right_chunk, summary, query)
                    
                    return final_summary
            
            # If we get here, splitting would create invalid child chunks, so we don't split
            should_not_split = True

        # Process the chunk without splitting
        if should_not_split:
            print(f"  Decision: NOT SPLIT")
            worker = WorkerAgent(self.llm, chunk.chunk_id)
            instruction = self.task_config.first_worker_instruction if self.is_first_chunk else self.task_config.worker_instruction
            self.is_first_chunk = False
            return await worker.process_chunk(
                chunk=chunk,
                previous_summary=current_summary,
                query=query,
                instruction=instruction
            )

async def main(
    worker_context_window: int = 16384,
    manager_context_window: int = 16384,
    max_tokens_per_chunk: int = 4096,
    max_tokens_response: int = 1024,
    instruction_format: str = "llama",
    model: str = "NousResearch/Meta-Llama-3.1-8B-Instruct", # For the tokenizer
    task_type: TaskType = TaskType.QA,
    query: str = None,
    min_tokens_to_split: int = 512
):
    llm = Model(
        max_tokens_response=max_tokens_response,
        context_window=worker_context_window,
        instruction_format=instruction_format,
        model=model,
        api_url="https://15ef-34-105-107-13.ngrok-free.app"
    )
    
    modes = [
        ProcessingMode.LTR,
        #ProcessingMode.RTL
        #*[ProcessingMode.RAND]*5
    ]
    
    txt_url = "https://openreview.net/pdf?id=LuCLf4BJsr"
    save_path = "/kaggle/working/chain_of_agent.pdf"

    !wget -O {save_path} {txt_url}
    
    text = extract_text(save_path)
    #query = "According to the paper what's the best context size window for the agents?'"
    query = "List all the datasets used in the paper."
    
    config = ChainOfAgentsConfig(
        worker_context_window=worker_context_window,
        manager_context_window=manager_context_window,
        max_tokens_per_chunk=max_tokens_per_chunk,
        task_type=task_type,
        min_tokens_to_split=min_tokens_to_split
    )
    temp_processor = ChunkProcessor(llm, config)
    initial_chunks = temp_processor._split_into_chunks(text)
    
    final_summaries = []
    for mode in modes:
        print(f"\n=== Processing with {mode.value} mode ===")
        config.processing_mode = mode
        coa = ChainOfAgents(llm, initial_chunks, config)
        chain_summary = await coa.process(query=query if task_type == TaskType.QA else None)
        final_summaries.append(chain_summary)
    
    manager = ManagerAgent(llm)
    if len(final_summaries) > 1:
        combined_summaries = "\n\n".join(
            f"Summary n°{i+1}:\n{summary}"
            for i, summary in enumerate(final_summaries))
        final_response = await manager.generate_response(
            summary=combined_summaries,
            query=query if task_type == TaskType.QA else None,
            instruction=coa.task_config.multi_summary_instruction,
            task_type=task_type
        )
    else:
        final_response = await manager.generate_response(
            summary=final_summaries[0],
            query=query if task_type == TaskType.QA else None,
            instruction=coa.task_config.manager_instruction,
            task_type=task_type
        )
    
    print("\n=== Final Manager Response ===")
    print(final_response)

if __name__ == "__main__":
    nest_asyncio.apply()
    try:
       asyncio.run(main())
    finally:
       gc.collect()
       torch.cuda.empty_cache()
       print("GPU memory has been cleared.")
