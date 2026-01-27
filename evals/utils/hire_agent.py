import asyncio
from typing import List, Dict, Any, Optional, final
import openai
from openai import AsyncOpenAI, OpenAI
import json
import jsonschema
import os
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from logger import get_logger
log = get_logger("main")
load_dotenv()

# ---------------- Hire Agent ----------------
class Agent:
    """
    Asynchronous wrapper for the OpenAI Chat API.
    Allows custom system prompt, user prompt, and optional JSON schema validation.
    """
    def __init__(
        self, 
        system_prompt: str, 
        output_schema: Optional[Dict] = None,
        model: Optional[str] = 'gpt-4.1',
    ):
        self.model = model
        self.messages: List[Dict] = [{'role':'system','content':system_prompt + (f"""ONLY output a valid JSON object conforming to the schema. The json schema is given below: {json.dumps(output_schema, ensure_ascii=False, indent = 1)}""" if output_schema else "")}]
        self.output_schema = output_schema or None
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key = api_key)

    def get_agent_config(self) -> Dict[str, Any]:
        """Return the current configuration for the agent."""
        config = {
            "model": self.model,
            "model_config": self.config.__dict__,
            "system_prompt": self.messages[0],
            "history": [msg for msg in self.messages[1:]]
        }
        return config

    async def ask(self, user_input, with_full_msg: bool = False, with_history: bool = False, temperature = None, reasoning_effort = None) -> Any:
        """
        Send a user message asynchronously and return the model's response.
        If an output schema is defined, the response will be parsed and validated.
        """
        self.messages.append({'role':'user','content':user_input})
        if not with_history:
            chat_messages = [self.messages[0], self.messages[-1]]
        else:
            # max_history_msgs = 1 + 5 * 2  # keep up to 5 rounds (system + last 10 messages)
            # if len(self.messages) > max_history_msgs:
            #     omitted_msg = {
            #         "role": "system",
            #         "content": f"...omitted {len(self.messages) - max_history_msgs} earlier dialogue turns for brevity..."
            #     }
            #     chat_messages = [self.messages[0], omitted_msg] + self.messages[-(max_history_msgs - 1):]
            #     # print(chat_messages)
            # else:
            chat_messages = [self.messages[0]] + self.messages[1:]

        if with_full_msg:
            chat_messages = user_input
        
        kwargs = {
            'model': self.model,
            'messages': chat_messages
        }
        # print(chat_messages)
        if self.output_schema:
            kwargs['response_format'] = {"type": "json_object"}
        if temperature:
            kwargs['temperature'] = temperature
        if reasoning_effort:
            kwargs['reasoning_effort'] = reasoning_effort
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(e)
            raise
        # print(resp)
        from agents.base import GLOBAL_TOKEN_TRACKER
        if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage"):
            # print(resp.usage)
            await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)

        answer = resp.choices[0].message.content
        if not answer or answer.strip() == "":
            answer = "Error occur when generating answer."
            print('Error occur when generating answer.')
        
        self.messages.append({'role':'assistant','content':answer})

        # --- Parse according to output schema if provided ---
        if self.output_schema:
            try:
                # print(answer)
                parsed = json.loads(answer)
                jsonschema.validate(instance=parsed, schema=self.output_schema)
                # print(parsed)
                return parsed
            except json.JSONDecodeError:
                log.warning(f"fail to parse: {answer}")
                print(f"fail to parse: {answer}")
                return {"error": "Model output is not valid JSON.", "raw_output": answer}
            except jsonschema.ValidationError as e:
                log.warning(f"fail to parse: {answer}")
                print(f"fail to parse: {answer}")
                return {"error": f"Output does not match schema: {e.message}", "raw_output": answer}

        return answer  # Return raw string if no schema is provided

class Embedding:
    """
    Async embedding manager for computing single or batch embeddings,
    with optional similarity calculation. Supports token tracking.
    """

    def __init__(self, model: str = "text-embedding-3-small", retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize embedding model parameters.
        :param model: embedding model name
        :param retries: number of retry attempts for API call
        :param retry_delay: delay between retries in seconds
        """
        self.model = model
        self.retries = retries
        self.retry_delay = retry_delay
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key = api_key)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Safe synchronous entry point that works both inside and outside async loops.
        """
        if not texts:
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop → safe to use asyncio.run
            return asyncio.run(self.get_batch_embeddings(texts))

        # Already inside an async loop → offload to background thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: asyncio.run(self.get_batch_embeddings(texts)))
            return future.result()

    async def get_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for a single text string asynchronously.
        Updates global token tracker automatically.
        """
        attempt = 0
        final_error = ''
         
        assert isinstance(text, str)

        while attempt < self.retries:
            try:
                resp = await self.client.embeddings.create(model=self.model, input=text)

                # Update global token tracker
                from agents.base import GLOBAL_TOKEN_TRACKER
                if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage"):
                    await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)

                return resp.data[0].embedding
            except Exception as e:
                attempt += 1
                log.error(f"[EmbeddingManager] Attempt {attempt} failed: {e}")
                final_error = e
                await asyncio.sleep(self.retry_delay)
        raise RuntimeError(f"Failed to get embedding after {self.retries} attempts, with error: {final_error}")

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a batch of texts asynchronously.
        Updates global token tracker automatically.
        """
        attempt = 0
        final_error = ''
        while attempt < self.retries:
            try:
                resp = await self.client.embeddings.create(model=self.model, input=texts)

                # Update global token tracker
                from agents.base import GLOBAL_TOKEN_TRACKER
                if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage"):
                    await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)

                return [item.embedding for item in resp.data]
            except Exception as e:
                attempt += 1
                log.error(f"[EmbeddingManager] Batch attempt {attempt} failed: {e}")
                final_error = e
                await asyncio.sleep(self.retry_delay)
        log.error(f"Failed to get batch embeddings after {self.retries} attempts. With error: {final_error}")
        raise RuntimeError(f"Failed to get batch embeddings after {self.retries} attempts. With error: {final_error}")

    @staticmethod
    async def compute_similarity(emb1: List[float], emb2: List[float], metric: str = "cosine") -> float:
        """
        Asynchronously compute cosine similarity between two embeddings.
        """
        if metric == "cosine":
            # Offload to a thread to avoid blocking the event loop
            return await asyncio.to_thread(lambda: 1 - cosine(emb1, emb2))
        else:
            log.error(f"Unsupported similarity metric: {metric}")
            raise ValueError(f"Unsupported similarity metric: {metric}")

    @staticmethod
    async def compute_one_to_group_similarity(
        emb: List[float],
        group_emb: List[List[float]],
        metric: str = "cosine"
    ) -> List[float]:
        """
        Asynchronously compute similarity between one embedding and a group of embeddings.
        """
        if not group_emb:
            return []
        # Launch similarity computations concurrently
        tasks = [Embedding.compute_similarity(emb, g_emb, metric=metric) for g_emb in group_emb]
        return await asyncio.gather(*tasks)

    def embed_query(self, text: str) -> List[float]:
        """
        For Chroma compatibility: used when querying the collection.
        Runs the async embedding function in a blocking way.
        """
        # Chroma expects a sync call
        return self([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        For Chroma compatibility: used when adding documents to the collection.
        """
        return self(texts)

