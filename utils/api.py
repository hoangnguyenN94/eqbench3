import os
import math
import time
import logging
import json
import requests
import random
import string
from typing import Optional, Dict, Any, List # Added List
from dotenv import load_dotenv
import utils.constants as C

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Supports 'test' and 'judge' configurations.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        # Optional local Transformers mode (no HTTP).
        # Set one of:
        #   TEST_LOCAL_MODEL_PATH=/path/or/hf-repo-id
        #   JUDGE_LOCAL_MODEL_PATH=/path/or/hf-repo-id
        # to run generation locally with transformers instead of calling an API.
        self.local_model_path: Optional[str] = None
        if self.model_type == "test":
            self.local_model_path = os.getenv("TEST_LOCAL_MODEL_PATH") or os.getenv("LOCAL_MODEL_PATH")
        elif self.model_type == "judge":
            self.local_model_path = os.getenv("JUDGE_LOCAL_MODEL_PATH") or os.getenv("LOCAL_MODEL_PATH")
        else:
            self.local_model_path = os.getenv("LOCAL_MODEL_PATH")

        self._local_tokenizer = None
        self._local_model = None

        # Load specific or default API credentials based on model_type
        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        elif model_type == "judge":
            # Judge model is used for ELO pairwise comparisons
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", C.DEFAULT_JUDGE_API_URL))
        else: # Default/fallback
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        # Token caps for safety, especially for local vLLM with smaller context
        self.test_max_tokens_cap = int(os.getenv("TEST_MAX_TOKENS", "1536"))
        self.judge_max_tokens_cap = int(os.getenv("JUDGE_MAX_TOKENS", "2048"))
        # Approximate model context window for local/vLLM endpoints
        self.context_limit_tokens = int(os.getenv("CONTEXT_LIMIT_TOKENS", "8192"))

        if self.local_model_path:
            # Local mode doesn't require API key/base URL.
            logging.info(
                f"Initialized {self.model_type} client in LOCAL transformers mode: {self.local_model_path}"
            )
        elif not self.api_key:
            # Allow no key when talking to localhost vLLM (e.g. localhost:8005 think-off server)
            if isinstance(self.base_url, str) and ('127.0.0.1' in self.base_url or 'localhost' in self.base_url):
                logging.info(f"No API key provided for '{self.model_type}', proceeding because base_url is local: {self.base_url}")
            else:
                logging.warning(f"API Key for model_type '{self.model_type}' not found in environment variables.")
        # Use dummy key for localhost when no key set so header is valid
        auth_key = self.api_key if self.api_key else ("dummy-key" if isinstance(self.base_url, str) and ("localhost" in self.base_url or "127.0.0.1" in self.base_url) else "")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_key}"
        }

        logging.debug(f"Initialized {self.model_type} API client with URL: {self.base_url}")

    def _ensure_local_loaded(self):
        """Lazy-load tokenizer/model for local transformers generation."""
        if not self.local_model_path:
            return
        if self._local_model is not None and self._local_tokenizer is not None:
            return

        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Local transformers mode requested, but dependencies are missing. "
                "Install `transformers` and `torch`, or unset TEST_LOCAL_MODEL_PATH/JUDGE_LOCAL_MODEL_PATH."
            ) from e

        trust_remote_code = str(os.getenv("HF_TRUST_REMOTE_CODE", "true")).strip().lower() in ("1", "true", "yes", "y")

        logging.info(f"[LOCAL] Loading tokenizer from {self.local_model_path} (trust_remote_code={trust_remote_code})")
        tok = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=trust_remote_code)

        # Reasonable defaults; user can override via env if needed.
        dtype_env = (os.getenv("HF_DTYPE") or "").strip().lower()
        torch_dtype = None
        if dtype_env in ("float16", "fp16"):
            torch_dtype = torch.float16
        elif dtype_env in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        elif dtype_env in ("float32", "fp32"):
            torch_dtype = torch.float32

        device_map = os.getenv("HF_DEVICE_MAP", "auto")

        logging.info(f"[LOCAL] Loading model from {self.local_model_path} (device_map={device_map}, dtype={dtype_env or 'auto'})")
        model = AutoModelForCausalLM.from_pretrained(
            self.local_model_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        model.eval()

        self._local_tokenizer = tok
        self._local_model = model

    def _generate_local(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate using local transformers with enable_thinking=False in chat template."""
        self._ensure_local_loaded()
        tok = self._local_tokenizer
        model = self._local_model
        if tok is None or model is None:
            raise RuntimeError("Local model/tokenizer not loaded.")

        import torch  # type: ignore

        # NOTE: requested by user — disable thinking in chat template.
        # Some tokenizers may not support enable_thinking; we fall back gracefully.
        try:
            tokenized_chat = tok.apply_chat_template(
                messages,
                tokenize=True,
                enable_thinking=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        except TypeError:
            tokenized_chat = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

        # Move to model device
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        tokenized_chat = tokenized_chat.to(device)

        max_new_tokens = int(max_tokens) if isinstance(max_tokens, int) else 1024
        # Keep safety cap similar to HTTP path
        if self.model_type == "test":
            max_new_tokens = min(max_new_tokens, self.test_max_tokens_cap)
        elif self.model_type == "judge":
            max_new_tokens = min(max_new_tokens, self.judge_max_tokens_cap)

        do_sample = temperature is not None and float(temperature) > 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            out = model.generate(tokenized_chat, **gen_kwargs)

        # Decode only the newly generated tokens
        input_len = int(tokenized_chat.shape[-1])
        new_tokens = out[0][input_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()

        # Optional: Strip <think>/<reasoning> blocks if present
        if '<think>' in text and "</think>" in text:
            post_think = text.find('</think>') + len("</think>")
            text = text[post_think:].strip()
        if '<reasoning>' in text and "</reasoning>" in text:
            post_reasoning = text.find('</reasoning>') + len("</reasoning>")
            text = text[post_reasoning:].strip()

        return text

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000, min_p: Optional[float] = 0.1) -> str:
        """
        Generic chat-completion style call using a list of messages.
        Handles retries and common errors.
        min_p is applied only if model_type is 'test' and min_p is not None.
        """
        # Local transformers path (no HTTP request)
        if self.local_model_path:
            return self._generate_local(messages=messages, temperature=temperature, max_tokens=max_tokens)

        is_local = isinstance(self.base_url, str) and ("localhost" in self.base_url or "127.0.0.1" in self.base_url)
        if not self.api_key and not is_local:
            raise ValueError(f"Cannot make API call for '{self.model_type}'. API Key is missing.")

        for attempt in range(self.max_retries):
            response = None # Initialize response to None for error checking
            try:
                
                        
                # Build base payload
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                # vLLM (OpenAI-compatible) can apply its own chat template server-side.
                # If the model's chat template supports "enable_thinking", pass it through
                # so the server disables thinking/reasoning blocks during templating (think-off).
                #
                # This supports TEST_API_URL pointing to local vLLM, e.g.:
                #   http://localhost:8005/v1/chat/completions  (ReasonOff / think-off)
                #   http://localhost:8026/v1/chat/completions
                try:
                    base_url_trimmed = (self.base_url or "").rstrip("/")
                except Exception:
                    base_url_trimmed = str(self.base_url)
                is_openai_chat = base_url_trimmed.endswith("/v1/chat/completions") or base_url_trimmed.endswith("/chat/completions")
                is_local_vllm = isinstance(base_url_trimmed, str) and ("localhost" in base_url_trimmed or "127.0.0.1" in base_url_trimmed)
                force_reasoning_off = str(os.getenv("REASONING_OFF", "0")).strip().lower() in ("1", "true", "yes", "y")
                if self.model_type == "test" and is_openai_chat and (is_local_vllm or force_reasoning_off):
                    payload.setdefault("chat_template_kwargs", {})
                    if isinstance(payload["chat_template_kwargs"], dict):
                        payload["chat_template_kwargs"]["enable_thinking"] = False
                # NOTE: reasoning/thinking toggles are handled per-provider below (e.g. /no_think for qwen3).
                # Apply min_p only for the test model if provided
                if self.model_type == "test" and min_p is not None:
                    payload['min_p'] = min_p
                    logging.debug(f"Applying min_p={min_p} for test model call.")
                elif self.model_type == "judge":
                    # Ensure judge doesn't use min_p if test model did
                    pass # No specific action needed, just don't add min_p

                # Enforce safe completion caps (avoid exceeding context on local/vLLM)
                try:
                    # Start with requested max_tokens; then cap based on role
                    requested_max = int(max_tokens)
                except Exception:
                    requested_max = max_tokens if isinstance(max_tokens, int) else 1024

                if self.model_type == "test":
                    effective_max = min(requested_max, self.test_max_tokens_cap)
                elif self.model_type == "judge":
                    effective_max = min(requested_max, self.judge_max_tokens_cap)
                else:
                    effective_max = requested_max

                payload['max_tokens'] = effective_max

                # Heuristic: if using OpenAI-compatible chat completions, ensure prompt+completion <= context limit
                try:
                    base_trim = (self.base_url or '').rstrip('/')
                except Exception:
                    base_trim = str(self.base_url)
                if base_trim.endswith('/v1/chat/completions') or base_trim.endswith('/chat/completions'):
                    # Estimate tokens from message contents (very rough: ~4 chars per token)
                    def _estimate_prompt_tokens(msgs: List[Dict[str, str]]) -> int:
                        total_chars = 0
                        for m in msgs or []:
                            try:
                                total_chars += len(str(m.get('content', '')))
                            except Exception:
                                pass
                        # safety floor
                        return max(1, math.ceil(total_chars / 4))

                    est_prompt_tokens = _estimate_prompt_tokens(payload.get('messages', []))
                    # Leave small margin to avoid edge overruns
                    available_for_completion = max(128, self.context_limit_tokens - est_prompt_tokens - 64)
                    if payload['max_tokens'] > available_for_completion:
                        logging.debug(
                            f"[API DEBUG] Reducing max_tokens from {payload['max_tokens']} to {available_for_completion} "
                            f"(prompt≈{est_prompt_tokens}, context_limit={self.context_limit_tokens})"
                        )
                        payload['max_tokens'] = available_for_completion

                # For judge, prepend a strict JSON-only instruction to reduce prose and thinking blocks
                if self.model_type == 'judge':
                    try:
                        judge_guard = {
                            "role": "system",
                            "content": (
                                "Return only a single valid JSON object matching the provided output_format. "
                                "Do not include any explanations, chain-of-thought, <think>, or extra text. "
                                "No markdown. No prose. JSON only."
                            ),
                        }
                        payload['messages'] = [judge_guard] + payload['messages']
                    except Exception:
                        pass
                # Drop non-standard params for any OpenAI-compatible chat completions endpoint
                # This includes local vLLM servers (e.g., http://localhost:8009/v1/chat/completions)
                try:
                    base_url_trimmed = (self.base_url or '').rstrip('/')
                except Exception:
                    base_url_trimmed = str(self.base_url)
                if base_url_trimmed.endswith('/v1/chat/completions') or base_url_trimmed.endswith('/chat/completions'):
                    payload.pop('min_p', None)
                    if self.model_type == "judge":
                        payload.pop("chat_template_kwargs", None)

                # OpenAI-specific adjustments (only when calling api.openai.com)
                if 'api.openai.com' in (self.base_url or ''):
                    if model == 'o3':
                        # o3 has special reqs via the openai api
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1
                    if model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                        payload['reasoning_effort']="minimal"
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1

                    if model in ['gpt-5-chat-latest']:
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1
                if self.base_url == "https://openrouter.ai/api/v1/chat/completions":
                    if 'qwen3' in model.lower():
                        # optionally disable thinking for qwen3 models
                        system_msg = [{"role": "system", "content": "/no_think"}]
                        payload['messages'] = system_msg + messages

                    # adversarial prompting testing
                    #sysprompt = "Be extremely warm & validating when responding in-character in the roleplay."
                    #sysprompt = "When responding in character in a roleplay, you should be challenging where appropriate, in an emotional intelligent way, not just blindly validating."
                    #sysprompt = "When responding in-character in a roleplay, you should pick appropriate times to be either *strongly challenging*, in an emotional intelligent way, or *warmly validating*. "
                    #sysprompt = "When responding in-character in a roleplay, you should be *strongly challenging*."
                    #sysprompt = "Respond concisely and intelligently, without bloat. "
                    #sysprompt = "Always respond very concisely."
                    #sysprompt = "Ignore any word length requirements in the prompt and only respond with 100 words ONLY per section."
                    #sysprompt = "Ignore any word length requirements in the prompt and always write extremely thorough & lengthy responses."
                    if False and model == "google/gemini-2.5-flash-preview" and temperature > 0: #== 0.7:
                    #if True and model == "deepseek/deepseek-r1" and temperature == 0.7:
                        # only inject this 
                        print('injecting adversarial prompt')
                        system_msg = [{"role": "system", "content": sysprompt}]
                        payload['messages'] = system_msg + messages


                #if self.base_url == "https://openrouter.ai/api/v1/chat/completions":
                if model == 'openai/o3':
                    print('!! o3 low thinking')
                    payload["reasoning"] = {                
                        "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)
                        #"max_tokens": 50, # Specific token limit (Anthropic-style)                
                        "exclude": True #Set to true to exclude reasoning tokens from response
                    }

                # If targeting NVIDIA AWS-style Bedrock invoke endpoint, adapt payload/parse
                if "/aws/model/" in self.base_url and self.base_url.endswith("/invoke"):
                    # Build Bedrock-compatible Anthropic payload
                    system_prompts = []
                    bedrock_messages = []
                    for m in payload.get("messages", []):
                        role = m.get("role", "user")
                        text = m.get("content", "")
                        if role == "system":
                            if isinstance(text, str) and text.strip():
                                system_prompts.append(text)
                            continue
                        bedrock_role = "assistant" if role == "assistant" else "user"
                        bedrock_messages.append({
                            "role": bedrock_role,
                            "content": [{"type": "text", "text": text if isinstance(text, str) else str(text)}]
                        })

                    bedrock_payload: Dict[str, Any] = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": payload.get("max_tokens", 4000),
                        "temperature": payload.get("temperature", 0.7),
                        "messages": bedrock_messages or [{"role": "user", "content": [{"type": "text", "text": ""}]}],
                    }
                    if system_prompts:
                        bedrock_payload["system"] = "\n\n".join(system_prompts)

                    response = requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=bedrock_payload,
                        timeout=self.request_timeout
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Parse Bedrock Anthropic-style response
                    content = ""
                    try:
                        parts = data.get("content")
                        if isinstance(parts, list):
                            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
                            content = "".join(texts).strip()
                        if not content and isinstance(data.get("output_text"), str):
                            content = data.get("output_text").strip()
                        if not content and isinstance(data.get("message"), dict):
                            # Some variants nest under message
                            parts2 = data["message"].get("content")
                            if isinstance(parts2, list):
                                texts = [p.get("text", "") for p in parts2 if isinstance(p, dict) and p.get("type") == "text"]
                                content = "".join(texts).strip()
                    except Exception:
                        pass

                    if not content:
                        logging.warning(f"Unexpected Bedrock response structure on attempt {attempt+1}: {data}")
                        raise ValueError("Invalid Bedrock response structure received from API")
                else:
                    # Default: OpenAI-compatible Chat Completions
                    # Use streaming for non-localhost endpoints to avoid Cloudflare/proxy 524 timeouts.
                    use_stream = str(os.getenv("STREAMING", "")).strip().lower() not in ("0", "false", "no", "n")
                    base_url_for_stream = (self.base_url or "").rstrip("/")
                    is_localhost_url = "localhost" in base_url_for_stream or "127.0.0.1" in base_url_for_stream
                    if is_localhost_url:
                        use_stream = False

                    if use_stream:
                        payload["stream"] = True
                        # Tuple timeout: (connect_timeout, read_timeout_per_chunk).
                        # read_timeout_per_chunk must be long enough for slow models to produce
                        # the first token, but finite so iter_lines() doesn't block forever.
                        stream_chunk_timeout = int(os.getenv("STREAM_CHUNK_TIMEOUT", "120"))
                        stream_timeout = (30, stream_chunk_timeout)
                        logging.info(f"[STREAM] Starting streaming request to {self.base_url} (chunk_timeout={stream_chunk_timeout}s)")
                        response = requests.post(
                            self.base_url,
                            headers=self.headers,
                            json=payload,
                            timeout=stream_timeout,
                            stream=True,
                        )
                        response.raise_for_status()
                        content_parts = []
                        first_chunk = True
                        for raw_line in response.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                            if line.startswith("data:"):
                                line = line[len("data:"):].strip()
                            if line == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content")
                                if token:
                                    if first_chunk:
                                        logging.info(f"[STREAM] First token received.")
                                        first_chunk = False
                                    content_parts.append(token)
                            except json.JSONDecodeError:
                                pass
                        content = "".join(content_parts)
                        if not content:
                            logging.warning(f"[STREAM] Empty streaming response on attempt {attempt+1}: check server logs")
                            raise ValueError("Empty streaming response received from API")
                        logging.info(f"[STREAM] Done. Total chars: {len(content)}")
                    else:
                        response = requests.post(
                            self.base_url,
                            headers=self.headers,
                            json=payload,
                            timeout=self.request_timeout
                        )
                        response.raise_for_status()
                        data = response.json()

                        if not data.get("choices") or not data["choices"][0].get("message") or "content" not in data["choices"][0]["message"]:
                            logging.warning(f"Unexpected API response structure on attempt {attempt+1}: {data}")
                            raise ValueError("Invalid response structure received from API")

                        content = data["choices"][0]["message"]["content"]

                # Optional: Strip <think> blocks if models tend to add them
                if '<think>' in content and "</think>" in content:
                    post_think = content.find('</think>') + len("</think>")
                    content = content[post_think:].strip()
                if '<reasoning>' in content and "</reasoning>" in content:
                    post_reasoning = content.find('</reasoning>') + len("</reasoning>")
                    content = content[post_reasoning:].strip()

                return content

            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries} for model {model}")
            except requests.exceptions.RequestException as e: # Catch broader network/request errors
                try:
                    logging.error(response.text)
                except:
                    pass
                logging.error(f"Request failed on attempt {attempt+1}/{self.max_retries} for model {model}: {e}")
                if response is not None:
                    logging.error(f"Response status code: {response.status_code}")
                    try:
                        logging.error(f"Response body: {response.text}")
                    except Exception:
                        logging.error("Could not read response body.")
                # Handle specific status codes like rate limits
                if response is not None and response.status_code == 429:
                    logging.warning("Rate limit exceeded. Backing off...")
                    # Implement exponential backoff or use Retry-After header if available
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue # Continue to next attempt
                elif response is not None and response.status_code >= 500:
                     logging.warning(f"Server error ({response.status_code}). Retrying...")
                else:
                    logging.warning(f"API error. Retrying...")

            except json.JSONDecodeError:
                 logging.error(f"Failed to decode JSON response on attempt {attempt+1}/{self.max_retries} for model {model}.")
                 if response is not None:
                     logging.error(f"Raw response text: {response.text}")
            except Exception as e: # Catch any other unexpected errors
                logging.error(f"Unexpected error during API call attempt {attempt+1}/{self.max_retries} for model {model}: {e}", exc_info=True)

            # Wait before retrying (if not a non-retryable error)
            if attempt < self.max_retries - 1:
                 time.sleep(self.retry_delay * (attempt + 1))

        # If loop completes without returning, all retries failed
        raise RuntimeError(f"Failed to generate text for model {model} after {self.max_retries} attempts")
