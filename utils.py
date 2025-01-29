import os
import json
import logging
import requests
import time as time_module
import random
from functools import wraps
from typing import Dict, Any, Callable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_env():
    """Read environment variables from .env file"""
    env_dict = {}
    with open(file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict

def validate_env_variables():
    """Validate required environment variables are present"""
    required_keys = ["HF_TOKEN"]
    env = get_env()
    for key in required_keys:
        if key not in env or not env[key]:
            raise ValueError(f"Missing required environment variable: {key}")
    return env

def get_api_key(key_name: str) -> str:
    """Get API key from environment variables"""
    env = get_env()
    api_key = env.get(key_name)
    if not api_key:
        raise ValueError(f"{key_name} not found in environment file")
    return api_key

def with_retries(max_retries=3, delay=1):
    """Decorator to retry a function with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    logger.warning(f"Attempt {retries} failed, retrying in {delay * retries} seconds: {str(e)}")
                    time_module.sleep(delay * retries)
            return None
        return wrapper
    return decorator

def get_bert_similarity(response: str, ground_truth: str) -> float:
    """Calculate semantic similarity between response and ground truth"""
    global bert_model
    if 'bert_model' not in globals():
        logger.info("Loading BERT model for similarity calculation...")
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Check for None or empty responses
    if response is None or ground_truth is None:
        logger.warning("Received None response or ground truth, returning 0.0 similarity")
        return 0.0
        
    # Convert to string and handle empty strings
    response = str(response).strip()
    ground_truth = str(ground_truth).strip()
    if not response or not ground_truth:
        logger.warning("Empty response or ground truth after stripping, returning 0.0 similarity")
        return 0.0
    
    try:
        query_embedding = bert_model.encode(response, convert_to_tensor=True)
        text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)
        return cosine_score.item()
    except Exception as e:
        logger.error(f"Error calculating BERT similarity: {str(e)}")
        logger.error(f"Response: {response}")
        logger.error(f"Ground truth: {ground_truth}")
        return 0.0

def unique_path(path: str, i: int = 0) -> str:
    """Generate a unique file path by appending a number if the file already exists"""
    if os.path.exists(path):
        return unique_path(path + "_" + str(i), i + 1)
    return path

def get_bnb_config():
    """Get BitsAndBytes config for model quantization"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

def load_quantized_model(model_name: str, hf_token: str = None):
    """Load a quantized model from HuggingFace"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    
    return tokenizer, model

@with_retries()
def generate_with_ollama(prompt: str, model_name: str = "llama2") -> str:
    """Generate text using local Ollama instance"""
    response = requests.post('http://localhost:11434/api/generate',
                           json={
                               'model': model_name,
                               'prompt': prompt,
                               'stream': False
                           })
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(f"Ollama generation failed: {response.text}")

def generate_with_openrouter(prompt: str, model_name: str = "anthropic/claude-3-opus-20240229") -> tuple[str, float]:
    """Generate text using OpenRouter API with improved retry logic"""
    OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")
    headers = {
        "HTTP-Referer": "https://github.com/OpenRouterTeam/openrouter-python",
        "X-Title": "CAG Benchmark",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,         # Equivalent to do_sample=False
        "max_tokens": 300,          # Same as max_new_tokens
        "top_p": 1.0,              # Ensure deterministic output
        "frequency_penalty": 0.0,   # No frequency penalty
        "presence_penalty": 0.0,    # No presence penalty
        "stream": False
    }
    
    max_retries = 5
    base_delay = 2
    total_retry_time = 0
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        if "message" in response_json["choices"][0]:
                            return response_json["choices"][0]["message"]["content"], total_retry_time
                        elif "text" in response_json["choices"][0]:
                            return response_json["choices"][0]["text"], total_retry_time
                    # If we get here, the response was 200 but didn't have the expected content
                    error_msg = f"Unexpected response format: {response.text}"
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse response: {str(e)}, Response text: {response.text}"
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"{error_msg}, retrying in {delay} seconds...")
                    time_module.sleep(delay)
                    total_retry_time += delay
                    continue
                raise Exception(error_msg)
            
            # Handle rate limits (429) with longer delays
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit in response, retrying in {delay} seconds...")
                    time_module.sleep(delay)
                    total_retry_time += delay
                    continue
            
            # Handle other non-200 status codes
            error_msg = f"OpenRouter API request failed with status {response.status_code}: {response.text}"
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"{error_msg}, retrying in {delay} seconds...")
                time_module.sleep(delay)
                total_retry_time += delay
                continue
            raise Exception(error_msg)
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Network error, retrying in {delay} seconds: {str(e)}")
                time_module.sleep(delay)
                total_retry_time += delay
                continue
            raise Exception(f"Network error while calling OpenRouter API: {str(e)}")
    
    raise Exception(f"Failed to generate response after {max_retries} retries")

def load_model(args):
    """Load either OpenRouter, Ollama or HuggingFace model based on args"""
    if args.model_type in ["openrouter", "ollama"]:
        try:
            if args.model_type == "openrouter":
                headers = {
                    "Authorization": f"Bearer {get_api_key('OPENROUTER_API_KEY')}"
                }
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
                if response.status_code != 200:
                    raise Exception("Failed to connect to OpenRouter")
                logger.info(f"Successfully connected to OpenRouter")
            else:  # ollama
                response = requests.get('http://localhost:11434/api/tags')
                if response.status_code != 200:
                    raise Exception("Failed to connect to Ollama")
                logger.info(f"Successfully connected to Ollama")
            return None, None
        except Exception as e:
            raise Exception(f"Failed to connect to {args.model_type}: {str(e)}")
    else:
        if args.quantized:
            return load_quantized_model(model_name=args.modelname, hf_token=get_api_key("HF_TOKEN"))
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.modelname, token=get_api_key("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(
                args.modelname,
                torch_dtype=torch.float16,
                device_map="auto",
                token=get_api_key("HF_TOKEN")
            )
            return tokenizer, model

def get_kis_dataset(filepath: str):
    """Load KIS dataset from CSV file"""
    import pandas as pd
    df = pd.read_csv(filepath)
    dataset = zip(df['sample_question'], df['sample_ground_truth'])
    text_list = df["ki_text"].to_list()
    return text_list, dataset 