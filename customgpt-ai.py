import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, Document
from transformers.cache_utils import DynamicCache
import argparse
import os
import json
from transformers import BitsAndBytesConfig
import random
import requests
import time as time_module
from time import time
import logging
import warnings
import sys  # Add sys import
from typing import Dict, List, Tuple, Optional
import traceback

# Suppress BM25 escape sequence warning
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

from utils import (
 validate_env_variables, get_bert_similarity, 
     generate_with_ollama, generate_with_openrouter,
    get_kis_dataset, unique_path,
    get_api_key,
    logger
)

# Add ensure_output_directory function
def ensure_output_directory(directory):
    """Create output directory if it doesn't exist"""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

"""Sentence-BERT for evaluate semantic similarity"""
from sentence_transformers import SentenceTransformer, util
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight sentence-transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

env = validate_env_variables()
HF_TOKEN = env["HF_TOKEN"]
OPENROUTER_API_KEY = env.get("OPENROUTER_API_KEY", "")

global model_name, model, tokenizer
global rand_seed

# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

def get_document_class():
    """Get Document class only when needed"""
    from llama_index.core import Document
    return Document

def load_torch_dependencies():
    """Load PyTorch and related dependencies only when needed"""
    global torch, F, AutoTokenizer, AutoModelForCausalLM, VectorStoreIndex, Document, DynamicCache
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llama_index.core import VectorStoreIndex, Document
    from transformers.cache_utils import DynamicCache
    
    # Allowlist the DynamicCache class
    torch.serialization.add_safe_globals([DynamicCache])
    torch.serialization.add_safe_globals([set])

def get_bert_similarity_fn():
    """Get BERT similarity function only when needed"""
    from utils import get_bert_similarity
    return get_bert_similarity

def generate_agent_name(knowledge_size: int, benchmark_type: str, dataset: str) -> str:
    """Generate deterministic agent name based on knowledge size, benchmark type and dataset
    
    Args:
        knowledge_size: Size of knowledge base
        benchmark_type: Type of benchmark (test/unified)
        dataset: Name of the dataset being used
    
    Returns:
        Agent name with format: agent_k{knowledge_size}_{dataset}_{benchmark_type}
    """
    return f"agent_k{knowledge_size}_{dataset}_{benchmark_type}"

def upload_document(args: Tuple[str, int, str, Dict, str]) -> Dict:
    """Helper function to upload a single document to CustomGPT.ai"""
    doc, idx, project_id, headers, api_endpoint = args
    
    logger.info(f"Processing document {idx + 1}")
    
    # Format document for upload using multipart/form-data
    files = {
        'file': (f'document_{idx}.txt', doc.encode('utf-8'), 'text/plain')
    }
    
    # Create new headers for upload without content-type (will be set by requests)
    upload_headers = headers.copy()
    upload_headers.pop('content-type', None)  # Remove content-type if present
    upload_headers['accept'] = 'application/json'
    
    # Use the correct sources endpoint for file upload
    upload_url = f'{api_endpoint}projects/{project_id}/sources'
    
    try:
        response = requests.post(
            upload_url,
            headers=upload_headers,
            files=files
        )
        
        if response.status_code not in [200, 201]:
            logger.error(f"Upload failed with status code {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return {
                'status': 'error',
                'error': f"Failed to upload document: {response.text}"
            }
            
        response_data = response.json()
        if response_data.get('status') != 'success':
            logger.error(f"Upload failed with API error: {response_data}")
            return {
                'status': 'error',
                'error': f"Failed to upload document: {response.text}"
            }
            
        # Get source ID and page ID from the response data
        source_id = response_data['data']['id']
        page_id = response_data['data']['pages'][0]['id']
        logger.info(f"Document {idx + 1} uploaded successfully with ID: {source_id}, page ID: {page_id}")
        
        return {
            'status': 'success',
            'source_id': source_id,
            'page_id': page_id
        }
        
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}")
        return {'status': 'error', 'error': str(e)}

def check_agent_documents(api_endpoint: str, headers: Dict, project_id: str) -> bool:
    """Check if an agent has any documents uploaded"""
    sources_url = f'{api_endpoint}projects/{project_id}/sources'
    get_headers = headers.copy()
    get_headers['accept'] = 'application/json'
    
    try:
        response = requests.get(sources_url, headers=get_headers)
        if response.status_code != 200:
            logger.error(f"Failed to check documents. Status code: {response.status_code}")
            return False
            
        sources_data = response.json()
        if sources_data.get('status') != 'success':
            logger.error("Failed to get sources data")
            return False
            
        sources = sources_data.get('data', {}).get('data', [])
        return len(sources) > 0
    except Exception as e:
        logger.error(f"Error checking documents: {str(e)}")
        return False

def wait_for_documents_processing(api_endpoint: str, headers: Dict, project_id: str, expected_count: int, max_wait_time: int = 120) -> bool:
    """Wait for documents to be processed with timeout"""
    logger.info("Checking document processing status...")
    start_time = time()
    
    while (time() - start_time) < max_wait_time:
        sources_url = f'{api_endpoint}projects/{project_id}/sources'
        get_headers = headers.copy()
        get_headers['accept'] = 'application/json'
        
        try:
            response = requests.get(sources_url, headers=get_headers)
            if response.status_code == 200:
                sources_data = response.json()
                if sources_data.get('status') == 'success':
                    sources = sources_data.get('data', {}).get('data', [])
                    if len(sources) >= expected_count:
                        logger.info("All documents processed successfully")
                        return True
            
            time_module.sleep(2)  # Short delay between checks
            
        except Exception as e:
            logger.error(f"Error checking processing status: {str(e)}")
            time_module.sleep(2)
            
    logger.warning(f"Document processing check timed out after {max_wait_time} seconds")
    return False

def get_or_create_agent(api_endpoint: str, headers: Dict, agent_name: str, documents: Optional[List[str]] = None):
    """Create new agent with the given name and documents"""
    max_retries = 5
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # First check if agent already exists
            get_headers = headers.copy()
            get_headers['content-type'] = 'application/json'
            response = requests.get(f'{api_endpoint}projects', headers=get_headers)
            projects = response.json()
            
            project_id = None
            if isinstance(projects, dict) and 'data' in projects:
                projects_data = projects['data'].get('data', [])
                for project in projects_data:
                    if isinstance(project, dict) and project.get('project_name') == agent_name:
                        project_id = project.get('id')
                        logger.info(f"Found existing agent {agent_name} with ID {project_id}")
                        break
            
            if project_id:
                # Check if agent has documents
                if check_agent_documents(api_endpoint, headers, project_id):
                    logger.info("Agent has existing documents")
                    return project_id, 0
                else:
                    logger.info("Agent exists but has no documents, uploading documents...")
                    if documents is None:
                        raise ValueError("Documents required for agent with no existing documents")
            else:
                # Add delay before creation to reduce race condition chance
                time_module.sleep(base_delay * (attempt + 1))
                
                # Create new agent
                logger.info(f"Creating new agent {agent_name}...")
                
                create_headers = headers.copy()
                create_headers['content-type'] = 'application/json'
                
                response = requests.post(
                    f'{api_endpoint}projects',
                    headers=create_headers,
                    json={
                        'project_name': agent_name,
                        'type': 'FILE'
                    }
                )
                
                create_response = response.json()
                if response.status_code == 409:  # Conflict - agent already exists
                    continue  # Retry to get the existing agent
                    
                if response.status_code not in [200, 201] or create_response.get('status') != 'success':
                    raise Exception(f"Failed to create agent. Status code: {response.status_code}, Response: {create_response}")
                
                project_id = create_response.get('data', {}).get('id')
                if not project_id:
                    raise Exception("Project ID not found in response")
                
                logger.info(f"Successfully created new agent {agent_name} with ID: {project_id}")
                
                # Add delay after project creation
                time_module.sleep(5)  # Increased delay to ensure project is ready
            
            # Upload documents
            if documents:
                logger.info("Starting document uploads...")
                upload_results = []
                
                for idx, doc in enumerate(documents):
                    result = upload_document((doc, idx, project_id, headers, api_endpoint))
                    if result['status'] == 'error':
                        raise Exception(f"Failed to upload document {idx}: {result['error']}")
                    upload_results.append(result)
                    logger.info(f"Document {idx + 1}/{len(documents)} completed")
                    
                # Wait for processing with a reasonable timeout
                logger.info("Checking document processing status...")
                if not wait_for_documents_processing(api_endpoint, headers, project_id, len(documents)):
                    logger.warning("Processing check timed out, but continuing as documents may still be usable")
                
                logger.info("Document upload and processing completed")
            
            return project_id, 0
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Error occurred, retrying in {delay} seconds: {str(e)}")
                time_module.sleep(delay)
                continue
            raise Exception(f"Failed to create agent after {max_retries} retries: {str(e)}")
    
    raise Exception(f"Failed to create agent after {max_retries} retries")

def getCustomGPTRetriever(documents: Optional[List[str]] = None, similarity_top_k: int = 1, knowledge_size: int = 16, benchmark_type: str = "test", dataset: Optional[str] = None):
    """CustomGPT.ai RAG model with parallel document uploading"""
    CUSTOMGPT_API_KEY = get_api_key("CUSTOMGPT_API_KEY")
    
    api_endpoint = 'https://app.customgpt.ai/api/v1/'
    base_headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
        'authorization': f'Bearer {CUSTOMGPT_API_KEY}'
    }
    
    if dataset is None:
        raise ValueError("Dataset name is required for agent creation")
    
    # Generate deterministic agent name
    agent_name = generate_agent_name(knowledge_size, benchmark_type, dataset)
    
    # First check if agent exists before loading any models or data
    response = requests.get(f'{api_endpoint}projects', headers=base_headers)
    projects = response.json()
    
    project_id = None
    prepare_time = 0
    
    if isinstance(projects, dict) and 'data' in projects:
        projects_data = projects['data'].get('data', [])
        for project in projects_data:
            if isinstance(project, dict) and project.get('project_name') == agent_name:
                project_id = project.get('id')
                logger.info(f"Found existing agent {agent_name} with ID {project_id}")
                break
    
    if project_id is None:
        if documents is None:
            raise ValueError("Documents required to create new agent")
        
        # Create new agent
        t1 = time()
        project_id, prepare_time = get_or_create_agent(api_endpoint, base_headers, agent_name, documents)
        t2 = time()
        prepare_time = t2 - t1
    
    class CustomGPTRetriever:
        def __init__(self, project_id: str, headers: dict, api_endpoint: str, similarity_top_k: int):
            self.project_id = project_id
            self.base_headers = headers.copy()
            self.api_endpoint = api_endpoint
            self.similarity_top_k = similarity_top_k
            self.conversation_id = None
            self._create_conversation()
        
        def _create_conversation(self):
            """Create a conversation for this retriever instance"""
            if self.conversation_id is not None:
                try:
                    requests.delete(
                        f'{self.api_endpoint}projects/{self.project_id}/conversations/{self.conversation_id}',
                        headers=self.base_headers
                    )
                except:
                    pass
            
            conversation_url = f'{self.api_endpoint}projects/{self.project_id}/conversations'
            
            headers = self.base_headers.copy()
            headers['content-type'] = 'application/json'
            
            response = requests.post(
                conversation_url, 
                headers=headers, 
                json={"name": f"conversation_top{self.similarity_top_k}"}
            )
            
            response_data = response.json()
            if response_data.get('status') != 'success':
                raise Exception(f"Failed to create conversation: {response.text}")
            
            self.conversation_id = response_data['data']['id']
            logger.info(f"Created conversation with ID: {self.conversation_id}")
        
        def retrieve(self, query: str) -> List:
            """Get similar documents using the messages endpoint and retrieve their citations"""
            message_url = f'{self.api_endpoint}projects/{self.project_id}/conversations/{self.conversation_id}/messages'
            
            headers = self.base_headers.copy()
            headers['content-type'] = 'application/json'
            
            try:
                # Step 1: Get citation IDs from message
                response = requests.post(
                    message_url,
                    headers=headers,
                    json={
                        "prompt": query,
                        "stream": False,
                        "top_k": self.similarity_top_k
                    }
                )
                
                message_data = response.json().get('data', {})
                citations = message_data.get('citations', [])[:self.similarity_top_k]
                logger.info(f"Found citations: {citations}")
                
                class Node:
                    def __init__(self, text):
                        self.text = text
                    def __str__(self):
                        return self.text
                
                documents = []
                for citation_id in citations:
                    try:
                        # Step 2: Get citation details
                        citation_url = f'{self.api_endpoint}projects/{self.project_id}/citations/{citation_id}'
                        citation_response = requests.get(citation_url, headers=headers)
                        
                        if citation_response.status_code != 200:
                            logger.warning(f"Failed to get citation {citation_id}")
                            continue
                            
                        citation_data = citation_response.json()
                        
                        # Step 3: Get the preview URL
                        preview_url = citation_data.get('data', {}).get('url')
                        if not preview_url:
                            logger.warning(f"No preview URL found for citation {citation_id}")
                            continue
                        
                        # Step 4: Get the actual content - Use Bearer token for preview
                        preview_headers = {
                            'Authorization': f'Bearer {self.base_headers["authorization"].split(" ")[-1]}'
                        }
                        preview_response = requests.get(preview_url, headers=preview_headers)
                        
                        if preview_response.status_code != 200:
                            logger.warning(f"Failed to get preview content from {preview_url} with status {preview_response.status_code}")
                            # Try alternative URL format
                            alt_preview_url = f'{self.api_endpoint}preview/{citation_id}'
                            preview_response = requests.get(alt_preview_url, headers=preview_headers)
                            if preview_response.status_code != 200:
                                logger.warning(f"Also failed with alternative URL {alt_preview_url}")
                                continue
                        
                        citation_text = preview_response.text
                        if citation_text:
                            documents.append(Node(citation_text))
                            logger.info(f"Retrieved citation text: {citation_text[:500]}...")
                        
                    except Exception as e:
                        logger.error(f"Error fetching citation {citation_id}: {str(e)}")
                        continue
                
                if not documents:
                    logger.warning("No documents retrieved!")
                else:
                    logger.info(f"Retrieved {len(documents)} documents")
                    for i, doc in enumerate(documents):
                        logger.info(f"Document {i+1} content: {str(doc)[:500]}...")
                
                return documents
                
            except Exception as e:
                logger.error(f"Error in retrieve: {str(e)}")
                raise
    
    retriever = CustomGPTRetriever(project_id, base_headers, api_endpoint, similarity_top_k)
    return retriever, prepare_time

def test_customgpt_retriever():
    """Test function to verify CustomGPT.ai retriever functionality"""
    test_documents = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Python is a programming language created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms."
    ]
    
    logger.info("\n=== Testing CustomGPT.ai Retriever ===")
    logger.info("Initializing retriever with test documents...")
    
    try:
        retriever, prep_time = getCustomGPTRetriever(
            test_documents, 
            similarity_top_k=1,
            dataset="test_dataset"  # Added test dataset name
        )
        logger.info(f"✓ Retriever initialized in {prep_time:.2f} seconds")
        
        test_queries = [
            "What is the capital of France?",
            "Who created Python?",
            "What is machine learning?"
        ]
        
        logger.info("\nTesting retrieval with sample queries:\n")
        for query in test_queries:
            logger.info(f"Query: {query}")
            t1 = time()
            results = retriever.retrieve(query)
            t2 = time()
            logger.info(f"Retrieved in {t2-t1:.2f} seconds")
            logger.info(f"Response: {results[0].text}\n")
                
        logger.info("\n✓ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {str(e)}")
        raise

def parse_squad_data(raw):
    dataset = { "ki_text": [], "qas": [] }
    
    for k_id, data in enumerate(raw['data']):
        article = []
        for p_id, para in enumerate(data['paragraphs']):
            article.append(para['context'])
            for qa in para['qas']:
                ques = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                dataset['qas'].append({"title": data['title'], "paragraph_index": tuple((k_id, p_id)) ,"question": ques, "answers": answers})
        dataset['ki_text'].append({"id": k_id, "title": data['title'], "paragraphs": article})
    
    return dataset

def get_squad_dataset(filepath: str, max_knowledge: int = None, max_paragraph: int = None, max_questions: int = None):
    # Open and read the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = parse_squad_data(data)
    
    logger.info(f"max_knowledge: {max_knowledge}, max_paragraph: {max_paragraph}, max_questions: {max_questions}")
    
    # Set the limit Maximum Articles, use all Articles if max_knowledge is None or greater than the number of Articles
    max_knowledge = max_knowledge if max_knowledge != None and max_knowledge < len(parsed_data['ki_text']) else len(parsed_data['ki_text'])
    
    # Shuffle the Articles and Questions
    if rand_seed != None:
        random.seed(rand_seed)
        random.shuffle(parsed_data["ki_text"])
        random.shuffle(parsed_data["qas"])
        k_ids = [i['id'] for i in parsed_data["ki_text"][:max_knowledge]]
        
    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data['ki_text'][:max_knowledge]:
        max_para = max_paragraph if max_paragraph != None and max_paragraph < len(article['paragraphs']) else len(article['paragraphs'])
        text_list.append(article['title'])
        text_list.append('\n'.join(article['paragraphs'][0:max_para]))
    
    # Check if the knowledge id of qas is less than the max_knowledge
    questions = [qa['question'] for qa in parsed_data['qas'] if qa['paragraph_index'][0] in k_ids and (max_paragraph == None or qa['paragraph_index'][1] < max_paragraph)]
    answers = [qa['answers'][0] for qa in parsed_data['qas'] if qa['paragraph_index'][0]  in k_ids and (max_paragraph == None or qa['paragraph_index'][1] < max_paragraph)]
    
    dataset = zip(questions, answers)
    
    return text_list, dataset

def get_hotpotqa_dataset(filepath: str, max_knowledge: int = None):
    # Open and read the JSON
    with open (filepath, "r") as file:
        data = json.load(file)
    
    if rand_seed != None:
        random.seed(rand_seed)
        random.shuffle(data)
    
    questions = [ qa['question'] for qa in data ]
    answers = [ qa['answer'] for qa in data ]
    dataset = zip(questions, answers)
    
    if max_knowledge == None:
        max_knowledge = len(data)
    else:
        max_knowledge = min(max_knowledge, len(data))
    
    text_list = []
    for i, qa in enumerate(data[:max_knowledge]):
        context = qa['context']
        context = [ c[0] + ": \n" + "".join(c[1]) for c in context ]
        article = "\n\n".join(context)

        text_list.append(article)
    
    return text_list, dataset

def rag_test(args: argparse.Namespace):
    answer_instruction = None
    if args.dataset == "kis_sample":
        datapath = "./datasets/rag_sample_qas_from_kis.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "kis":
        datapath = "./datasets/synthetic_knowledge_items.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "squad-dev":
        datapath = "./datasets/squad/dev-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-dev":
        datapath = "./datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-test":
        datapath = "./datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-train":
        datapath = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."

    # Update default answer instruction if none provided
    if answer_instruction is None:
        answer_instruction = """Answer the question with a super short answer, providing only the essential information in a few words. 
Do not give explanations or additional context.

Examples:
❌ Wrong (too verbose): "Rosie Mac was the body double for Emilia Clarke in her portrayal of Daenerys Targaryen in Game of Thrones."
✓ Correct (concise): "Rosie Mac."
"""

    ensure_output_directory(os.path.dirname(args.output))
    
    Document = get_document_class()  # Get Document class only when needed
    documents = [Document(text=t) for t in text_list]
    
    # Determine benchmark type from the output path more robustly
    if args.output == "/dev/null":
        # For create_only mode, infer from the script name
        script_name = os.path.basename(sys.argv[0])
        benchmark_type = "unified" if "unified" in script_name else "test"
    else:
        # For normal mode, infer from the output path
        benchmark_type = "unified" if "unified_results" in args.output else "test"
    
    # Create retriever with initial top-k value
    retriever, prepare_time = getCustomGPTRetriever(
        text_list, 
        args.topk, 
        knowledge_size=args.maxKnowledge,
        benchmark_type=benchmark_type,
        dataset=args.dataset
    )
    logger.info(f"Testing CustomGPT retriever with {len(documents)} documents.")
    logger.info(f"Retriever prepared in {prepare_time} seconds")
    
    with open(args.output, "a") as f:
        f.write(f"Retriever prepared in {prepare_time} seconds\n")
    
    results = {
        "retrieve_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }
    
    dataset = list(dataset)
    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion is not None else len(dataset)
    
    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):
        try:
            retrieve_t1 = time()
            nodes = retriever.retrieve(question)
            retrieve_t2 = time()
            retrieve_time = retrieve_t2 - retrieve_t1
            
            # Use the same joining logic as KVCache
            knowledge = '\n\n\n\n\n\n'.join([node.text for node in nodes])
            
            prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant for giving very concise answers based on given context. Always provide the shortest possible correct answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context information is below.
------------------------------------------------
{knowledge}
------------------------------------------------
{answer_instruction}
Question:
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

            generate_t1 = time()
            if args.model_type == "openrouter":
                generated_text, retry_time = generate_with_openrouter(prompt, args.modelname)
                if generated_text is None:
                    logger.warning(f"[{id}] Failed to generate response, skipping...")
                    continue
                generate_t2 = time()
                generate_time = generate_t2 - generate_t1 - retry_time
            elif args.model_type == "ollama":
                generated_text = generate_with_ollama(prompt, args.modelname)
                generate_t2 = time()
                generate_time = generate_t2 - generate_t1
            else:  # huggingface
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    input_ids,
                    max_new_tokens=300,
                    do_sample=False    # Removed temperature and top_p parameters
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_text = generated_text[generated_text.find(question) + len(question):]
                generated_text = generated_text[generated_text.find('assistant') + len('assistant'):].lstrip()
                generate_t2 = time()
                generate_time = generate_t2 - generate_t1

            logger.info("\nQuestion: %s", question)
            logger.info("Ground Truth: %s", ground_truth)
            logger.info("\nPrompt:\n%s", prompt)
            logger.info("\nGenerated Response: %s", generated_text)
            logger.info("\nRetrieved Knowledge:\n%s", knowledge)

            similarity = get_bert_similarity(generated_text, ground_truth)
            
            logger.info(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t"
                       f"retrieve time: {retrieve_time},\t"
                       f"generate time: {generate_time}")
            
            with open(args.output, "a") as f:
                f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t retrieve time: {retrieve_time},\t generate time: {generate_time}\n")
            
            results["prompts"].append(prompt)
            results["responses"].append(generated_text)
            results["retrieve_time"].append(retrieve_time)
            results["generate_time"].append(generate_time)
            results["similarity"].append(similarity)
            
            with open(args.output, "a") as f:
                f.write(f"[{id}]: [Cumulative]: " 
                        + f"Semantic Similarity: {round(sum(results['similarity']) / (len(results['similarity'])) , 5)}," 
                        + f"\t retrieve time: {sum(results['retrieve_time']) / (len(results['retrieve_time'])) },"
                        + f"\t generate time: {sum(results['generate_time']) / (len(results['generate_time'])) }\n")
        except Exception as e:
            logger.error(f"Error processing question {id}: {str(e)}")
            continue
    
    # Write final summary
    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_retrieve_time = sum(results["retrieve_time"]) / len(results["retrieve_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    
    logger.info("\nFinal Summary:")
    logger.info(f"Prepare time: {prepare_time}")
    logger.info(f"Average Semantic Similarity: {avg_similarity}")
    logger.info(f"Average retrieve time: {avg_retrieve_time}")
    logger.info(f"Average generate time: {avg_generate_time}")
    
    with open(args.output, "a") as f:
        f.write("\nFinal Summary:\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"Average retrieve time: {avg_retrieve_time}\n")
        f.write(f"Average generate time: {avg_generate_time}\n")

# Define quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
)

def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=True,     # Required for some models
        token=hf_token
    )
    
    return tokenizer, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--create_only', action='store_true', help='Only create the agent without running benchmarks')
    
    if '--test' not in sys.argv:
        parser.add_argument('--modelname', required=False, default="meta-llama/Llama-3.2-1B-Instruct", type=str, help='Model name to use')
        parser.add_argument('--model_type', choices=['huggingface', 'ollama', 'openrouter'], required=False, default='huggingface', help='Type of model to use')
        parser.add_argument('--quantized', required=False, default=False, type=bool, help='Quantized model')
        parser.add_argument('--index', choices=['customgpt'], required=True, help='Index to use')
        parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use')
        parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
        parser.add_argument('--maxQuestion', required=False, default=None, type=int, help='Maximum number of questions to test')
        parser.add_argument('--maxKnowledge', required=False, default=None, type=int, help='Maximum number of knowledge items to use')
        parser.add_argument('--maxParagraph', required=False, default=None, type=int, help='Maximum number of paragraph to use')
        parser.add_argument('--topk', required=False, default=1, type=int, help='Top K retrievals to use')
        parser.add_argument('--dataset', required=True, choices=['kis', 'kis_sample', 'squad-dev', 'squad-train', 'hotpotqa-dev', 'hotpotqa-train', 'hotpotqa-test'], help='Dataset to use')
        parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')
    
    args = parser.parse_args()
    
    if args.test:
        test_customgpt_retriever()
    else:
        logger.info(f"maxKnowledge: {args.maxKnowledge}, maxParagraph: {args.maxParagraph}, maxQuestion: {args.maxQuestion}, randomSeed: {args.randomSeed}")
        
        model_name = args.modelname
        rand_seed = args.randomSeed if args.randomSeed is not None else None
        
        if args.model_type == 'huggingface':
            if args.quantized:
                tokenizer, model = load_quantized_model(model_name=model_name, hf_token=HF_TOKEN)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=HF_TOKEN
                )
        
        if os.path.exists(args.output):
            args.output = unique_path(args.output)
            
        if args.create_only:
            logger.info("Agent creation completed. Exiting as --create_only was specified.")
            sys.exit(0)
            
        rag_test(args)
