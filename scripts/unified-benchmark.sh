#!/bin/bash

# Configuration
BENCHMARK_NAME="test_q50_3_1_or"

# Parse command line arguments
RUN_KVCACHE=false
RUN_KVCACHE_NOKV=false
RUN_BM25=false
RUN_CUSTOMGPT=false
USE_QUANTIZED=false

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --kvcache)
            RUN_KVCACHE=true
            shift
            ;;
        --kvcache-nokv)
            RUN_KVCACHE_NOKV=true
            shift
            ;;
        --bm25)
            RUN_BM25=true
            shift
            ;;
        --customgpt)
            RUN_CUSTOMGPT=true
            shift
            ;;
        --quantized)
            USE_QUANTIZED=true
            shift
            ;;
        --all)
            RUN_KVCACHE=true
            RUN_KVCACHE_NOKV=true
            RUN_BM25=true
            RUN_CUSTOMGPT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--kvcache] [--kvcache-nokv] [--bm25] [--customgpt] [--quantized] [--all]"
            exit 1
            ;;
    esac
done

# If no components specified, show usage
if [[ "$RUN_KVCACHE" == "false" && "$RUN_KVCACHE_NOKV" == "false" && "$RUN_BM25" == "false" && "$RUN_CUSTOMGPT" == "false" ]]; then
    echo "Please specify at least one component to run:"
    echo "Usage: $0 [--kvcache] [--kvcache-nokv] [--bm25] [--customgpt] [--quantized] [--all]"
    exit 1
fi

# Default dataset and model configurations
datasets=("hotpotqa-train")
declare -A model_configs=(
    ["openrouter"]="meta-llama/llama-3.1-8b-instruct"
    # ["ollama"]="llama3.2:3b-instruct-q8_0"
    # ["huggingface"]="meta-llama/Llama-3.1-8B-Instruct"
)
k=("50")
maxParagraph=("100")    # Max paragraphs to process
maxQuestions=("50")     # Number of questions to process
top_k=("5")            # Top-k values for retrieval
randomSeed=42          # Fixed seed for reproducibility

# Setup paths and logging
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
mkdir -p "${PROJECT_ROOT}/logs"
logfilename="${PROJECT_ROOT}/logs/unified-benchmark.log"
touch "$logfilename"

# Function to log messages both to file and console
log_message() {
    echo "$1"
    echo "$1" >> "$logfilename"
}

log_message "Starting unified benchmark at $(date)"
log_message "----------------------------------------"

# Log test configuration
log_message "Benchmark configuration:"
log_message "Datasets: ${datasets[*]}"
log_message "Model configs: ${!model_configs[@]}"
log_message "Knowledge sizes: ${k[*]}"
log_message "Max paragraphs: ${maxParagraph[*]}"
log_message "Max questions: ${maxQuestions[*]}"
log_message "Top-k values: ${top_k[*]}"
log_message "Random seed: $randomSeed"
log_message "Quantized: $USE_QUANTIZED"

# Phase 1: Create CustomGPT agents
log_message "Creating all necessary CustomGPT agents..."
for dataset in "${datasets[@]}"; do
    for model_type in "${!model_configs[@]}"; do
        model="${model_configs[$model_type]}"
        for knowledge in "${k[@]}"; do
            result_dir="${PROJECT_ROOT}/results/${dataset}_unified-benchmark_${model_type}_${BENCHMARK_NAME}/result_${knowledge}"
            mkdir -p "${result_dir}"
            
            log_message "Creating agent for dataset: $dataset, knowledge size: $knowledge"
            python "${PROJECT_ROOT}/customgpt-ai.py" \
                --index "customgpt" \
                --dataset "$dataset" \
                --similarity bertscore \
                --maxKnowledge "$knowledge" \
                --maxQuestion 1 \
                --topk 1 \
                --model_type "$model_type" \
                --modelname "$model" \
                --randomSeed "$randomSeed" \
                --output "${result_dir}/agent_creation.log" \
                --create_only
            
            log_message "Agent creation completed for knowledge size: $knowledge"
        done
    done
done

log_message "All agents created successfully"
log_message "----------------------------------------"

# Phase 2: Run benchmarks
for dataset in "${datasets[@]}"; do
    for model_type in "${!model_configs[@]}"; do
        model="${model_configs[$model_type]}"
        
        log_message "Testing with model type: $model_type, model: $model"
        
        for maxQuestion in "${maxQuestions[@]}"; do
            for knowledge in "${k[@]}"; do
                for p in "${maxParagraph[@]}"; do  # Added loop for maxParagraph
                    # Calculate iterations based on batch size
                    batch=$knowledge
                    iteration=$(($maxQuestion / $batch))

                    # Create result directory
                    result_dir="${PROJECT_ROOT}/results/${dataset}_unified-benchmark_${model_type}_${BENCHMARK_NAME}/result_${knowledge}"
                    mkdir -p "${result_dir}"
                    
                    # Log current run configuration
                    log_message "Running benchmark:"
                    log_message "Dataset: $dataset"
                    log_message "Model type: $model_type"
                    log_message "Model: $model"
                    log_message "Knowledge size: $knowledge"
                    log_message "Max Paragraph: $p"
                    log_message "Questions: $maxQuestion"
                    log_message "Random seed: $randomSeed"
                    log_message "Components to run:"
                    [[ "$RUN_KVCACHE" == "true" ]] && log_message "- KV Cache"
                    [[ "$RUN_KVCACHE_NOKV" == "true" ]] && log_message "- KV Cache (no-kv)"
                    [[ "$RUN_BM25" == "true" ]] && log_message "- BM25"
                    [[ "$RUN_CUSTOMGPT" == "true" ]] && log_message "- CustomGPT"
                    log_message "----------------------------------------"
                    
                    # Clear PID arrays for this iteration
                    declare -a bm25_pids=()
                    declare -a customgpt_pids=()
                    
                    # Run selected methods in parallel
                    
                    # Run KV Cache if selected
                    if [[ "$RUN_KVCACHE" == "true" ]]; then
                        # Run KV Cache with cache
                        log_message "Starting KV Cache (with cache) in background..."
                        python "${PROJECT_ROOT}/kvcache.py" \
                            --kvcache file \
                            --dataset "$dataset" \
                            --similarity bertscore \
                            --maxKnowledge "$knowledge" \
                            --maxParagraph "$p" \
                            --maxQuestion "$maxQuestion" \
                            --model_type "$model_type" \
                            --modelname "$model" \
                            --randomSeed "$randomSeed" \
                            $([ "$USE_QUANTIZED" == "true" ] && echo "--quantized true") \
                            --output "${result_dir}/result_${knowledge}_p${p}_kvcache$([ "$USE_QUANTIZED" == "true" ] && echo "_quantized").txt" &
                        kvcache_pid=$!
                    fi
                    
                    # Run KV Cache No-KV if selected
                    if [[ "$RUN_KVCACHE_NOKV" == "true" ]]; then
                        # Run KV Cache without cache (nokv)
                        log_message "Starting KV Cache (without cache) in background..."
                        python "${PROJECT_ROOT}/kvcache.py" \
                            --kvcache file \
                            --dataset "$dataset" \
                            --similarity bertscore \
                            --maxKnowledge "$knowledge" \
                            --maxParagraph "$p" \
                            --maxQuestion "$maxQuestion" \
                            --usePrompt \
                            --model_type "$model_type" \
                            --modelname "$model" \
                            --randomSeed "$randomSeed" \
                            $([ "$USE_QUANTIZED" == "true" ] && echo "--quantized true") \
                            --output "${result_dir}/result_${knowledge}_p${p}_kvcache_nokv$([ "$USE_QUANTIZED" == "true" ] && echo "_quantized").txt" &
                        kvcache_nokv_pid=$!
                    fi
                    
                    # Run BM25 if selected
                    if [[ "$RUN_BM25" == "true" ]]; then
                        for topk in "${top_k[@]}"; do
                            log_message "Starting BM25 with top_k=${topk} in background..."
                            python "${PROJECT_ROOT}/rag.py" \
                                --index "bm25" \
                                --dataset "$dataset" \
                                --similarity bertscore \
                                --maxKnowledge "$knowledge" \
                                --maxParagraph "$p" \
                                --maxQuestion "$maxQuestion" \
                                --topk "$topk" \
                                --model_type "$model_type" \
                                --modelname "$model" \
                                --randomSeed "$randomSeed" \
                                $([ "$USE_QUANTIZED" == "true" ] && echo "--quantized true") \
                                --output "${result_dir}/result_${knowledge}_p${p}_rag_Index_bm25_top${topk}$([ "$USE_QUANTIZED" == "true" ] && echo "_quantized").txt" &
                            bm25_pids+=($!)
                        done
                    fi
                    
                    # Run CustomGPT if selected
                    if [[ "$RUN_CUSTOMGPT" == "true" ]]; then
                        for topk in "${top_k[@]}"; do
                            log_message "Starting CustomGPT with top_k=${topk} in background..."
                            python "${PROJECT_ROOT}/customgpt-ai.py" \
                                --index "customgpt" \
                                --dataset "$dataset" \
                                --similarity bertscore \
                                --maxKnowledge "$knowledge" \
                                --maxParagraph "$p" \
                                --maxQuestion "$maxQuestion" \
                                --topk "$topk" \
                                --model_type "$model_type" \
                                --modelname "$model" \
                                --randomSeed "$randomSeed" \
                                $([ "$USE_QUANTIZED" == "true" ] && echo "--quantized true") \
                                --output "${result_dir}/result_${knowledge}_p${p}_rag_Index_customgpt_top${topk}$([ "$USE_QUANTIZED" == "true" ] && echo "_quantized").txt" &
                            customgpt_pids+=($!)
                        done
                    fi
                    
                    # Wait for all background processes to complete
                    log_message "Waiting for all processes to complete..."
                    
                    # Only wait for processes that were started
                    if [[ "$RUN_KVCACHE" == "true" ]]; then
                        wait $kvcache_pid && log_message "KV Cache (with cache) completed"
                    fi

                    if [[ "$RUN_KVCACHE_NOKV" == "true" ]]; then
                        wait $kvcache_nokv_pid && log_message "KV Cache (without cache) completed"
                    fi
                    
                    if [[ "$RUN_BM25" == "true" ]]; then
                        for pid in "${bm25_pids[@]}"; do
                            wait $pid
                        done
                        log_message "All BM25 processes completed"
                    fi
                    
                    if [[ "$RUN_CUSTOMGPT" == "true" ]]; then
                        for pid in "${customgpt_pids[@]}"; do
                            wait $pid
                        done
                        log_message "All CustomGPT processes completed"
                    fi
                    
                    log_message "Completed benchmark for knowledge size $knowledge and paragraph size $p"
                    log_message "----------------------------------------"
                done # for maxParagraph
            done # for knowledge
        done # for maxQuestion
    done # for model_type
done # for dataset

log_message "Benchmark completed at $(date)"
log_message "Results can be found in: ${result_dir}"

# Phase 3: Generate visualizations
log_message "Generating visualizations..."
for dataset in "${datasets[@]}"; do
    for model_type in "${!model_configs[@]}"; do
        dataset_model_dir="${PROJECT_ROOT}/results/${dataset}_unified-benchmark_${model_type}_${BENCHMARK_NAME}"
        log_message "Generating visualizations for ${dataset} - unified-benchmark_${model_type}_${BENCHMARK_NAME}..."
        python "${PROJECT_ROOT}/scripts/visualize_results.py" "${dataset_model_dir}" "${BENCHMARK_NAME}"
    done
done

log_message "Visualizations completed"
log_message "All tasks completed successfully"