#!/bin/bash

################################################################################
# MAS-Consensus Experiment Runner
# Runs multiple experiments in parallel with logging and progress tracking
################################################################################

# Note: We don't use 'set -e' because we want to track job failures individually

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Activated Python virtual environment"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Activated Python virtual environment"
else
    echo "Please create Python virtual environment"
    echo "Use: uv sync --extra dev"
    exit 1
fi

# Add src directory to PYTHONPATH
# export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
MAX_PARALLEL=${MAX_PARALLEL:-4}  # Maximum number of parallel jobs
DATASETS=("csqa" "gsm8k" "fact" "bias")
GRAPH_TYPES=("complete" "chain" "circle" "tree")
NUM_AGENTS=6
NUM_AUDITORS=2
REG_TURN=9
SAMPLE_ID=3
THREADS=16  # Number of threads per Python program

# Parse command line arguments
EXPERIMENTS=()
DRY_RUN=false
VERBOSE=false

print_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run MAS-Consensus experiments in parallel.

OPTIONS:
    --experiments EXPS   Comma-separated list of experiments to run
                         Groups: all, basic, analysis
                         Individual: csqa, gsm8k, fact, bias, adv, chain, defense,
                                    efficiency, malicious
                         Default: basic

    --datasets DATASETS  Comma-separated list of datasets. Default: csqa,gsm8k,fact,bias
                         Applies to: basic experiments and analysis experiments

    --graphs GRAPHS      Comma-separated list of graph types. Default: complete,chain,circle,tree
                         Applies to: basic experiments only (csqa, gsm8k, fact, bias, adv)
                         Note: Analysis experiments use fixed optimal topologies

    --num-agents N       Number of agents. Default: 6
                         Note: Efficiency analysis uses 4 agents by default for faster testing

    --num-auditors N     Number of auditors. Default: 2
                         Set to 0 to disable auditing mechanism

    --reg-turn N         Number of regulation turns. Default: 9
                         Note: Efficiency analysis uses 3 turns by default for faster testing

    --sample-id N        Sample ID to use from dataset. Default: 3

    --max-parallel N     Maximum parallel jobs. Default: 4
                         Adjust based on available CPU cores and memory

    --threads N          Number of threads per Python program. Default: 16
                         Controls internal parallelism within each experiment

    --log-dir DIR        Log directory. Default: ./logs/TIMESTAMP
                         All experiment output saved here with individual log files

    --dry-run            Preview commands without executing them
                         Use this to see what would run before committing resources

    --verbose            Show detailed output including commands and log locations

    -h, --help           Show this help message

EXPERIMENT GROUPS:
    basic                Run basic dataset experiments across multiple graph types
                         - Includes: csqa, gsm8k, fact, bias, adv
                         - Each runs with all specified graph types (complete, chain, circle, tree)
                         - Use this to test how different datasets perform across topologies

    analysis             Run specialized analysis experiments (one dataset at a time)
                         - chain: Analyze chain topology with sequential agent communication
                         - defense: Compare baseline, attacked, and defended scenarios
                         - efficiency: Measure execution time under different attack types
                         - malicious: Analyze impact of different malicious behaviors
                         - These focus on understanding system behavior, not just accuracy

    all                  Run everything: all basic experiments + all analysis experiments
                         - Combines both basic and analysis groups
                         - Runs ~20+ separate experiments (depends on datasets/graphs)

INDIVIDUAL EXPERIMENTS:
    You can also specify individual experiments by name:
    - csqa, gsm8k, fact, bias, adv  (run with multiple graph types)
    - chain, defense, efficiency, malicious  (run for each dataset)

EXAMPLES:
    # Example 1: Run basic experiments (tests datasets across graph topologies)
    $0 --experiments basic
    # Runs: csqa, gsm8k, fact, bias, adv on complete, chain, circle, tree graphs
    # Total: 5 datasets × 4 graphs = 20 experiments

    # Example 2: Run analysis suite (specialized analysis experiments)
    $0 --experiments analysis
    # Runs: chain analysis, defense comparison, efficiency analysis, malicious behavior
    # Total: 4 analysis types × 4 datasets = 16 experiments

    # Example 3: Quick test with one dataset and one graph type
    $0 --experiments csqa --graphs complete --datasets csqa
    # Runs: Just csqa with complete graph (1 experiment)

    # Example 4: Defense comparison for specific datasets
    $0 --experiments defense --datasets csqa,gsm8k
    # Runs: Defense comparison (baseline vs attacked vs defended) for 2 datasets

    # Example 5: Custom parameters for analysis
    $0 --experiments analysis --num-agents 10 --num-auditors 4 --reg-turn 12
    # Runs: All analysis experiments with custom agent/auditor counts

    # Example 6: Dry run to preview (recommended first step!)
    $0 --experiments all --dry-run
    # Shows: All commands that would be executed without actually running them

    # Example 7: High-performance parallel execution
    $0 --experiments all --max-parallel 8
    # Runs: Everything with 8 parallel jobs (requires sufficient CPU/memory)

    # Example 8: Control parallelism (jobs and threads)
    $0 --experiments basic --max-parallel 4 --threads 8
    # Runs: 4 experiments at once, each using 8 threads internally
    # Note: Total threads = max-parallel × threads (4 × 8 = 32 threads)

    # Example 9: Mix individual experiments
    $0 --experiments csqa,defense,malicious --datasets csqa
    # Runs: csqa basic + defense analysis + malicious analysis for csqa dataset

EOF
}

log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments)
            IFS=',' read -ra EXPERIMENTS <<< "$2"
            shift 2
            ;;
        --datasets)
            IFS=',' read -ra DATASETS <<< "$2"
            shift 2
            ;;
        --graphs)
            IFS=',' read -ra GRAPH_TYPES <<< "$2"
            shift 2
            ;;
        --num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --num-auditors)
            NUM_AUDITORS="$2"
            shift 2
            ;;
        --reg-turn)
            REG_TURN="$2"
            shift 2
            ;;
        --sample-id)
            SAMPLE_ID="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Default to basic experiments if none specified
if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    EXPERIMENTS=("basic")
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Job tracking
declare -a PIDS
declare -a JOB_NAMES
declare -a JOB_LOGS
JOB_COUNT=0

# Function to run a command in background with logging
run_job() {
    local name="$1"
    local cmd="$2"
    local log_file="${PWD}/${LOG_DIR}/${name}.log"  # Use absolute path
    local project_root="${PWD}"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $name: $cmd"
        echo "          Log: $log_file"
        return
    fi

    # Wait if we've reached max parallel jobs
    # Count only OUR jobs that are still running
    local running_jobs=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((running_jobs++))
        fi
    done
    while [ $running_jobs -ge $MAX_PARALLEL ]; do
        sleep 1
        running_jobs=0
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running_jobs++))
            fi
        done
    done

    if [ "$VERBOSE" = true ]; then
        log "Queuing job: $name"
        echo "  Command: $cmd"
        echo "  Log: $log_file"
    fi

    # Run command in background
    {
        echo "=== Job: $name ===" > "$log_file"
        echo "Command: $cmd" >> "$log_file"
        echo "Working Directory: ${project_root}/src/" >> "$log_file"
        echo "Started: $(date)" >> "$log_file"
        echo "==================" >> "$log_file"
        echo "" >> "$log_file"

        # Change to src directory where datasets are located
        cd "${project_root}/src"
        eval "$cmd" >> "$log_file" 2>&1
        local exit_code=$?

        echo "" >> "$log_file"
        echo "==================" >> "$log_file"
        echo "Finished: $(date)" >> "$log_file"
        echo "Exit code: $exit_code" >> "$log_file"

        # Print completion message to console (redirect to fd 3, which we'll connect to stdout)
        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} Completed: $name" >&3
        else
            echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} Failed: $name (exit code: $exit_code)" >&3
        fi

        exit $exit_code
    } 3>&1 &

    local job_pid=$!
    PIDS+=($job_pid)
    JOB_NAMES+=("$name")
    JOB_LOGS+=("$log_file")
    ((JOB_COUNT++))
}

# Function to wait for all jobs
wait_for_jobs() {
    log "Waiting for all jobs to complete..."

    local failed=0
    local succeeded=0

    for i in "${!PIDS[@]}"; do
        local pid=${PIDS[$i]}
        local name=${JOB_NAMES[$i]}
        local log_file=${JOB_LOGS[$i]}

        if wait $pid; then
            ((succeeded++))
        else
            ((failed++))
            log_error "Job failed: $name (log: $log_file)"
        fi
    done

    echo ""
    echo "═══════════════════════════════════════"
    log_success "Completed: $succeeded jobs"
    if [ $failed -gt 0 ]; then
        log_error "Failed: $failed jobs"
    fi
    echo "Logs saved to: $LOG_DIR"
    echo "═══════════════════════════════════════"

    return $failed
}

# Expand experiment groups (recursive expansion)
expand_experiments() {
    local exp="$1"
    case $exp in
        all)
            echo "csqa gsm8k fact bias adv chain defense efficiency malicious"
            ;;
        basic)
            echo "csqa gsm8k fact bias adv"
            ;;
        analysis)
            echo "chain defense efficiency malicious"
            ;;
        *)
            echo "$exp"
            ;;
    esac
}

EXPANDED_EXPERIMENTS=()
for exp in "${EXPERIMENTS[@]}"; do
    for expanded in $(expand_experiments "$exp"); do
        EXPANDED_EXPERIMENTS+=("$expanded")
    done
done

# Remove duplicates
EXPANDED_EXPERIMENTS=($(echo "${EXPANDED_EXPERIMENTS[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

# Print configuration
echo "═══════════════════════════════════════"
echo "MAS-Consensus Experiment Runner"
echo "═══════════════════════════════════════"
echo "Experiments: ${EXPANDED_EXPERIMENTS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Graph types: ${GRAPH_TYPES[*]}"
echo "Num agents: $NUM_AGENTS"
echo "Num auditors: $NUM_AUDITORS"
echo "Reg turns: $REG_TURN"
echo "Sample ID: $SAMPLE_ID"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Threads per job: $THREADS"
echo "Log directory: $LOG_DIR"
if [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN"
fi
echo "═══════════════════════════════════════"
echo ""

# Check for required environment variables
if [ -z "${OPENAI_API_KEY:-}" ]; then
    log_warning "OPENAI_API_KEY is not set. Experiments will fail."
    log_warning "Set it with: export OPENAI_API_KEY='your-key-here'"
    if [ "$DRY_RUN" != true ]; then
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Exiting..."
            exit 1
        fi
        echo ""
    fi
fi

# Run experiments
log "Scheduling ${#EXPANDED_EXPERIMENTS[@]} experiment type(s)..."

for exp in "${EXPANDED_EXPERIMENTS[@]}"; do
    case $exp in
        csqa|gsm8k|fact|bias|adv)
            # Basic dataset experiments
            for graph in "${GRAPH_TYPES[@]}"; do
                name="${exp}_${graph}"
                cmd="python -m mas_consensus.run_${exp} \
                    --sample_id $SAMPLE_ID \
                    --graph_type $graph \
                    --num_agents $NUM_AGENTS \
                    --num_auditors $NUM_AUDITORS \
                    --reg_turn $REG_TURN \
                    --parallel $THREADS"
                run_job "$name" "$cmd"
            done
            ;;

        chain)
            # Chain analysis
            for dataset in "${DATASETS[@]}"; do
                name="chain_analysis_${dataset}"
                cmd="python -m mas_consensus.run_chain_analysis \
                    --dataset $dataset \
                    --sample_id $SAMPLE_ID \
                    --graph_type chain \
                    --num_agents $NUM_AGENTS \
                    --num_auditors $NUM_AUDITORS \
                    --reg_turn $REG_TURN \
                    --attacker_num 1 \
                    --parallel $THREADS"
                run_job "$name" "$cmd"
            done
            ;;

        defense)
            # Defense comparison
            for dataset in "${DATASETS[@]}"; do
                name="defense_comparison_${dataset}"
                cmd="python -m mas_consensus.run_defense_comparison \
                    --dataset $dataset \
                    --sample_id $SAMPLE_ID \
                    --num_agents $NUM_AGENTS \
                    --num_auditors $NUM_AUDITORS \
                    --reg_turn $REG_TURN \
                    --attacker_num 1 \
                    --parallel $THREADS"
                run_job "$name" "$cmd"
            done
            ;;

        efficiency)
            # Efficiency analysis (use smaller parameters for faster testing)
            for dataset in "${DATASETS[@]}"; do
                name="efficiency_analysis_${dataset}"
                cmd="python -m mas_consensus.run_efficiency_analysis \
                    --dataset $dataset \
                    --sample_id $SAMPLE_ID \
                    --graph_type complete \
                    --num_agents 4 \
                    --num_auditors $NUM_AUDITORS \
                    --reg_turn 3 \
                    --parallel $THREADS"
                run_job "$name" "$cmd"
            done
            ;;

        malicious)
            # Malicious behavior analysis
            for dataset in "${DATASETS[@]}"; do
                name="malicious_behavior_${dataset}"
                cmd="python -m mas_consensus.run_malicious_behavior_analysis \
                    --dataset $dataset \
                    --sample_id $SAMPLE_ID \
                    --num_agents $NUM_AGENTS \
                    --num_auditors $NUM_AUDITORS \
                    --reg_turn $REG_TURN \
                    --parallel $THREADS"
                run_job "$name" "$cmd"
            done
            ;;

        *)
            log_warning "Unknown experiment type: $exp (skipping)"
            ;;
    esac
done

echo ""
log "Finished scheduling. Queued ${#PIDS[@]} job(s)"

if [ "$DRY_RUN" = true ]; then
    echo ""
    log "Dry run complete. No jobs were executed."
    exit 0
fi

# Check if any jobs were started
if [ ${#PIDS[@]} -eq 0 ]; then
    echo ""
    log_warning "No jobs were started. Check your experiment/dataset configuration."
    exit 0
fi

# Show which jobs are queued
echo ""
log "Jobs queued (will run in parallel, max $MAX_PARALLEL at a time):"
for job_name in "${JOB_NAMES[@]}"; do
    echo "  - $job_name"
done
echo ""

# Now start all the jobs
log "Starting all jobs..."
for i in "${!PIDS[@]}"; do
    log "  ▶ ${JOB_NAMES[$i]}"
done
echo ""

# Wait for all jobs to complete
wait_for_jobs
exit_code=$?

exit $exit_code
