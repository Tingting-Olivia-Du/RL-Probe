#!/usr/bin/env python3
"""
Step 3: Compute KL Divergence

Compute token-level KL divergence between RLVR checkpoints and DPO baseline.
Uses teacher forcing to analyze how RLVR models respond to DPO-generated errors.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.analysis.kl_divergence import KLDivergenceAnalyzer
from src.analysis.spike_detector import SpikeDetector
from src.analysis.entropy import EntropyAnalyzer
from src.analysis.token_classifier import TokenClassifier
from src.data.preprocessor import DataPreprocessor
from src.models.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute KL divergence")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--rollouts", type=str, default=None, help="Rollouts path (if not specified, will auto-detect)")
    parser.add_argument("--rollouts-dir", type=str, default="rollouts", help="Base directory for rollouts")
    parser.add_argument("--use-dpo-responses", action="store_true", help="Use DPO responses as rollouts (from dpo_responses.json)")
    parser.add_argument("--dpo-responses", type=str, default="data/filtered/dpo_responses.json", help="Path to DPO responses file")
    parser.add_argument("--output-dir", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to analyze")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N problems (default: 10)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file (e.g., 'rlvr_step_500_results.json')")
    parser.add_argument("--skip-completed", action="store_true", default=True, help="Automatically skip checkpoints that have already been fully evaluated (default: True)")
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false", help="Do not skip completed checkpoints (re-evaluate all)")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Specify GPU(s) to use. Examples: '0' for cuda:0, '0,1,2' for multiple GPUs, 'cuda:1' for specific device",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override GPU settings from command line if provided
    if args.gpu:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",") if x.strip().isdigit()]
        if len(gpu_ids) == 1:
            config["hardware"]["device"] = f"cuda:{gpu_ids[0]}"
            config["hardware"]["multi_gpu"] = {"enabled": False}
            logger.info(f"Using single GPU: cuda:{gpu_ids[0]}")
        elif len(gpu_ids) > 1:
            config["hardware"]["device"] = "auto"
            if "multi_gpu" not in config["hardware"]:
                config["hardware"]["multi_gpu"] = {}
            config["hardware"]["multi_gpu"]["enabled"] = True
            config["hardware"]["multi_gpu"]["gpu_ids"] = gpu_ids
            logger.info(f"Using multiple GPUs: {gpu_ids}")
        else:
            if args.gpu.startswith("cuda:") or args.gpu == "cpu":
                config["hardware"]["device"] = args.gpu
                config["hardware"]["multi_gpu"] = {"enabled": False}
                logger.info(f"Using device: {args.gpu}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Results will be saved to: {output_dir.absolute()}")
    logger.info(f"💾 Auto-save interval: every {args.save_interval} problems")

    # Determine rollouts source
    if args.use_dpo_responses:
        # Use DPO responses as rollouts
        logger.info("Using DPO responses as rollouts...")
        dpo_responses_path = Path(args.dpo_responses)
        if not dpo_responses_path.exists():
            raise FileNotFoundError(
                f"DPO responses file not found: {dpo_responses_path}\n"
                f"Please generate DPO responses first using: python scripts/01_prepare_data.py --model dpo"
            )
        
        with open(dpo_responses_path) as f:
            dpo_responses = json.load(f)
        
        # Load filtered problems to get problem text
        filtered_problems_path = Path("data/filtered/filtered_problems.json")
        problem_text_map = {}
        if filtered_problems_path.exists():
            try:
                with open(filtered_problems_path) as f:
                    filtered_problems = json.load(f)
                for prob in filtered_problems:
                    prob_id = prob.get("unique_id") or f"hash_{abs(hash(prob.get('problem', '')))}"
                    problem_text_map[prob_id] = prob.get("problem", "")
                logger.info(f"Loaded problem texts for {len(problem_text_map)} problems")
            except Exception as e:
                logger.warning(f"Could not load filtered_problems.json: {e}")
        
        # Convert DPO responses to rollouts format
        all_rollouts = {}
        for problem_id, response_data in dpo_responses.items():
            # Get problem text from map or use empty string
            problem_text = problem_text_map.get(problem_id, "")
            
            if isinstance(response_data, dict) and "response" in response_data:
                # Convert to rollouts format: {problem_id: [rollout1, rollout2, ...]}
                all_rollouts[problem_id] = [{
                    "rollout": response_data["response"],
                    "problem_id": problem_id,
                    "problem_text": problem_text,
                    "ground_truth": response_data.get("ground_truth", ""),
                    "dpo_correct": response_data.get("correct", False),
                    "sample_idx": 0,
                    "source": "dpo_responses"
                }]
            else:
                # Handle old format (just response string)
                all_rollouts[problem_id] = [{
                    "rollout": response_data if isinstance(response_data, str) else str(response_data),
                    "problem_id": problem_id,
                    "problem_text": problem_text,
                    "ground_truth": "",
                    "sample_idx": 0,
                    "source": "dpo_responses"
                }]
        
        logger.info(f"Loaded {len(all_rollouts)} DPO responses as rollouts")
        logger.info("Note: Using DPO-generated wrong rollouts for teacher forcing analysis")
        
    elif args.rollouts:
        rollouts_path = args.rollouts
        logger.info(f"Loading rollouts from specified path: {rollouts_path}")
        if not Path(rollouts_path).exists():
            raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")
        with open(rollouts_path) as f:
            all_rollouts = json.load(f)
        logger.info(f"Loaded rollouts for {len(all_rollouts)} problems")
    else:
        # Auto-detect rollouts path based on checkpoint
        if args.checkpoint:
            rollouts_path = Path(args.rollouts_dir) / args.checkpoint / "all_rollouts.json"
        else:
            # Default to final checkpoint if not specified
            rollouts_path = Path(args.rollouts_dir) / "rlvr_step_1000" / "all_rollouts.json"
        logger.info(f"Auto-detected rollouts path: {rollouts_path}")
        
        if not Path(rollouts_path).exists():
            raise FileNotFoundError(
                f"Rollouts file not found: {rollouts_path}\n"
                f"Please generate rollouts first using: python scripts/02_generate_rollouts.py --checkpoint {args.checkpoint or 'rlvr_step_1000'}\n"
                f"Or use DPO responses: python scripts/03_compute_kl.py --use-dpo-responses"
            )
        
        with open(rollouts_path) as f:
            all_rollouts = json.load(f)
        logger.info(f"Loaded rollouts for {len(all_rollouts)} problems")

    # Load models
    logger.info("Loading models...")
    loader = ModelLoader(
        device=config["hardware"]["device"],
        dtype=getattr(torch, config["hardware"]["dtype"]),
    )
    loader.register_from_config(config)

    tokenizer = loader.load_tokenizer()
    preprocessor = DataPreprocessor(tokenizer)

    # Load DPO as reference
    dpo_model = loader.load_model("dpo")

    # Determine which checkpoints to analyze
    if args.checkpoint:
        checkpoints_to_analyze = [args.checkpoint]
    else:
        checkpoints_to_analyze = loader.get_rlvr_checkpoints()
        logger.info(f"Found {len(checkpoints_to_analyze)} checkpoints to analyze: {checkpoints_to_analyze}")
    
    # Function to check if a checkpoint is fully completed
    def is_checkpoint_completed(ckpt_name: str, expected_problem_count: int) -> bool:
        """Check if checkpoint has been fully evaluated."""
        ckpt_output = output_dir / f"{ckpt_name}_results.json"
        if not ckpt_output.exists():
            return False
        
        try:
            with open(ckpt_output) as f:
                results = json.load(f)
            
            # Check if we have results for all expected problems
            kl_analyses = results.get("kl_analyses", [])
            unique_problems = set(a["problem_id"] for a in kl_analyses)
            
            # A checkpoint is considered complete if it has results for all expected problems
            is_complete = len(unique_problems) >= expected_problem_count
            
            if is_complete:
                logger.info(f"✅ Checkpoint {ckpt_name} already completed ({len(unique_problems)}/{expected_problem_count} problems)")
            else:
                logger.info(f"⚠️  Checkpoint {ckpt_name} partially completed ({len(unique_problems)}/{expected_problem_count} problems)")
            
            return is_complete
        except Exception as e:
            logger.warning(f"Failed to check completion status for {ckpt_name}: {e}")
            return False
    
    # Filter out completed checkpoints if requested
    if args.skip_completed and not args.checkpoint:
        expected_problem_count = len(all_rollouts)
        filtered_checkpoints = []
        skipped_count = 0
        
        for ckpt_name in checkpoints_to_analyze:
            if is_checkpoint_completed(ckpt_name, expected_problem_count):
                skipped_count += 1
            else:
                filtered_checkpoints.append(ckpt_name)
        
        if skipped_count > 0:
            logger.info(f"\n⏭️  Skipping {skipped_count} already completed checkpoint(s)")
            logger.info(f"📋 Remaining checkpoints to process: {len(filtered_checkpoints)}")
            if filtered_checkpoints:
                logger.info(f"   {filtered_checkpoints}")
        
        checkpoints_to_analyze = filtered_checkpoints
        
        if not checkpoints_to_analyze:
            logger.info("\n✅ All checkpoints have been fully evaluated!")
            logger.info("   Use --no-skip-completed to re-evaluate all checkpoints")
            return

    # Initialize analyzers
    # Ensure epsilon is a float (YAML might parse 1e-10 as string)
    epsilon = config["analysis"]["kl_divergence"]["epsilon"]
    if isinstance(epsilon, str):
        epsilon = float(epsilon)
    
    kl_analyzer = KLDivergenceAnalyzer(
        epsilon=epsilon,
        compute_forward=config["analysis"]["kl_divergence"]["compute_forward"],
        compute_reverse=config["analysis"]["kl_divergence"]["compute_reverse"],
        compute_js=config["analysis"]["kl_divergence"]["compute_js"],
    )

    spike_detector = SpikeDetector(
        method=config["analysis"]["spike_detection"]["method"],
        threshold=config["analysis"]["spike_detection"]["threshold"],
        min_spike_distance=config["analysis"]["spike_detection"]["min_spike_distance"],
    )

    entropy_analyzer = EntropyAnalyzer(normalize=config["analysis"]["entropy"]["normalize"])
    token_classifier = TokenClassifier()

    # Results storage
    all_results = {}

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    for ckpt_idx, ckpt_name in enumerate(checkpoints_to_analyze, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Processing checkpoint {ckpt_idx}/{len(checkpoints_to_analyze)}: {ckpt_name}")
        logger.info(f"{'='*60}")
        
        # Check if resume checkpoint exists
        ckpt_output = output_dir / f"{ckpt_name}_results.json"
        processed_problem_ids = set()
        ckpt_results = {
            "kl_analyses": [],
            "spike_analyses": [],
            "entropy_analyses": [],
            "category_stats": [],
        }
        
        # Always try to resume if checkpoint file exists (for partial completion)
        if ckpt_output.exists():
            logger.info(f"📂 Found existing results file: {ckpt_output}")
            logger.info(f"   Attempting to resume from partial results...")
            try:
                with open(ckpt_output) as f:
                    existing_results = json.load(f)
                # Extract processed problem IDs
                for analysis in existing_results.get("kl_analyses", []):
                    processed_problem_ids.add(analysis["problem_id"])
                # Load existing results
                ckpt_results = existing_results
                logger.info(f"   ✅ Resumed: {len(processed_problem_ids)}/{len(all_rollouts)} problems already processed")
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to load checkpoint, starting fresh: {e}")
                processed_problem_ids = set()
                ckpt_results = {
                    "kl_analyses": [],
                    "spike_analyses": [],
                    "entropy_analyses": [],
                    "category_stats": [],
                }
        else:
            logger.info(f"   🆕 Starting fresh evaluation for {ckpt_name}")
        
        # Log memory status before loading new checkpoint
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"   📊 GPU Memory before loading {ckpt_name}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        logger.info(f"   ⬇️  Loading checkpoint model: {ckpt_name}")
        ckpt_model = loader.load_model(ckpt_name)
        
        # Log memory status after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"   📊 GPU Memory after loading {ckpt_name}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        problem_count = 0
        total_problems = len(all_rollouts)
        for problem_id, rollouts in tqdm(all_rollouts.items(), desc=f"Processing {ckpt_name}", total=total_problems):
            # Skip if already processed (when resuming)
            if problem_id in processed_problem_ids:
                continue
                
            for rollout_data in rollouts:
                rollout_text = rollout_data["rollout"]
                problem_text = rollout_data["problem_text"]

                prompt = preprocessor.format_prompt(problem_text)

                # Compute KL divergence
                kl_result = kl_analyzer.analyze_rollout(
                    model_p=ckpt_model,
                    model_q=dpo_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    rollout=rollout_text,
                )

                # Detect spikes
                spike_analysis = spike_detector.detect_spikes(
                    kl_values=kl_result.kl_forward,
                    token_ids=kl_result.token_ids,
                    token_strings=kl_result.token_strings,
                    prompt_length=kl_result.prompt_length,
                )

                # Analyze entropy
                entropy_result = entropy_analyzer.analyze_sequence(
                    model=ckpt_model,
                    input_ids=kl_result.token_ids.unsqueeze(0),
                    prompt_length=kl_result.prompt_length,
                )

                # Classify tokens (ensure float32 before numpy conversion)
                kl_values_for_classification = kl_result.kl_forward
                if kl_values_for_classification.dtype == torch.bfloat16:
                    kl_values_for_classification = kl_values_for_classification.float()
                classified_tokens = token_classifier.classify_sequence(
                    token_strings=kl_result.token_strings,
                    token_ids=kl_result.token_ids.tolist(),
                    kl_values=kl_values_for_classification.cpu().numpy(),
                )
                category_stats = token_classifier.get_category_statistics(classified_tokens)

                # Store results (convert tensors to lists for serialization)
                # All tensors should already be float32 from KLResult, but ensure it
                kl_forward_np = kl_result.kl_forward.cpu().numpy()
                kl_reverse_np = kl_result.kl_reverse.cpu().numpy() if kl_result.kl_reverse is not None else None
                js_div_np = kl_result.js_divergence.cpu().numpy() if kl_result.js_divergence is not None else None
                
                ckpt_results["kl_analyses"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "kl_forward": kl_forward_np.tolist(),
                    "kl_reverse": kl_reverse_np.tolist() if kl_reverse_np is not None else None,
                    "js_divergence": js_div_np.tolist() if js_div_np is not None else None,
                    "prompt_length": kl_result.prompt_length,
                    "token_strings": kl_result.token_strings,
                })

                ckpt_results["spike_analyses"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "num_spikes": len(spike_analysis.spikes),
                    "spike_positions": [s.position for s in spike_analysis.spikes],
                    "spike_values": [s.value for s in spike_analysis.spikes],
                    "spike_tokens": [s.token_string for s in spike_analysis.spikes],
                    "spike_regions": [s.region for s in spike_analysis.spikes],
                    "mean_kl": spike_analysis.mean_kl,
                    "std_kl": spike_analysis.std_kl,
                    "threshold": spike_analysis.threshold,
                    "spike_density": spike_analysis.spike_density,
                })

                ckpt_results["entropy_analyses"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "mean_entropy": entropy_result.mean_entropy,
                    "std_entropy": entropy_result.std_entropy,
                    "mean_normalized": entropy_result.mean_normalized,
                    "region_entropy": entropy_result.region_entropy,
                })

                ckpt_results["category_stats"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "stats": category_stats,
                })
            
            # Increment problem count after processing all rollouts for this problem
            problem_count += 1
            
            # Periodic save (every N problems)
            if problem_count % args.save_interval == 0 and problem_count > 0:
                # Convert all numpy types before saving
                ckpt_results_serializable = convert_to_json_serializable(ckpt_results)
                
                # Save checkpoint (temporary file first, then rename for atomic write)
                ckpt_output_temp = output_dir / f"{ckpt_name}_results.json.tmp"
                ckpt_output = output_dir / f"{ckpt_name}_results.json"
                
                with open(ckpt_output_temp, "w") as f:
                    json.dump(ckpt_results_serializable, f, indent=2)
                ckpt_output_temp.rename(ckpt_output)
                
                logger.info(f"💾 Auto-saved checkpoint after {problem_count} problems: {ckpt_output}")

        all_results[ckpt_name] = ckpt_results

        # Final save
        # Convert all numpy types before saving
        ckpt_results_serializable = convert_to_json_serializable(ckpt_results)

        # Save checkpoint results (final)
        ckpt_output = output_dir / f"{ckpt_name}_results.json"
        ckpt_output_temp = output_dir / f"{ckpt_name}_results.json.tmp"
        
        with open(ckpt_output_temp, "w") as f:
            json.dump(ckpt_results_serializable, f, indent=2)
        ckpt_output_temp.rename(ckpt_output)
        
        # Count final statistics
        final_problem_count = len(set(a["problem_id"] for a in ckpt_results["kl_analyses"]))
        total_analyses = len(ckpt_results["kl_analyses"])
        logger.info(f"\n✅ Completed checkpoint {ckpt_name}:")
        logger.info(f"   📊 Problems analyzed: {final_problem_count}/{len(all_rollouts)}")
        logger.info(f"   📈 Total analyses: {total_analyses}")
        logger.info(f"   💾 Results saved to: {ckpt_output}")

        # Unload to save memory - ensure complete cleanup
        logger.info(f"   🧹 Unloading checkpoint model to free GPU memory...")
        loader.unload_model(ckpt_name)
        
        # Additional memory cleanup steps
        if torch.cuda.is_available():
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            # Force garbage collection to ensure Python objects are freed
            import gc
            gc.collect()
            torch.cuda.empty_cache()  # Clear again after GC
            
            # Log memory status
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"   📊 GPU Memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        logger.info(f"   ✅ Ready for next checkpoint\n")

    # Save combined results
    combined_output = output_dir / "all_results.pkl"
    with open(combined_output, "wb") as f:
        pickle.dump(all_results, f)
    logger.info(f"Saved combined results to {combined_output}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 KL Analysis Summary")
    logger.info("=" * 60)

    if not all_results:
        logger.info("No results to summarize (all checkpoints were skipped or failed)")
        return

    # Sort checkpoints by step number for better visualization
    def get_step_number(ckpt_name: str) -> int:
        try:
            return int(ckpt_name.replace("rlvr_step_", ""))
        except:
            return 0
    
    sorted_results = sorted(all_results.items(), key=lambda x: get_step_number(x[0]))
    
    logger.info(f"\n{'Checkpoint':<20} {'Problems':<12} {'Mean KL':<15} {'Std KL':<12} {'Total Spikes':<15}")
    logger.info("-" * 80)
    
    for ckpt_name, results in sorted_results:
        mean_kls = [a["mean_kl"] for a in results["spike_analyses"]]
        total_spikes = sum(a["num_spikes"] for a in results["spike_analyses"])
        unique_problems = len(set(a["problem_id"] for a in results["kl_analyses"]))

        mean_kl = np.mean(mean_kls) if mean_kls else 0.0
        std_kl = np.std(mean_kls) if mean_kls else 0.0
        
        logger.info(f"{ckpt_name:<20} {unique_problems:<12} {mean_kl:<15.6f} {std_kl:<12.6f} {total_spikes:<15}")
    
    logger.info("\n💡 Tip: Use scripts/04_visualize.py to generate visualizations and find KL turning points")

    loader.unload_all()


if __name__ == "__main__":
    main()
