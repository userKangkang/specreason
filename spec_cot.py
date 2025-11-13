# %%
import os
import time
import openai
import pickle
import pprint
import logging
import argparse
import numpy as np
from openai import OpenAI
from collections import Counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model(model_size):
    client = clients[model_size]
    models = client.models.list()
    model = models.data[0].id
    return model

# %%
model_names = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "32b": "Qwen/QwQ-32B",
}
ports = {
    "1.5b": "30001",
    "32b": "30000",
}
clients = {}
for size, full_name in model_names.items():
    clients[size] = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{ports[size]}/v1",
    )

def get_first_user_msg(problem, options=None):
    if options is None:
        system_prompt = """
        Solve the following math problem efficiently and clearly. Please reason step by step, 
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        Problem: {problem}
        """
        return system_prompt.format(problem=problem)
    else:
        system_prompt = """
        What is the correct answer to the following problem? Please reason step by step. 
        Separate logical reasoning steps with two newline characters (\n\n).
        Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).

        **Example output:** \\boxed{{A}}

        Problem: {problem}.
        Choices: 
        (A) {ans_a}
        (B) {ans_b}
        (C) {ans_c}
        (D) {ans_d}
        """
        return system_prompt.format(
            problem=problem,
            ans_a=options["A"],
            ans_b=options["B"],
            ans_c=options["C"],
            ans_d=options["D"],
        )

# %%
def generate_step_base_model(problem, steps_so_far, model_size, options=None, stop_token="\n\n"):
    """
    Generates a single reasoning step using the specified (typically base) model.
    Used for fallback or initial steps.
    """
    client = clients[model_size]
    
    if steps_so_far == []:  # first step
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
        ]
        extra_body = {"add_generation_prompt": True}
    else:  # continuing on from a previous message
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        ]
        extra_body = {"add_generation_prompt": False, "continue_final_message": True}
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.6, top_p=0.95, # https://huggingface.co/Qwen/QwQ-32B#usage-guidelines
        max_tokens=512,
        stop=[stop_token],
        extra_body=extra_body,
    )

    elapsed = time.perf_counter() - start_time
    step_str = response.choices[0].message.content
    num_output_tokens = response.usage.completion_tokens
    finished = any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]])
    
    return step_str, finished, num_output_tokens, elapsed


def generate_drafts(problem, steps_so_far, model_size, options, num_drafts, stop_token="\n\n"):
    """
    Generates N drafts using the small model.
    """
    client = clients[model_size]
    
    if steps_so_far == []:  # first step
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
        ]
        extra_body = {"add_generation_prompt": True}
    else:  # continuing on from a previous message
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        ]
        extra_body = {"add_generation_prompt": False, "continue_final_message": True}
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        n=num_drafts,  # <-- SpecCoT: Generate N drafts
        temperature=0.7, # Use temperature for diversity in drafts
        top_p=0.95,
        max_tokens=512,
        stop=[stop_token],
        extra_body=extra_body,
    )
    
    elapsed = time.perf_counter() - start_time
    drafts = [choice.message.content for choice in response.choices]
    num_output_tokens_total = response.usage.completion_tokens # Total tokens for all drafts
    
    return drafts, elapsed, num_output_tokens_total


def select_draft_or_fallback(problem, steps_so_far, drafts, options, model_size="32b"):
    """
    Uses the large model to select the best draft or return 'FALLBACK'.
    """
    client = clients[model_size]
    
    # 1. Build the prompt for the large model
    # It needs the full context + the drafts as options
    
    # Context: problem + steps_so_far
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
    ]
    if steps_so_far:
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        messages.append({"role": "assistant", "content": f"<think>{steps_so_far_str}"})
    
    # New User Turn: The Selection Task
    draft_options_str = "\n\n".join(
        [f"Draft {chr(65+i)}:\n{d.strip()}" for i, d in enumerate(drafts)]
    )
    
    selection_prompt = f"""Here are {len(drafts)} candidate drafts for the next reasoning step. 
Evaluate them based on factual correctness and logical validity, continuing from the steps above.

{draft_options_str}

Which draft is the best continuation? 
Respond *only* with the single capital letter of the best draft (e.g., 'A', 'B', 'C', ...).
If *none* of the drafts are acceptable or correct, respond *only* with the word 'FALLBACK'.
"""
    messages.append({"role": "user", "content": selection_prompt})

    # 2. Call the large model for a choice
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.0, # We want a deterministic choice
        max_tokens=10, # Enough for 'FALLBACK' or 'A'
        extra_body={"add_generation_prompt": False, "continue_final_message": False}, # This is a new turn
    )
    elapsed = time.perf_counter() - start_time
    selection = response.choices[0].message.content.strip()
    logging.info(f"Large model selection: {selection}")

    # 3. Parse the selection
    try:
        if selection.upper() == 'FALLBACK':
            return None, "FALLBACK", elapsed
        
        selected_index = ord(selection[0].upper()) - 65 # 'A' -> 0
        if 0 <= selected_index < len(drafts):
            return drafts[selected_index], "SELECTED", elapsed
        else:
            logging.warning(f"Invalid selection '{selection}', falling back.")
            return None, "FALLBACK", elapsed
    except Exception as e:
        logging.warning(f"Error parsing selection '{selection}': {e}. Falling back.")
        return None, "FALLBACK", elapsed


def get_dataset(dataset_name):
    if dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "gpqa":
        if os.getenv("HF_HUB_OFFLINE", "0") == "1":
            dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
        else:    
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    else:
        raise NotImplementedError
    return dataset



# %%
parser = argparse.ArgumentParser(description="Runs Speculative CoT reasoning")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="gpqa",
                    help="Dataset")
parser.add_argument("--num_drafts", type=int, default=4,
                    help="Number of drafts for SpecCoT")
parser.add_argument("--token_budget", type=int, default=8192,
                    help="Max num of total output tokens in each step")
parser.add_argument("--problem_id", type=int, default=60,
                    help="Query ID (60-89 for AIME)")
parser.add_argument("--repeat_id", type=int, default=0,
                    help="Repeat ID (0-15, k=16)")
parser.add_argument("--output_dir", type=str, default="/data2/ruipan/specreason/playground", 
                    help="Where result pickle files will be written to")
parser.add_argument("--first_n_steps_base_model", type=int, default=0, 
                    help="First n steps use base model only")                    
args, _ = parser.parse_known_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

args.dataset = get_dataset(args.dataset_name)

# %%
if args.dataset_name == "aime":
    problem = args.dataset["problem"][args.problem_id - 60]
    options = None
elif args.dataset_name == "math":
    problem = args.dataset["problem"][args.problem_id]
    options = None
elif args.dataset_name == "gpqa":
    problem = args.dataset["Question"][args.problem_id]
    options = {
        "A": args.dataset["Correct Answer"][args.problem_id],
        "B": args.dataset["Incorrect Answer 1"][args.problem_id],
        "C": args.dataset["Incorrect Answer 2"][args.problem_id],
        "D": args.dataset["Incorrect Answer 3"][args.problem_id],
    }

problem_id = f"{args.dataset_name}_{args.problem_id}"


output_filename = os.path.join(args.output_dir, f"{args.problem_id}/{args.repeat_id}")
if os.path.exists(f"{output_filename}.pickle"):
    logging.info(f"Problem {args.problem_id} repeat {args.repeat_id} resolved, exiting")
    exit()
    
steps_so_far = []
step_id = 0
metadata_list = []
try:
    while True:
        warning_flag = False
        step_time = 0
        
        # Reset metadata for this step
        drafts_generated = None
        small_model_time = None
        num_output_tokens_small_total = None
        decision = None
        eval_time = None
        base_model_step = None
        num_output_tokens_base = None
        base_model_time = None
        final_num_output_tokens = 0

        if step_id < args.first_n_steps_base_model:  # First n steps use base model
            step_str, finished, num_output_tokens_base, base_model_time = generate_step_base_model(
                problem, steps_so_far, "32b", options=options
            )
            base_model_step = step_str
            step_time = base_model_time
            final_num_output_tokens = num_output_tokens_base
            decision = "BASE_FIRST_N"
            logging.info(f"[Step {step_id}] Base model first n: {step_str}")

        else:
            # 1. Generate N drafts using the small model
            drafts_generated, small_model_time, num_output_tokens_small_total = generate_drafts(
                problem, steps_so_far, "1.5b", options, num_drafts=args.num_drafts
            )
            step_time += small_model_time
            
            # 2. Use the base model to select a draft or fallback
            selected_draft_text, decision, eval_time = select_draft_or_fallback(
                problem, steps_so_far, drafts_generated, options, model_size="32b"
            )
            step_time += eval_time
            
            # 3. Handle the decision
            if decision == "SELECTED":
                logging.info(f"[Step {step_id}] Accepted draft!")
                step_str = selected_draft_text
                # Estimate tokens for the single selected draft (crude approximation)
                final_num_output_tokens = num_output_tokens_small_total // args.num_drafts 
            
            else: # decision == "FALLBACK"
                logging.info(f"[Step {step_id}] Drafts rejected, falling back to base model")
                step_str, finished, num_output_tokens_base, base_model_time = generate_step_base_model(
                    problem, steps_so_far, "32b", options=options
                )
                base_model_step = step_str
                step_time += base_model_time
                final_num_output_tokens = num_output_tokens_base
            
            # NOTE(ruipan): potential optimization is to pipeline the decoding of these two models rather than sequentially
            
            if "</think>" in step_str and not any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]]):
                logging.warning(f"Warning: step_str had a </think>, removing. {step_str}")
                step_str = step_str.replace("</think>", "")
                warning_flag = True
            
        # 4. repeat until an answer gets generated in the response
        steps_so_far.append(step_str)
        logging.info(f"[Step {step_id}] final step_str: {step_str}")
        
        metadata = {
            "step_id": step_id,
            "step_str": step_str,
            "decision": decision,
            "drafts_generated": drafts_generated,
            "num_output_tokens_small_total": num_output_tokens_small_total,
            "small_model_time": small_model_time, 
            "eval_time": eval_time,
            "base_model_step": base_model_step, # Not None if FALLBACK or BASE_FIRST_N
            "num_output_tokens_base": num_output_tokens_base,
            "base_model_time": base_model_time, 
            "final_num_output_tokens": final_num_output_tokens,
            "step_time": step_time,
        }
        if warning_flag:
            metadata["warning"] = "step_str had a </think>"
        metadata_list.append(metadata)
        step_id += 1
        
        # Check finish condition
        finished = any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]])
        if len(steps_so_far) > 2:
            finished = finished or steps_so_far[-1] == steps_so_far[-2]  # handles repetition
        
        if finished or sum([m["final_num_output_tokens"] for m in metadata_list]) >= args.token_budget:
            if sum([m["final_num_output_tokens"] for m in metadata_list]) >= args.token_budget:
                metadata_list[-1]["stop_reason"] = "budget"
            else:
                metadata_list[-1]["stop_reason"] = "finished"
            break
except ValueError:
    logging.error(f"ValueError caught in chat template application, continuing")

os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)

with open(f"{output_filename}.pickle", "wb") as f:
    pickle.dump(metadata_list, f)

with open(f"{output_filename}.txt", "w") as f:
    pprint.pprint(metadata_list, stream=f)