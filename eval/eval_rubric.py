#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import base64
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from tqdm import tqdm
import signal
import sys

# API Config
API_URL = "xxx"
API_KEY = "sk-xxx"
MODEL = "gemini-3-pro" # "gpt-5.1"

# Locks for thread safety
file_lock = Lock()
pause_lock = Lock()
pause_until = 0

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def wait_if_paused():
    global pause_until
    while True:
        with pause_lock:
            current_time = time.time()
            if current_time >= pause_until:
                return
            wait_remaining = pause_until - current_time
        time.sleep(min(1, wait_remaining))

def set_global_pause(duration):
    global pause_until
    with pause_lock:
        new_pause_until = time.time() + duration
        if new_pause_until > pause_until:
            pause_until = new_pause_until
            tqdm.write(f"[Global Pause] Pausing all threads for {duration} seconds...")
            return True
    return False

def evaluate_rubric_category(ori_img_path, gen_img_path, items_with_indices, category, max_retries=3):
    """Call API to evaluate a batch of rubric items for a category"""
    
    # Construct the criteria text
    criteria_text = ""
    for idx, item in items_with_indices:
        criteria_text += f"### Criterion ID {idx}\n"
        criteria_text += f"**Description**: {item['description']}\n"
        criteria_text += f"**Weight**: {item.get('weight', 1)}\n\n"

    system_prompt = """You are an expert evaluator tasked with assessing whether a generated image satisfies specific rubric criteria. Your evaluation must be precise, objective, and based solely on the evidence present in the image. 
## Evaluation Framework 
You will evaluate each rubric criterion using a binary satisfaction scale: 
1. **Not Satisfied (Score: 0.0)**: The image fails to meet the criterion. Key elements are missing, incorrect, or inadequately addressed. 
2. **Satisfied (Score: 1.0)**: The image meets the criterion. All required elements are present and correctly depicted. 
## Evaluation Process 
1. **Understand the Criterion**: Carefully read and interpret what each rubric item is asking for. 
2. **Search for Evidence**: Systematically review the generated image for relevant visual content that addresses the criterion. 
3. **Assess Completeness**: Evaluate whether the evidence fully satisfies the criterion. 
4. **Provide Reasoning**: Explain your evaluation with specific references to visual elements in the image. 
## Important Guidelines 
- Base your evaluation ONLY on what is explicitly present in the generated image 
- Do not make assumptions about implied or missing content 
- Be consistent in your evaluation standards across all criteria 
- Provide specific examples from the image to support your verdict"""

    user_text = f"""## Visual Content
You are provided with two images. Please pay close attention to their order:
1. The **FIRST** image is the **Original Reference Image**. This serves as the ground truth.
2. The **SECOND** image is the **Generated Image to Evaluate**. 

## Rubric Criteria to Evaluate
Category: {category}

{criteria_text}

## Your Task
Evaluate whether the **SECOND** image (Generated Image) satisfies EACH of the above rubric criteria. You should use the **FIRST** image (Original Reference) to understand the expected visual elements and ground truth.

## Required Response Format
Provide your evaluation in the following JSON format:
```json
{{
  "evaluations": {{
    "criterion_id_here": {{
      "verdict": "[Not Satisfied/Satisfied]",
      "score": [0.0/1.0],
      "confidence": [0.0-1.0],
      "reasoning": "Detailed explanation with specific evidence from the image",
      "evidence_description": ["Description of visual element 1", ...],
      "missing_elements": ["Element 1 that would improve satisfaction", ...]
    }},
    ...
  }}
}}
```
**Important**: The keys in "evaluations" MUST be the string representation of the Criterion IDs provided above (e.g., "{items_with_indices[0][0]}", "{items_with_indices[1][0] if len(items_with_indices)>1 else '...'}", etc.).
Ensure your response is ONLY the JSON object, with no additional text."""

    try:
        ori_b64 = image_to_base64(ori_img_path)
        gen_b64 = image_to_base64(gen_img_path)
    except Exception as e:
        return {"error": f"Image read error: {e}"}

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{ori_b64}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}"}
                    }
                ]
            }
        ],
        "temperature": 0.0 # Deterministic evaluation
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    for attempt in range(max_retries):
        wait_if_paused()
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            
            # Clean up content to ensure JSON
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
            
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"Request failed ({attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
            else:
                return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images against rubrics.")
    parser.add_argument("--json_file", type=str, default="/Users/rswang/Desktop/illubench/illubench.json", help="Path to the JSON file containing rubrics.")
    parser.add_argument("--ori_folder", type=str, required=True, help="Folder containing original images.")
    parser.add_argument("--gen_folder", type=str, nargs='+', required=True, help="Folder(s) containing generated images.")
    parser.add_argument("--model_name", type=str, nargs='+', required=True, help="Name(s) of the model being evaluated (used for output filename).")
    parser.add_argument("--output_folder", type=str, default="evaluation_results", help="Folder to save results.")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of parallel worker threads.")
    
    args = parser.parse_args()

    if len(args.gen_folder) != len(args.model_name):
        print("Error: The number of generator folders must match the number of model names.")
        sys.exit(1)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(args.json_file, 'r', encoding='utf-8') as f:
        illubench_data = json.load(f)

    # Flatten data if it's a list of dicts
    data_map = {}
    if isinstance(illubench_data, list):
        for item in illubench_data:
            data_map.update(item)
    else:
        data_map = illubench_data

    # Iterate over each model
    for gen_folder, model_name in zip(args.gen_folder, args.model_name):
        print(f"Processing model: {model_name}")
        
        output_file = os.path.join(args.output_folder, f"{model_name}_eval_results.json")
        
        # Load existing results if any, to resume
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            print(f"Resuming from existing results file: {output_file}")
        else:
            results_data = {}

        tasks = []
        
        # Iterate over images
        for img_name, info in data_map.items():
            if "rubric" not in info:
                continue
                
            ori_path = os.path.join(args.ori_folder, img_name)
            
            # Determine generated image filename
            base_name = os.path.splitext(img_name)[0]
            gen_img_name = base_name + ".png"
            gen_path = os.path.join(gen_folder, gen_img_name)
            
            # Check files existence
            if not os.path.exists(ori_path):
                # Try png for original if jpg not found
                ori_path_png = os.path.join(args.ori_folder, base_name + ".png")
                if os.path.exists(ori_path_png):
                    ori_path = ori_path_png
                else:
                    continue
                
            if not os.path.exists(gen_path):
                # tqdm.write(f"Generated image missing: {gen_img_name}, skipping...")
                continue
                
            # Ensure structure in results_data matches current rubric
            if img_name not in results_data:
                results_data[img_name] = {"rubric_eval": {}}
            
            target_eval_dict = results_data[img_name]["rubric_eval"]
            
            # Group items by category for batch processing
            rubric = info["rubric"]
            for category, items in rubric.items():
                # Init category if missing
                if category not in target_eval_dict:
                    target_eval_dict[category] = json.loads(json.dumps(items))
                else:
                    # Ensure existing list is long enough (in case rubric was extended)
                    if len(items) > len(target_eval_dict[category]):
                        for i in range(len(target_eval_dict[category]), len(items)):
                            target_eval_dict[category].append(json.loads(json.dumps(items[i])))

                items_to_eval = []
                for idx, item in enumerate(items):
                    # Check if this specific item already has an evaluation
                    current_eval = target_eval_dict[category][idx]
                    if "evaluation" not in current_eval:
                        items_to_eval.append((idx, item))
                
                if items_to_eval:
                    tasks.append({
                        "img_name": img_name,
                        "ori_path": ori_path,
                        "gen_path": gen_path,
                        "category": category,
                        "items_with_indices": items_to_eval
                    })

        print(f"Found {len(tasks)} categories to evaluate (containing multiple items) for {model_name}.")
        
        def save_progress():
            with file_lock:
                # Atomic write to prevent corruption
                temp_file = output_file + ".tmp"
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(results_data, f, indent=2, ensure_ascii=False)
                    os.replace(temp_file, output_file)
                except Exception as e:
                    tqdm.write(f"Error saving progress: {e}")

        # Graceful exit handler
        def signal_handler(sig, frame):
            print("\nInterrupt received, saving progress...")
            save_progress()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

        if not tasks:
            continue

        # Process tasks
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(evaluate_rubric_category, t["ori_path"], t["gen_path"], t["items_with_indices"], t["category"]): t
                for t in tasks
            }
            
            completed_count = 0
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Evaluating {model_name}"):
                task = future_to_task[future]
                img_name_task = task["img_name"]
                category = task["category"]
                
                try:
                    result = future.result()
                    
                    if "error" in result:
                        tqdm.write(f"Error evaluating {img_name_task} {category}: {result['error']}")
                        continue

                    if "evaluations" in result:
                        for str_idx, eval_data in result["evaluations"].items():
                            try:
                                idx = int(str_idx)
                                # Store the evaluation in the correct position
                                results_data[img_name_task]["rubric_eval"][category][idx]["evaluation"] = eval_data
                            except (ValueError, IndexError, KeyError) as e:
                                tqdm.write(f"Error mapping result for {img_name_task} {category} ID {str_idx}: {e}")
                    else:
                        tqdm.write(f"Unexpected response format for {img_name_task} {category}: {result}")

                except Exception as e:
                    tqdm.write(f"Exception evaluating {img_name_task} {category}: {e}")
                
                completed_count += 1
                if completed_count % 5 == 0:
                    save_progress()
            
            save_progress()

if __name__ == "__main__":
    main()
