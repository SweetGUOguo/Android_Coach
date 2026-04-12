import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from verl.utils.dataset.utils import USER_PROMPT_WThought
from qwen_vl_utils import process_vision_info
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np

CONFIG = {
    "csv_file": 'data/prm_data_examples/ac_prm_train_dataset_mini.csv',
    "output_csv_file": 'YOUR_OUTPUT_PATH',
    "model_path": 'checkpoints/prm-sft-androidcontrol/global_step_1404/huggingface',
    "num_gpus": 8,
    "result_column_name": "generated_text"
}

def inference_worker(worker_id, gpu_id, dataset_chunk, results_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    llm = LLM(
        model=CONFIG["model_path"],
        tensor_parallel_size=1, 
        limit_mm_per_prompt={"image": 10, "video": 10}, 
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(CONFIG["model_path"])
    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.001, repetition_penalty=1.05,
        max_tokens=256, stop_token_ids=[],
    )
    progress_bar = tqdm(
        dataset_chunk.iterrows(), 
        total=len(dataset_chunk), 
        desc=f"Worker-{worker_id} (GPU-{gpu_id})"
    )

    for idx, info in progress_bar:
        image_path = info['NAS_PATH']
        task = info['PROMPT']
        action = info['RESPONSE']

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Your are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT_WThought.format(instruction=task, agent_thought_and_action=action)},
            ]}
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        mm_data = {"image": image_inputs} if image_inputs is not None else {}
        llm_inputs = {"prompt": text, "multi_modal_data": mm_data}
        
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        print(generated_text)
        results_queue.put({
            "index": idx,
            CONFIG["result_column_name"]: generated_text
        })
        
    print(f"Worker-{worker_id} (GPU-{gpu_id}) Finished.")

def parse_prediction(generated_text):
    text = str(generated_text).strip()
    
    if 'score:' in text.lower():
        score_part = text.lower().split('score:')[1].strip()
        if score_part.startswith('true'):
            return True
        elif score_part.startswith('false'):
            return False
        elif 'true' in score_part or '1' in score_part:
            return True
        elif 'false' in score_part or '0' in score_part:
            return False
    
    return None

def main():
    mp.set_start_method('spawn', force=True)
    print("--- Read and prepare dataset ---")
    dataset = pd.read_csv(CONFIG["csv_file"])
    dataset['GPT_JUDGE'] = dataset['GPT_JUDGE'].astype(bool)
    print(f"dataset loaded successfully, total {len(dataset)} records.")

    dataset_chunks = np.array_split(dataset, CONFIG["num_gpus"])
    print(f"dataset has been split into {len(dataset_chunks)} chunks for parallel processing.")
    print("\n--- Start multi-process parallel inference (using mp.Process) ---")
    
    manager = mp.Manager()
    results_queue = manager.Queue()
    processes = []

    for i in range(CONFIG["num_gpus"]):
        args = (i, i, dataset_chunks[i], results_queue)
        
        p = mp.Process(target=inference_worker, args=args, daemon=False)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n--- Inference completed, merging results from queue ---")
    # Retrieve all results from the queue
    final_results = []
    while not results_queue.empty():
        final_results.append(results_queue.get())
    
    results_df = pd.DataFrame(final_results).set_index('index')
    dataset = dataset.join(results_df)

    print("\n--- calculate acc ---")
    dataset['predicted_value'] = dataset[CONFIG["result_column_name"]].apply(parse_prediction)
    valid_predictions = dataset.dropna(subset=['predicted_value', 'GPT_JUDGE'])
    # valid_predictions = dataset.dropna(subset=['predicted_value'])
    correct_predictions = (valid_predictions['predicted_value'] == valid_predictions['GPT_JUDGE']).sum()
    # correct_predictions = (valid_predictions['predicted_value'] == 'true').sum()
    total_valid = len(valid_predictions)
    accuracy = correct_predictions / total_valid if total_valid > 0 else 0
    
    tt_count = ((valid_predictions['predicted_value'] == True) & (valid_predictions['GPT_JUDGE'] == True)).sum()
    ff_count = ((valid_predictions['predicted_value'] == False) & (valid_predictions['GPT_JUDGE'] == False)).sum()
    ft_count = ((valid_predictions['predicted_value'] == True) & (valid_predictions['GPT_JUDGE'] == False)).sum()
    tf_count = ((valid_predictions['predicted_value'] == False) & (valid_predictions['GPT_JUDGE'] == True)).sum()

    print(f"Valid Predictions: {total_valid}/{len(dataset)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Model Accuracy (ACC): {accuracy:.4f}")
    print("-" * 20)
    print(f"Predicted True, Actual True (TT): {tt_count}")
    print(f"Predicted False, Actual False (FF): {ff_count}")
    print(f"Predicted True, Actual False (FT/FP): {ft_count}")
    print(f"Predicted False, Actual True (TF/FN): {tf_count}")
    print("-" * 20)


    print("\n--- Saving results to CSV ---")
    dataset.to_csv(CONFIG["output_csv_file"], index=False, encoding='utf-8')
    print(f"All results have been saved to: {CONFIG['output_csv_file']}")

if __name__ == '__main__':
    main()