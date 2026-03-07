import pandas as pd
import json 
from pathlib import Path
from openai import OpenAI
import os
from time import sleep

API_KEY = os.getenv("OPENAI_API_KEY_UNIVERSITY")
client = OpenAI(api_key=API_KEY)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "inputs"
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs"
BATCH_FILES_DIR = BASE_DIR / "batch_files"
API_INPUT_DIR = BATCH_FILES_DIR / "input"
API_JOBS_DIR = BATCH_FILES_DIR / "job"
API_OUTPUT_DIR = BATCH_FILES_DIR / "output"

def create_prompt(prompt_filename: str, input: str, input_placeholder: str = None) -> str:
    prompt_path = PROMPTS_DIR / (prompt_filename + ".txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt_template = file.read()
    if input_placeholder is None:
        prompt = prompt_template + "\n" + input
    else:
        prompt = prompt_template.replace(input_placeholder, input)
    return prompt

def create_batch_input_file(batch_input_filename: str, user_prompt_filename: str, system_prompt_filename: str,
                            inputs: pd.DataFrame, input_column: str, id_column: str, 
                            input_placeholder: str = None, model = 'gpt-5', temperature: float = None) -> Path:
    
    batch_input_path = API_INPUT_DIR / (batch_input_filename + ".jsonl")
    
    # 1. Pre-load prompts to avoid reading files inside the loop
    system_prompt_content = None
    if system_prompt_filename:
        system_prompt_content = open(PROMPTS_DIR / (system_prompt_filename + ".txt"), "r", encoding="utf-8").read()

    # 2. Open file in write mode
    with open(batch_input_path, "w", encoding="utf-8") as file:
        
        for _, row in inputs.iterrows():
            # A. Build the messages list for THIS specific request
            messages = []
            
            # Add System Prompt (if exists)
            if system_prompt_content:
                messages.append({"role": "system", "content": system_prompt_content})
            
            # Add User Prompt
            user_input = row[input_column]
            # Assuming create_prompt returns the final string
            user_prompt_text = create_prompt(user_prompt_filename, user_input, input_placeholder)
            messages.append({"role": "user", "content": user_prompt_text})

            # B. Construct the Body (The actual parameters for GPT)
            body = {
                "model": model,
                "messages": messages
            }
            if temperature is not None:
                body["temperature"] = temperature

            # C. Construct the Batch Request Envelope
            # This is the specific format OpenAI Batch API requires
            request_object = {
                "custom_id": str(row[id_column]), # specific ID for this row
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }

            # D. Write as a single line JSON
            file.write(json.dumps(request_object, ensure_ascii=False) + "\n")

    return batch_input_path

# def send_input_to_api(batch_input_filename: str, job_filename: str) -> Path:
#     job_path = API_JOBS_DIR / (job_filename + ".json")
#     batch_input_path = API_INPUT_DIR / (batch_input_filename + ".jsonl")
#     print(f"Sending batch input file to API: {batch_input_path}")
#     with open(batch_input_path, "r", encoding="utf-8") as file:
#         batch_data = json.load(file)
#     uploaded = client.files.create(file=open(batch_input_path, "rb"), purpose="batch")
#     batch = client.batch.create(
#         input_file_id=uploaded.id,
#         model=batch_data["model"],
#         metadata={"job_filename": job_filename},
#         temperature=batch_data.get("temperature", None),
#     )
#     with open(job_path, "w", encoding="utf-8") as file:
#         json.dump(batch.dict(), file, ensure_ascii=False, indent=4)
#     print(f"Batch job created with ID: {batch.id}, job file saved at: {job_path}")
#     return job_path

def send_input_to_api(batch_input_filename, job_filename):
    batch_input_path = API_INPUT_DIR / (batch_input_filename + ".jsonl")
    
    print(f"Sending batch input file to API: {batch_input_path}")
    
    # 1. Upload File
    with open(batch_input_path, "rb") as file:
        uploaded = client.files.create(file=file, purpose="batch")
    
    # 2. Create Batch Job
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"job_filename": job_filename}
    )

    # 3. Save Job Info locally (so you can retrieve the ID later)
    # Ensure filename has .json extension
    if not job_filename.endswith(".json"):
        save_filename = job_filename + ".json"
    else:
        save_filename = job_filename

    job_file_path = API_JOBS_DIR / save_filename
    
    with open(job_file_path, "w", encoding="utf-8") as f:
        # batch.json() serializes the API response object to a JSON string
        f.write(batch.json()) 

    print(f"✅ Job created: {batch.id}")
    print(f"   Info saved to: {job_file_path}")
    
    return batch

def wait_for_batch_completion(job_filename: str, poll_interval: int = 30) -> None:
    job_path = API_JOBS_DIR / (job_filename + ".json")
    with open(job_path, "r", encoding="utf-8") as file:
        job_data = json.load(file)
    batch_id = job_data["id"]
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            print(f"Batch job {batch_id} completed successfully.")
            break
        elif batch.status == "failed":
            print(f"Batch job {batch_id} failed.")
            break
        else:
            print(f"Batch job {batch_id} status: {batch.status}. Checking again in {poll_interval} seconds...")
            sleep(poll_interval)

def retrieve_api_output(job_filename: str, output_filename: str) -> Path:
    job_path = API_JOBS_DIR / (job_filename + ".json")
    output_path = API_OUTPUT_DIR / (output_filename + ".jsonl")
    
    with open(job_path, "r", encoding="utf-8") as file:
        job_data = json.load(file)
        
    batch_id = job_data["id"]
    batch = client.batches.retrieve(batch_id)
    output_file_id = batch.output_file_id
    
    file_response = client.files.content(output_file_id)
    
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(file_response.text)
        
    print(f"Batch job output retrieved for ID: {batch_id}, output file saved at: {output_path}")
    return output_path

def output_jsonl_to_dataframe(output_filename: str, save: bool = True) -> pd.DataFrame:
    output_path = API_OUTPUT_DIR / (output_filename + ".jsonl")
    records = []

    with open(output_path, "r", encoding="utf-8") as file:
        for line in file:
            raw = json.loads(line)

            row = {
                # identifiers
                "batch_id": raw.get("id"),
                "custom_id": raw.get("custom_id"),

                # defaults (in case of errors)
                "status_code": None,
                "model": None,
                "answer_text": None,

                # token usage
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
                "cached_tokens": None,
            }

            response = raw.get("response")
            if response:
                row["status_code"] = response.get("status_code")

                body = response.get("body")
                if body:
                    row["model"] = body.get("model")

                    # extract answer text
                    choices = body.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        row["answer_text"] = message.get("content")

                    # extract usage
                    usage = body.get("usage", {})
                    row["prompt_tokens"] = usage.get("prompt_tokens")
                    row["completion_tokens"] = usage.get("completion_tokens")
                    row["total_tokens"] = usage.get("total_tokens")

                    completion_details = usage.get("completion_tokens_details", {})
                    row["reasoning_tokens"] = completion_details.get("reasoning_tokens")

                    prompt_details = usage.get("prompt_tokens_details", {})
                    row["cached_tokens"] = prompt_details.get("cached_tokens")

            records.append(row)

    df = pd.DataFrame(records)

    if save:
        df.to_excel(API_OUTPUT_DIR / (output_filename + ".xlsx"), index=False)

    return df

def run_whole_batch_process(batch_filename_prefix: str, user_prompt_filename: str, inputs: pd.DataFrame, input_column: str, id_column: str="id", 
                            input_placeholder: str = None, system_prompt_filename: str = None,
                            model = 'gpt-5', temperature: float = None, poll_interval: int = 30, save: bool = True) -> pd.DataFrame:
    batch_input_filename = f"{batch_filename_prefix}_input"
    job_filename = f"{batch_filename_prefix}_job"
    output_filename = f"{batch_filename_prefix}_output"
    create_batch_input_file(batch_input_filename, user_prompt_filename, system_prompt_filename,
                            inputs, input_column, id_column, input_placeholder, model, temperature)
    send_input_to_api(batch_input_filename, job_filename)
    wait_for_batch_completion(job_filename, poll_interval)
    retrieve_api_output(job_filename, output_filename)
    return output_jsonl_to_dataframe(output_filename, save)