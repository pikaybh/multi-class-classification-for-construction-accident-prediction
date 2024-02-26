# %%
from utils.osutil import  get_time
from utils.console import Console

import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from time import sleep
import argparse

# Argparse
parser = argparse.ArgumentParser(description="Fine-tuning processor.")
parser.add_argument("--openai-api-key", default=None, help="OpenAI API key.")
parser.add_argument("--log-lv", default="info", help="Logging level. (default: 'info')")
parser.add_argument("--train", default="", help="Fine-tuning file name.")
parser.add_argument("--base-model", default="gpt-3.5-turbo", help="Fine-tuning base model. (default: 'gpt-3.5-turbo')")
parser.add_argument("--retrieve", default="", help="Fine-tuning job.")
parser.add_argument("--list", help="List fine-tuning job. (param: limit)")
parser.add_argument("--calcel", help="Cancel fine-tuning job. (format: 'ft:gpt-3.5-turbo:acemeco:suffix:abc123')")
parser.add_argument("--delete", help="Delete fine-tuning job. (format: 'ft:gpt-3.5-turbo:acemeco:suffix:abc123')")
parser.add_argument("--sleep-check", default=60, help="Sleep checking term. (unit: second) (default: 60)")
args = parser.parse_args()
# API key
load_dotenv()
OPENAI_API_KEY = args.openai_api_key if args.openai_api_key else os.getenv("OPENAI_API_KEY")
# Fine Tune Model
MODEL = args.base_model  # "babbage-002"  # "gpt-3.5-turbo"
FILE = args.train  # "data/train/CSI_gpt-3.5-turbo_1000_samples_frac_8_to_2_par (13).jsonl"
logger = Console(level=args.log_lv)
client = OpenAI()
# Tracking
def tracking(fine_tuning_id : str, status : str) -> None:
    __status = client.fine_tuning.jobs.retrieve(fine_tuning_id).status
    while __status == status:
        __status = client.fine_tuning.jobs.retrieve(fine_tuning_id).status
        sleep(60)
        logger.debug(f"[id: {ft_id}] I'm not sleeping...")
    logger.info(f"Fine-tuning status: {__status}")


if __name__ == "__main__":
    # Upload file
    if FILE and not FILE.startswith("file-"):
        file_id = client.files.create(file=open(FILE, "rb"), purpose="fine-tune").id
        logger.info(f"File uploaded successfully. (file id: {file_id})")
        # Create fine-tune
        ft_id = client.fine_tuning.jobs.create(training_file=file_id, model=MODEL).id
        logger.info(f"Fine-tuning process started. (fine tune id: {ft_id})")

        logger.info(f"Fine-tuning status: {client.fine_tuning.jobs.retrieve(ft_id).status}")
        for status in ["validating_files", "running", "queued"]:
            tracking(fine_tuning_id=ft_id, status=status)
        model_id = client.fine_tuning.jobs.retrieve(ft_id).fine_tuned_model
        logger.info("{client.fine_tuning.jobs.retrieve(ft_id)}")

    if FILE and FILE.startswith("file-"):
        # Create fine-tune
        ft_id = client.fine_tuning.jobs.create(training_file=file_id, model=MODEL).id
        logger.info(f"Fine-tuning process started. (fine tune id: {ft_id})")

        logger.info(f"Fine-tuning status: {client.fine_tuning.jobs.retrieve(ft_id).status}")
        for status in ["validating_files", "running", "queued"]:
            tracking(fine_tuning_id=ft_id, status=status)
        model_id = client.fine_tuning.jobs.retrieve(ft_id).fine_tuned_model
        logger.info(f"Rertrieve: {client.fine_tuning.jobs.retrieve(ft_id)}")
    # List 10 fine-tuning jobs
    if args.list: client.fine_tuning.jobs.list(limit=args.list)
    # Retrieve the state of a fine-tune
    if args.retrieve: client.fine_tuning.jobs.retrieve(args.retrieve)
    # Cancel a job
    if args.cancel: client.fine_tuning.jobs.cancel(args.cancel)
    # List up to 10 events from a fine-tuning job
    # client.fine_tuning.jobs.list_events(fine_tuning_job_id=ft_id, limit=10)
    # Delete a fine-tuned model (must be an owner of the org the model was created in)
    if args.delete: client.models.delete(args.delete)
# %%
