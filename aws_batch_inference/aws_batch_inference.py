import os
import math
import argparse
import boto3
from pathlib import Path
from datetime import datetime

# Initialize the Amazon Bedrock client
bedrock = boto3.client(service_name="bedrock", region_name="us-west-2")

def split_large_file(file_path, output_folder, batch_size, min_batch_size=100):
    """
    Splits a large JSONL file into smaller files of specified batch size.
    If the last batch does not meet the minimum batch size, the last line is duplicated.

    :param file_path: Path to the input JSONL file.
    :param output_folder: Path to the folder where split files will be saved.
    :param batch_size: Desired number of records per split file.
    :param min_batch_size: Minimum number of records for a valid batch (default is 100).
    :return: List of file paths to the split files.
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    total_records = len(lines)
    num_batches = math.ceil(total_records / batch_size)
    
    split_files = []

    for i in range(num_batches):
        batch_file_path = os.path.join(output_folder, f'batch_{i+1}.jsonl')
        with open(batch_file_path, 'w') as batch_file:
            # Write the appropriate batch of records into the new file
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_records)
            batch_lines = lines[start_idx:end_idx]

            # If it's the last batch and it has fewer than the minimum number of records
            if len(batch_lines) < min_batch_size:
                last_line = batch_lines[-1]  # Get the last line
                # Repeat the last line until the batch size reaches the minimum batch size
                while len(batch_lines) < min_batch_size:
                    batch_lines.append(last_line)

            batch_file.writelines(batch_lines)
        
        split_files.append(batch_file_path)
    
    return split_files

def submit_batch_inference(file_path, input_bucket, output_bucket, input_folder, output_folder, role_arn, model_id, job_name_base, batch_number):
    """
    Submits a batch inference job with a unique job name. Uploads input files to the specified input folder in S3
    and generates output files in the specified output folder.

    :param file_path: Path to the JSONL file to submit for inference.
    :param input_bucket: S3 bucket where the input file is stored.
    :param output_bucket: S3 bucket where the output will be saved.
    :param input_folder: Folder name in the input S3 bucket.
    :param output_folder: Folder name in the output S3 bucket.
    :param role_arn: The ARN of the IAM role for batch inference.
    :param model_id: The model ID to use for inference.
    :param job_name_base: Base name of the batch job.
    :param batch_number: Unique batch number to append to the job name.
    :return: The job ARN of the submitted job.
    """
    file_name = os.path.basename(file_path)
    s3_input_uri = f"s3://{input_bucket}/{input_folder}/{file_name}"
    s3_output_uri = f"s3://{output_bucket}/{output_folder}/"

    # Generate a unique job name by appending the batch number and timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    unique_job_name = f"{job_name_base}-{batch_number}-{timestamp}"

    inputDataConfig = {
        "s3InputDataConfig": {
            "s3Uri": s3_input_uri
        }
    }

    outputDataConfig = {
        "s3OutputDataConfig": {
            "s3Uri": s3_output_uri
        }
    }

    # Submit the batch inference job
    response = bedrock.create_model_invocation_job(
        roleArn=role_arn,
        modelId=model_id,
        jobName=unique_job_name,
        inputDataConfig=inputDataConfig,
        outputDataConfig=outputDataConfig
    )

    job_arn = response.get('jobArn')
    print(f"Submitted job {unique_job_name}. Job ARN: {job_arn}")
    return job_arn

def batch_inference_pipeline(file_path, input_bucket, output_bucket, input_folder, output_folder, role_arn, model_id, job_name_base, batch_size, min_batch_size=100):
    """
    Splits the input file into batches, uploads them to S3 (within a specified folder), and submits them for inference.
    Ensures that the last batch has at least the minimum batch size by duplicating the last line if needed.

    :param file_path: Path to the large input file.
    :param input_bucket: S3 bucket where input files will be uploaded.
    :param output_bucket: S3 bucket where output files will be stored.
    :param input_folder: Folder in the input S3 bucket where the files will be uploaded.
    :param output_folder: Folder in the output S3 bucket where the output will be stored.
    :param role_arn: The ARN of the IAM role for batch inference.
    :param model_id: The model ID to use for inference.
    :param job_name_base: Base name of the batch jobs.
    :param batch_size: Desired number of records per batch.
    :param min_batch_size: Minimum number of records for a valid batch.
    """
    # Split the large file into smaller files
    output_folder_local = f'./aws_submitted_split_batches/{os.path.basename(file_path)}'
    split_files = split_large_file(file_path, output_folder_local, batch_size, min_batch_size)

    # Upload each split file to S3 and submit a separate inference job
    s3 = boto3.client('s3')

    # Create folders (if not already present) in the input and output buckets
    # S3 does not actually require an explicit folder creation, but uploading files with a prefix will create the structure.
    for batch_number, split_file in enumerate(split_files, start=1):
        # Upload the split file to the input S3 bucket under the specified folder
        file_name = os.path.basename(split_file)
        s3.upload_file(split_file, input_bucket, f"{input_folder}/{file_name}")
        print(f"Uploaded {file_name} to S3 bucket: {input_bucket}/{input_folder}/")

        # Submit the batch inference job with a unique job name
        submit_batch_inference(split_file, input_bucket, output_bucket, input_folder, output_folder, role_arn, model_id, job_name_base, batch_number)


def main(large_file_path, model_id, role_arn, input_bucket, output_bucket, batch_size, min_batch_size):
    input_folder = os.path.basename(large_file_path).split(".js")[0]
    output_folder = os.path.basename(large_file_path).split(".js")[0]
    job_name_base = os.path.basename(large_file_path).split(".js")[0]

    batch_inference_pipeline(large_file_path, input_bucket, output_bucket, input_folder, output_folder, role_arn, model_id, job_name_base, batch_size, min_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run batch inference pipeline.')
    parser.add_argument('--large_file_path', type=str, required=True, help='Path to the large file')
    parser.add_argument('--model_id', type=str, default="meta.llama3-1-70b-instruct-v1:0", help='Your model ID')
    parser.add_argument('--role_arn', type=str, required=True, help='Your IAM role ARN')
    parser.add_argument('--input_bucket', type=str, required=True, help='Your input S3 bucket')
    parser.add_argument('--output_bucket', type=str, required=True, help='Your output S3 bucket')
    parser.add_argument('--batch_size', type=int, default=25000, help='Batch size')
    parser.add_argument('--min_batch_size', type=int, default=100, help='Minimum batch size')

    args = parser.parse_args()

    main(args.large_file_path, args.model_id, args.role_arn, args.input_bucket, args.output_bucket, args.batch_size, args.min_batch_size)