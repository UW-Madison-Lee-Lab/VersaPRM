# AWS Batch Inference

This directory contains two Python scripts for performing batch inference using AWS services.

## Files

- `aws_batch_inference.py`
- `download_s3.py`

## Usage

### aws_batch_inference.py

This script splits a large input file into smaller batches, uploads them to an S3 bucket, and submits batch inference jobs.

<!-- #### Command -->

```sh
python aws_batch_inference.py --large_file_path <path_to_large_file> --model_id <model_id> --role_arn <role_arn> --input_bucket <input_bucket> --output_bucket <output_bucket> --batch_size <batch_size> --min_batch_size <min_batch_size>
```

### download_s3.py

This script downloads a folder from an S3 bucket to a local directory.

```sh
python download_s3.py --bucket <bucket_name> --folder <s3_folder> --local-dir <local_directory>
```



