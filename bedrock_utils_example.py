"""
Example implementation of bedrock_utils.py with get_bedrockruntime function
This shows different ways to implement the function based on your use case
"""

import boto3
import os
from typing import Any

# Method 1: Simple implementation with environment variables
def get_runtime():
    """
    Get Bedrock runtime client with credentials from environment variables
    """
    # Option 1: Use AWS credentials from environment
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        # AWS SDK will automatically use these env vars:
        # AWS_ACCESS_KEY_ID
        # AWS_SECRET_ACCESS_KEY
        # AWS_SESSION_TOKEN (optional)
    )


# Method 2: Using AWS profile
def get_runtime_with_profile():
    """
    Get Bedrock runtime client using a specific AWS profile
    """
    session = boto3.Session(profile_name='your-aws-profile-name')
    return session.client('bedrock-runtime', region_name='us-east-1')


# Method 3: With explicit credentials (not recommended for production)
def get_runtime_with_explicit_credentials():
    """
    Get Bedrock runtime client with explicit credentials
    """
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id='YOUR_ACCESS_KEY_ID',
        aws_secret_access_key='YOUR_SECRET_ACCESS_KEY'
    )


# Method 4: Using STS role assumption
def get_runtime_with_role():
    """
    Get Bedrock runtime client by assuming an IAM role
    """
    sts_client = boto3.client('sts')
    
    # Assume the role
    response = sts_client.assume_role(
        RoleArn='arn:aws:iam::123456789012:role/YourBedrockRole',
        RoleSessionName='bedrock-session'
    )
    
    # Get temporary credentials
    credentials = response['Credentials']
    
    # Create bedrock client with temporary credentials
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )


# Method 5: Dynamic credentials from external service
def get_runtime_dynamic():
    """
    Get Bedrock runtime client with dynamically fetched credentials
    """
    # Example: Fetch credentials from your credential management service
    # This is where you'd integrate with your existing credential system
    
    # Simulated credential fetching (replace with your actual implementation)
    def fetch_dynamic_credentials():
        # Your logic to get fresh credentials
        return {
            'access_key': os.getenv('DYNAMIC_AWS_ACCESS_KEY'),
            'secret_key': os.getenv('DYNAMIC_AWS_SECRET_KEY'),
            'session_token': os.getenv('DYNAMIC_AWS_SESSION_TOKEN', None)
        }
    
    creds = fetch_dynamic_credentials()
    
    client_params = {
        'service_name': 'bedrock-runtime',
        'region_name': 'us-east-1',
        'aws_access_key_id': creds['access_key'],
        'aws_secret_access_key': creds['secret_key']
    }
    
    if creds.get('session_token'):
        client_params['aws_session_token'] = creds['session_token']
    
    return boto3.client(**client_params)


# Choose which implementation to use as the main function
# The system will look for a function named exactly "get_runtime"
get_runtime = get_runtime_dynamic  # Change this to use different method