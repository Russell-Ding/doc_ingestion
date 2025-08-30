# Setting Up get_runtime Function

## Overview
The system supports two ways to provide the `get_runtime` function:

### Method 1: Using bedrock_utils.py (Recommended)
1. Create a file named `bedrock_utils.py` in the project root directory
2. Implement the `get_runtime` function
3. The system will automatically detect and use it

### Method 2: Using Custom Path
1. Create your function file anywhere
2. Set the path in configuration

## Setup Instructions

### Option 1: Create bedrock_utils.py in Project Root

```bash
# From the project root directory
cp bedrock_utils_template.py bedrock_utils.py
# Edit bedrock_utils.py with your implementation
```

**File Structure:**
```
/doc_ingestion/
├── backend/
├── frontend/
├── bedrock_utils.py          # <-- Create this file here
├── bedrock_utils_template.py
└── ...
```

### Option 2: Use Environment Variable for Custom Path

Create a `.env` file in the backend directory:

```bash
# backend/.env
USE_DYNAMIC_BEDROCK_RUNTIME=true
BEDROCK_RUNTIME_FUNCTION_PATH=/absolute/path/to/your/bedrock_module.py

# Example paths:
# Mac/Linux: /Users/username/projects/my_bedrock_utils.py
# Windows: C:/Users/username/projects/my_bedrock_utils.py
```

### Option 3: Use Relative Path from Backend

If your function is in a relative location:

```bash
# backend/.env
USE_DYNAMIC_BEDROCK_RUNTIME=true
BEDROCK_RUNTIME_FUNCTION_PATH=../bedrock_utils.py  # One level up from backend
```

## Implementation Examples

### Example 1: Simple AWS Profile
```python
# bedrock_utils.py
import boto3

def get_runtime():
    """Use default AWS credentials or profile"""
    return boto3.client('bedrock-runtime', region_name='us-east-1')
```

### Example 2: Dynamic Credentials
```python
# bedrock_utils.py
import boto3
import requests

def get_runtime():
    """Fetch credentials from your service"""
    # Get fresh credentials from your service
    response = requests.get('https://your-service/api/aws-credentials')
    creds = response.json()
    
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=creds['access_key'],
        aws_secret_access_key=creds['secret_key'],
        aws_session_token=creds.get('session_token')
    )
```

### Example 3: Using AWS SSO
```python
# bedrock_utils.py
import boto3

def get_runtime():
    """Use AWS SSO profile"""
    session = boto3.Session(profile_name='your-sso-profile')
    return session.client('bedrock-runtime', region_name='us-east-1')
```

## Verification

### 1. Test Your Function Directly
```python
# test_bedrock.py
from bedrock_utils import get_runtime

# Test the function
client = get_runtime()
print("✅ Bedrock client created successfully")

# Test with a simple API call
try:
    response = client.list_foundation_models()
    print(f"✅ Found {len(response['modelSummaries'])} models")
except Exception as e:
    print(f"❌ Error: {e}")
```

### 2. Check Backend Logs
When you start the backend, you should see:
```
INFO: Loaded get_runtime from bedrock_utils module
INFO: Bedrock service initialized successfully mode=dynamic
```

## Troubleshooting

### Error: "Function 'get_runtime' not found"
- Ensure your function is named exactly `get_runtime` (not `get_bedrockruntime` or other variations)
- Check that the file path is correct

### Error: "No module named 'bedrock_utils'"
- Make sure `bedrock_utils.py` is in the project root directory
- Or set `BEDROCK_RUNTIME_FUNCTION_PATH` to the correct path

### Error: "Failed to initialize dynamic Bedrock runtime"
- Check that your function returns a valid boto3 client
- Verify AWS credentials are accessible
- Check AWS permissions for bedrock:InvokeModel

## Environment Variables Reference

```bash
# backend/.env

# Enable dynamic runtime (required)
USE_DYNAMIC_BEDROCK_RUNTIME=true

# Optional: Path to your function file
# If not set, looks for bedrock_utils.py in project root
BEDROCK_RUNTIME_FUNCTION_PATH=/path/to/your/module.py

# AWS Settings (used by your function)
AWS_REGION=us-east-1
AWS_PROFILE=your-profile  # If using profiles

# Optional: For static credentials mode (not recommended)
# USE_DYNAMIC_BEDROCK_RUNTIME=false
# AWS_ACCESS_KEY_ID=your-key
# AWS_SECRET_ACCESS_KEY=your-secret
```

## Best Practices

1. **Never hardcode credentials** in your function
2. **Use environment variables** or external services for credentials
3. **Implement error handling** in your function
4. **Add logging** to debug credential issues
5. **Cache clients** if credentials don't change frequently

## Security Notes

- Keep `bedrock_utils.py` in `.gitignore` if it contains sensitive logic
- Use IAM roles when running on AWS infrastructure
- Rotate credentials regularly
- Limit permissions to only what's needed for Bedrock