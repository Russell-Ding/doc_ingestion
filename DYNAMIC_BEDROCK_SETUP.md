# 🔑 Dynamic Bedrock Runtime Setup

This guide shows how to configure the system to use your dynamic key function instead of static AWS credentials.

## 🎯 Overview

The system now supports **two modes** for AWS Bedrock authentication:

1. **Dynamic Runtime Mode** (recommended for your use case) - Uses your `get_bedrockruntime` function
2. **Static Credentials Mode** - Uses traditional AWS access keys

## 🛠️ Setup for Dynamic Mode

### Step 1: Configure Environment Variables

In your `backend/.env` file:

```env
# Enable dynamic mode
USE_DYNAMIC_BEDROCK_RUNTIME=true

# AWS Region (still needed)
AWS_REGION=us-east-1

# Optional: Path to your function file (if not using bedrock_utils.py)
# BEDROCK_RUNTIME_FUNCTION_PATH=/path/to/your/bedrock_script.py

# Static credentials not needed in dynamic mode
# AWS_ACCESS_KEY_ID=  # Leave empty or comment out
# AWS_SECRET_ACCESS_KEY=  # Leave empty or comment out
```

### Step 2: Create Your Bedrock Utils Function

**Option A: Create `bedrock_utils.py` in the project root:**

```python
# bedrock_utils.py
import boto3

def get_bedrockruntime():
    """
    Your custom function to get Bedrock runtime with dynamic credentials
    Replace this with your actual implementation
    """
    
    # Your dynamic credential logic here
    # Example:
    # dynamic_creds = your_credential_service.get_latest_credentials()
    
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id='your_dynamic_key',
        aws_secret_access_key='your_dynamic_secret',
        # aws_session_token='token_if_needed'  # for STS tokens
    )
```

**Option B: Use existing file with custom path:**

If you already have your function in another file:

```env
# In .env file
BEDROCK_RUNTIME_FUNCTION_PATH=/path/to/your/existing_bedrock_script.py
```

### Step 3: Function Requirements

Your `get_bedrockruntime` function should:

- **Return**: A configured `boto3.client('bedrock-runtime')` 
- **Handle**: Dynamic credential fetching
- **Support**: Both sync and async versions (async is auto-detected)

**Examples:**

```python
# Sync version
def get_bedrockruntime():
    # Your credential fetching logic
    creds = get_latest_aws_credentials()
    
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=creds['access_key'],
        aws_secret_access_key=creds['secret_key']
    )

# Async version (optional)
async def get_bedrockruntime():
    # Your async credential fetching
    creds = await get_latest_aws_credentials_async()
    
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=creds['access_key'],
        aws_secret_access_key=creds['secret_key']
    )
```

## 🔄 How It Works

1. **System Startup**: Loads your `get_bedrockruntime` function
2. **Each AI Request**: Calls your function to get fresh credentials
3. **Automatic Refresh**: Ensures always-current credentials for Bedrock calls
4. **Error Handling**: Falls back gracefully if credential refresh fails

## 📍 File Location Options

The system looks for your function in this order:

1. **Imported module**: `import bedrock_utils` (if available in Python path)
2. **Custom path**: File specified in `BEDROCK_RUNTIME_FUNCTION_PATH`
3. **Project root**: `./bedrock_utils.py` in current directory

## 🧪 Testing Your Setup

Test your dynamic function:

```python
# Test script
from bedrock_utils import get_bedrockruntime

# Test the function
client = get_bedrockruntime()
print("✅ Bedrock runtime client created successfully")

# Test with a simple call
try:
    response = client.list_foundation_models()
    print("✅ Bedrock API call successful")
except Exception as e:
    print(f"❌ Bedrock API call failed: {e}")
```

## 🔧 Advanced Configuration

### Multiple Credential Sources

```python
def get_bedrockruntime():
    """Handle multiple credential sources with fallback"""
    
    # Try primary source
    try:
        creds = primary_credential_source()
        return create_bedrock_client(creds)
    except Exception:
        pass
    
    # Fallback to secondary source
    try:
        creds = secondary_credential_source()
        return create_bedrock_client(creds)
    except Exception:
        raise Exception("All credential sources failed")

def create_bedrock_client(creds):
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        **creds  # Assuming creds dict has the right keys
    )
```

### Caching with Expiration

```python
import time
from datetime import datetime, timedelta

_cached_client = None
_cache_expiry = None

def get_bedrockruntime():
    """Get bedrock client with 1-hour caching"""
    global _cached_client, _cache_expiry
    
    now = datetime.now()
    
    # Return cached client if still valid
    if _cached_client and _cache_expiry and now < _cache_expiry:
        return _cached_client
    
    # Fetch new credentials
    creds = fetch_fresh_credentials()
    _cached_client = boto3.client('bedrock-runtime', **creds)
    _cache_expiry = now + timedelta(hours=1)
    
    return _cached_client
```

## 🚨 Important Notes

1. **Security**: Your function is called frequently - ensure it's secure and efficient
2. **Error Handling**: The system will log errors but continue with existing client if refresh fails  
3. **Performance**: Consider caching if credential fetching is expensive
4. **Testing**: Always test your function independently before integrating

## 🔄 Switching Between Modes

To switch back to static credentials:

```env
# Disable dynamic mode
USE_DYNAMIC_BEDROCK_RUNTIME=false

# Enable static credentials
AWS_ACCESS_KEY_ID=your_static_key
AWS_SECRET_ACCESS_KEY=your_static_secret
```

## ✅ Verification

After setup, check the logs when starting the backend:

```bash
# Start backend and look for these log messages:
# ✅ "Bedrock service initialized successfully mode=dynamic"
# ✅ "Loaded get_bedrockruntime from bedrock_utils module"
```

Your system is now configured to use dynamic Bedrock credentials! 🎉