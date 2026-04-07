# Setting Up Hugging Face Spaces for OpenEnv

## Critical Setup Steps

After the Space is created and shows as "Running", you MUST configure these environment variables:

### Step 1: Navigate to Space Settings
1. Go to your Space: https://huggingface.co/spaces/Abhirammahesh05122005/MetaENVHackathon
2. Click the **Settings** gear icon (top right)

### Step 2: Add Repository Secrets
1. Scroll down to **Repository secrets**
2. Click **Add secret** for each variable below:

| Variable Name | Value |
|---|---|
| `API_BASE_URL` | `https://api-inference.huggingface.co/models` |
| `MODEL_NAME` | `meta-llama/Llama-2-7b-chat-hf` |
| `HF_TOKEN` | Your Hugging Face API token from https://huggingface.co/settings/tokens |

### Step 3: Restart the Space
After adding all secrets:
1. Click the three dots menu (top right of Space)
2. Select **Restart Space**

### Step 4: Verify It's Running
The Space should show 🟢 **Running** status with a green indicator.

## Getting Your HF Token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Select **Read** permission
4. Click **Create token**
5. Copy the value (starts with `hf_`)

## Testing the API

Once running, test the endpoint:
```bash
curl -X POST https://huggingface.co/spaces/Abhirammahesh05122005/MetaENVHackathon/api/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "fix_csv_encoding"}'
```

You should get a 200 response with the observation data.
