# 🔑 API Key Setup Guide

## Option 1: Using Hugging Face Inference API (Recommended for Hackathon)

### Step 1: Create Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Click **Sign Up** in the top right
3. Complete registration with email or social login

### Step 2: Accept Model License
1. Visit the model page: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (or your chosen model)
2. Click **Accept** to agree to the license terms
3. Your account must be approved by Meta

### Step 3: Generate API Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Give it a name (e.g., "OpenEnv Hackathon")
4. Select **Read** permission (sufficient for inference)
5. Click **Create token**
6. Copy the token (starts with `hf_`)

### Step 4: Update .env File
```env
API_BASE_URL=https://api-inference.huggingface.co/models
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
HF_TOKEN=hf_your_token_here
```

---

## Option 2: Using OpenAI API (Higher Costs)

### Step 1: Create OpenAI Account
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in with your account
3. Verify email if required

### Step 2: Generate API Key
1. Click your profile icon (bottom left)
2. Select **API keys**
3. Click **Create new secret key**
4. Copy the key (starts with `sk-`)
5. **Keep this secret!** Don't share it

### Step 3: Add Billing
1. Go to [Billing → Overview](https://platform.openai.com/account/billing/overview)
2. Add a payment method (credit/debit card)
3. Monitor usage as API calls incur costs

### Step 4: Update .env File
```env
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4
HF_TOKEN=sk_your_openai_key_here
```

---

## Option 3: Using Together AI (Budget-Friendly)

### Step 1: Create Together AI Account
1. Go to [together.ai](https://www.together.ai)
2. Sign up with email
3. Verify your account

### Step 2: Generate API Key
1. Go to your [API dashboard](https://api.together.xyz/settings/api-keys)
2. Click **Create new API key**
3. Copy the key

### Step 3: Update .env File
```env
API_BASE_URL=https://api.together.xyz/v1
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
HF_TOKEN=your_together_key_here
```

---

## Option 4: Using Local LLM (Completely Free)

### Setup with Ollama
1. Download [Ollama](https://ollama.ai)
2. Install and run: `ollama pull llama2`
3. Start the server: `ollama serve` (runs on `http://localhost:11434`)

### Update .env File
```env
API_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama2
HF_TOKEN=not-needed
```

---

## Quick Setup Cheatsheet

```bash
# 1. Create .env file from template
cp .env.example .env

# 2. Edit .env with your credentials
# Choose ONE option above and fill in:
#   - API_BASE_URL
#   - MODEL_NAME
#   - HF_TOKEN

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test the inference script
python inference.py

# 5. You should see:
# [START]
# [STEP] step_id: 1, task: ..., score: ...
# [END]
```

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `API_BASE_URL` | Base URL of LLM API | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4` or `meta-llama/Llama-2-7b-chat-hf` |
| `HF_TOKEN` | API authentication key | `sk_...` or `hf_...` |

---

## Troubleshooting

### Error: "API_BASE_URL not set"
→ Make sure `.env` file exists and is in the root project directory

### Error: "401 Unauthorized"
→ Check your `HF_TOKEN` is correct and not expired

### Error: "Model not found"
→ Verify `MODEL_NAME` is spelled correctly

### Error: "Rate limit exceeded"
→ Wait a few minutes before retrying, or upgrade your API plan

---

## Cost Estimates (Approximate)

| Provider | Model | Cost per 1M tokens | Free Tier |
|----------|-------|------------------|-----------|
| OpenAI | GPT-4 | $30 (input), $60 (output) | No |
| OpenAI | GPT-3.5 | $0.50 (input), $1.50 (output) | No |
| Hugging Face | Various | Free (limited) | Yes (75K requests/month) |
| Together AI | Llama 2 | $0.70 | Yes |
| Local (Ollama) | Llama 2 | Free (after download) | Yes |

---

**Note for Hackathon:** We recommend **Hugging Face or Local Ollama** to avoid costs during development!
