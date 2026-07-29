import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# 1. READ ENVIRONMENT VARIABLES
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip("/")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-06-01"

# Double-check which environment key you actually used in your .env file!
deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME") or os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# 2. CONSTRUCT THE EXACT URL AZURE SEES
constructed_url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"

print("--- DEBUG INITIALIZATION ---")
print(f"Endpoint:    {endpoint}")
print(f"Deployment:  {deployment}")
print(f"Target URL:  {constructed_url}\n")

if not endpoint or not deployment:
    print("❌ ERROR: One of your environment variables is empty or None!")
    exit()

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version
)

try:
    response = client.embeddings.create(
        input=["Is this endpoint reachable?"],
        model=deployment
    )
    print(f"🎉 SUCCESS! Response payload received.")
except Exception as e:
    print(f"❌ FAILED WITH ERROR:\n{e}")
