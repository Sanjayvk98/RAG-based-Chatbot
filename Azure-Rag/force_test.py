import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Extract and sanitize values
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME") or os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

print("====================================")
print(f"DEBUGGING TARGET:")
print(f"Endpoint:   {endpoint}")
print(f"Deployment: {deployment}")
print("====================================\n")

# Array of versions to attempt
api_versions = ["2024-06-01", "2024-02-01", "2023-05-15"]

for version in api_versions:
    print(f"🔄 Attempting connection using API Version: '{version}'...")
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=version
        )
        
        response = client.embeddings.create(
            input=["Test connectivity"],
            model=deployment
        )
        print(f"🎉 SUCCESS WITH VERSION {version}! Service is fully online.\n")
        print(f"-> Use api_version='{version}' in your App initialization code.")
        exit(0)
    except Exception as e:
        print(f"❌ Failed with version {version}: {e}\n")

print("🚨 ALL ATTEMPTS FAILED.")
print("CRITICAL CHECKLIST:")
print(f"1. Log into Azure OpenAI Studio and verify that the DEPLOYMENT NAME is EXACTLY: '{deployment}'")
print("2. Make sure you aren't pasting the base model name (like text-embedding-3-small) unless that is exactly what you named the deployment.")
