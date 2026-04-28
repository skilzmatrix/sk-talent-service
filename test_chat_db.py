import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env.local")
load_dotenv(BACKEND_ROOT / ".env")

try:
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    client = create_client(url, key)
    res = client.table("chat_histories").select("conversation_id, updated_at").execute()
    print("Chat Histories in DB:")
    for row in res.data:
        print(row)
    print(f"Total: {len(res.data)}")
except Exception as e:
    import traceback
    traceback.print_exc()
