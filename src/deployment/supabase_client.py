from supabase import create_client
from src.utils.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)