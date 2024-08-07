import os
from dotenv import load_dotenv
class APIKeyNotFoundError(Exception):
    def __init__(self, key_name):
        self.message = f"API key '{key_name}' not found in configuration."
        super().__init__(self.message)


class Config:
    def __init__(self, env_file_path=None):
        if env_file_path:
            load_dotenv(env_file_path)
        else:
            load_dotenv()

        self.api_keys = {
            'FRED_API_KEY': os.getenv('FRED_API_KEY'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        }
        for key, value in self.api_keys.items():
            if not value:
                raise APIKeyNotFoundError(key_name=key)

    def get_api_key(self, key_name):
        if key_name not in self.api_keys:
            raise APIKeyNotFoundError(key_name=key_name)
        return self.api_keys.get(key_name)

# Load the config once so it can be accessed globally
config = Config()
