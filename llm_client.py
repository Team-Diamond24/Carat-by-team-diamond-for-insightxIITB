import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=30.0)
        
        # Primary model, default to deepseek/deepseek-chat if not specified
        self.primary_model = os.environ.get("PRIMARY_MODEL", "deepseek/deepseek-chat")
        
        # Fallback model, if needed
        self.fallback_model = os.environ.get("FALLBACK_MODEL", "openai/gpt-3.5-turbo")

    def call_model(self, prompt: str, use_fallback: bool = False) -> dict:
        """
        Call the LLM. 
        Returns a dict: {"content": str, "model_used": str}
        """
        model_to_use = self.fallback_model if use_fallback else self.primary_model
        
        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            content = response.choices[0].message.content
            return {
                "content": content,
                "model_used": model_to_use,
                "fallback_used": use_fallback
            }
        except Exception as e:
            print(f"[LLMClient] Error calling {model_to_use}: {e}")
            raise e

llm_client = LLMClient()
