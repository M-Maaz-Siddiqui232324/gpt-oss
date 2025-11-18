"""LLM engine for text generation using Ollama"""
import logging
import requests

logger = logging.getLogger(__name__)


class LLMEngine:
    """Manages LLM loading and inference via Ollama"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.model = None
        self.device = "ollama"
        logger.info(f"Initialized LLMEngine (model={model_name}, url={base_url})")
    
    def load_model(self) -> bool:
        """Check if Ollama is running and model is available"""
        logger.info(f"Checking Ollama connection and model: {self.model_name}")
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error(f"Ollama API returned status {response.status_code}")
                return False
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            logger.info(f"Available Ollama models: {model_names}")
            
            if self.model_name not in model_names:
                logger.error(f"Model '{self.model_name}' not found in Ollama")
                return False
            
            self.model = self.model_name
            logger.info(f"Ollama model '{self.model_name}' is ready")
            return True
        
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is Ollama running?")
            logger.info("Start Ollama with: ollama serve")
            return False
        
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            return False
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 300,
        temperature: float = 0.3,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """Generate text from prompt using Ollama"""
        logger.debug(f"Generating response (max_tokens={max_tokens}, temp={temperature})")
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repetition_penalty,
                }
            }
            
            # Define stop strings to prevent continuation
            stop_strings = [
                "USER QUESTION",
                "QUESTION",
                "ANSWER:",
                "YOUR RESPONSE",
                "USER QUESTIONS",
                "\n\nHow do",
                "\n\nWhat is",
                "\n\nCan I",
                "\n\nWhere can",
                "\n\nIs there",
                "\n\nAre there",
                "\nRemember,",
                "what if i",
                "what if you"
            ]
            
            payload["options"]["stop"] = stop_strings
            
            # Make request to Ollama
            from config import OLLAMA_TIMEOUT
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I apologize, but I encountered an error generating a response."
            
            # Parse response
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            logger.debug(f"Generated response length: {len(generated_text)} chars")
            
            # Additional cleanup for stop strings (in case Ollama didn't stop)
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0].strip()
                    logger.debug(f"Stopped generation at: {stop_str}")
                    break
            
            return generated_text
        
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "I apologize, but the request timed out. Please try again."
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
