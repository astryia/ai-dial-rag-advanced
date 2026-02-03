from typing import Union, Dict, List
import logging

import requests

logger = logging.getLogger(__name__)

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'
API_VERSION = '2023-12-01-preview'


class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be null or empty")
        if not deployment_name or not deployment_name.strip():
            raise ValueError("Deployment name cannot be null or empty")
        
        self.endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }

    def get_embeddings(self, inputs: Union[str, List[str]], dimensions: int) -> Dict[int, List[float]]:
        if not inputs:
            raise ValueError("Input cannot be empty")
        if dimensions <= 0:
            raise ValueError("Dimensions must be a positive integer")
        
        request_body = {
            "input": inputs,
            "dimensions": dimensions,
            "encoding_format": "float"
        }
        
        params = {
            "api-version": API_VERSION
        }
        
        try:
            response = requests.post(
                url=self.endpoint,
                headers=self.headers,
                json=request_body,
                params=params,
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data:
                raise ValueError("Invalid response format: missing 'data' field")
            
            result = {}
            for item in data["data"]:
                if "index" not in item or "embedding" not in item:
                    raise ValueError("Invalid response format: missing 'index' or 'embedding' field")
                result[item["index"]] = item["embedding"]
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling embeddings API: {e}", exc_info=True)
            if hasattr(e, 'response') and e.response is not None:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing embeddings response: {e}", exc_info=True)
            raise
