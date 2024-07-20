import asyncio
import json
import platform
import subprocess


class LlamaInterface:
    def __init__(self):
        self.mock_mode = False

    async def __aenter__(self):
        try:
            await self._check_connection()
        except ConnectionRefusedError:
            print("Warning: Unable to connect to Llama server. Switching to mock mode.")
            self.mock_mode = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def _check_connection(self):
        response = await self._query_llama("connection test")
        if "Error" in response:
            raise ConnectionRefusedError("Unable to connect to Llama server.")

    async def _query_llama(self, prompt):
        if self.mock_mode:
            return f"Mock response for: {prompt}"

        system = platform.system().lower()

        if system == "windows":
            command = [
                "powershell",
                "-Command",
                f"""
                $url = "http://localhost:11434/api/generate"
                $body = @{{ model = "llama3"; prompt = "{prompt}"; format = "json"; stream = $false }}
                $jsonBody = $body | ConvertTo-Json
                $response = Invoke-WebRequest -Uri $url -Method Post -Body $jsonBody -ContentType "application/json"
                $response.Content
                """,
            ]
        else:
            command = f"""
            curl http://localhost:11434/api/generate -d '{{
              "model": "llama3",
              "prompt": "{prompt}",
              "format": "json",
              "stream": false
            }}'
            """

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                response = json.loads(result.stdout)
                return response.get("response", "No response key in JSON")
            else:
                raise Exception(f"Command failed with return code {result.returncode}")
        except Exception as e:
            print(f"Error querying Llama: {e}")
            return f"Error response for: {prompt}"

    async def extract_concepts(self, text):
        prompt = f"Extract key concepts from the following text:\n\n{text}\n\nConcepts:"
        response = await self._query_llama(prompt)
        return [concept.strip() for concept in response.split(",")]

    async def process(self, task):
        return await self._query_llama(task)
