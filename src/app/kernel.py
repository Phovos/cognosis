import asyncio
from src.app.llama import LlamaInterface

class SymbolicKernel:
    def __init__(self, kb_dir, output_dir, max_memory):
        self.kb_dir = kb_dir
        self.output_dir = output_dir
        self.max_memory = max_memory
        self.llama = None
        self.running = False
        self.knowledge_base = set()  # Simplified knowledge base as a set of concepts

    async def __aenter__(self):
        self.llama = await LlamaInterface().__aenter__()  # Correctly use __aenter__ for initialization
        self.running = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.llama:
            await self.llama.__aexit__(exc_type, exc, tb)  # Correctly use __aexit__ for cleanup
        self.running = False

    async def process_task(self, task):
        if not self.running:
            raise RuntimeError("Kernel is not initialized or has been stopped")

        result = await self.llama.process(task)
        concepts = await self.llama.extract_concepts(result)
        self.knowledge_base.update(concepts)

        return result

    async def stop(self):
        if self.running:
            self.running = False
            if self.llama:
                await self.llama.__aexit__(None, None, None)  # Use __aexit__ for cleanup

    def get_status(self):
        return {"kb_size": len(self.knowledge_base), "running": self.running}

    async def query(self, query):
        if not self.running:
            raise RuntimeError("Kernel is not initialized or has been stopped")
        return await self.llama._query_llama(query)  # Assuming _query_llama is an async method

async def main():
    kb_dir = "/path/to/kb"
    output_dir = "/path/to/output"
    max_memory = 1024

    async with SymbolicKernel(kb_dir, output_dir, max_memory) as kernel:
        result = await kernel.process_task("Describe the water cycle.")
        print(result)
        status = kernel.get_status()
        print(status)

        response = await kernel.query("What are the key components of the water cycle?")
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
