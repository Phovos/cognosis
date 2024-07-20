#!/usr/bin/env python3
import asyncio
import logging
import platform
from functools import partial

import runtime
from src.app.kernel import SymbolicKernel
from src.app.llama import LlamaInterface
from src.models import EventBus, User, AtomicData, FormalTheory

# Logger setup
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger

logger = setup_logger("MainLogger")
event_bus = EventBus()

async def usermain(failure_threshold=10) -> bool:
    user_logger = setup_logger("UserMainLogger")

    async def do_something() -> bool:
        # Example function that simulates an operation
        return True

    try:
        result = await do_something()
        if result:
            user_logger.info("usermain successful, returns True")
            return True
    except Exception as e:
        user_logger.error(f"Failed with error: {e}")
        return False

    failure_count = sum(
        1 for _ in range(failure_threshold) if not await usermain(failure_threshold)
    )
    failure_rate = failure_count / failure_threshold
    user_logger.info(f"Failure rate: {failure_rate:.2%}")
    return failure_rate < 1.0

CurriedUsermain = partial(usermain, failure_threshold=10)

def log_error(error: Exception):
    logger.error(f"Error occurred: {error}")

async def main():
    try:
        logger.info("Starting runtime")
        runtime.main()

        # Initialize and run the SymbolicKernel
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

        # Example usage of models
        user = User(ID=1, name="John Doe")
        print(user.json())

        atomic_data = AtomicData("Sample data")
        print(atomic_data.encode())

        theory = FormalTheory()
        theory.add_axiom("sample_axiom", lambda x: x * 2)
        print(theory.execute("sample_axiom", 5))

        # Example usage of EventBus
        def handle_event(data):
            print(f"Received event: {data}")

        event_bus.subscribe("sample_event", handle_event)
        event_bus.publish("sample_event", "Hello, EventBus!")

        if platform.system() == "Windows":
            try:
                await asyncio.wait_for(CurriedUsermain(), timeout=60)  # 60 seconds timeout
            except asyncio.TimeoutError:
                logger.error("CurriedUsermain timed out on Windows")
            except Exception as e:
                log_error(e)
        else:
            await CurriedUsermain()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        logger.info("Exiting...")

if __name__ == "__main__":
    asyncio.run(main())