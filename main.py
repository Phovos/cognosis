#!/usr/bin/env python3
import asyncio
import logging
import platform
import os
from functools import partial
from typing import Dict, Any

# Import necessary modules
import runtime
from src.app.kernel import SymbolicKernel
from src.app.llama import LlamaInterface
from src.model import EventBus, create_model, Field, AtomicData, FormalTheory

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

# Create a default model profile based on common filesystem attributes
FileStat = create_model(
    "FileStat",
    st_mode=Field(int),
    st_ino=Field(int, required=False),  # Inode not available on Windows
    st_dev=Field(int, required=False),  # Device not meaningful on Windows
    st_nlink=Field(int, required=False),  # Hard links not meaningful on Windows
    st_uid=Field(int, required=False),  # User ID not available on Windows
    st_gid=Field(int, required=False),  # Group ID not available on Windows
    st_size=Field(int),
    st_atime=Field(float),  # Access time
    st_mtime=Field(float),  # Modification time
    st_ctime=Field(float)   # Creation time on Windows, change time on Unix
)

async def usermain(failure_threshold=10) -> bool:
    user_logger = setup_logger("UserMainLogger")

    async def do_something() -> bool:
        # Simulate a filesystem stat operation
        stat_info = os.stat(__file__)
        stat_data = {
            'st_mode': stat_info.st_mode,
            'st_size': stat_info.st_size,
            'st_atime': stat_info.st_atime,
            'st_mtime': stat_info.st_mtime,
            'st_ctime': stat_info.st_ctime
        }

        # Optional fields depending on OS
        optional_fields = ['st_ino', 'st_dev', 'st_nlink', 'st_uid', 'st_gid']
        for field in optional_fields:
            if hasattr(stat_info, field):
                stat_data[field] = getattr(stat_info, field)

        file_stat = FileStat(**stat_data)
        user_logger.info(f"FileStat: {file_stat.dict()}")
        return True

    try:
        result = await do_something()
        if result:
            user_logger.info("usermain successful, returns True")
            return True
    except Exception as e:
        user_logger.error(f"Failed with error: {e}")
        return False

    failure_count = sum(1 for _ in range(failure_threshold) if not await do_something())
    failure_rate = failure_count / failure_threshold
    user_logger.info(f"Failure rate: {failure_rate:.2%}")
    return failure_rate < 1.0

CurriedUsermain = partial(usermain, failure_threshold=10)

def log_error(error: Exception):
    logger.error(f"Error occurred: {error}")

async def main():
    try:
        logger.info("Starting runtime")
        runtime.main(["python", "main.py"])

        # Initialize and run the SymbolicKernel
        kb_dir = "/path/to/kb"
        output_dir = "/path/to/output"
        max_memory = 1024

        try:
            async with SymbolicKernel(kb_dir, output_dir, max_memory) as kernel:
                result = await kernel.process_task("Describe the water cycle.")
                print(result)
                status = kernel.get_status()
                print(status)

                response = await kernel.query("What are the key components of the water cycle?")
                print(response)
        except Exception as e:
            logger.error(f"Failed to run SymbolicKernel: {e}")

        # Example usage of models
        stat_info = os.stat(__file__)
        stat_data = {
            'st_mode': stat_info.st_mode,
            'st_size': stat_info.st_size,
            'st_atime': stat_info.st_atime,
            'st_mtime': stat_info.st_mtime,
            'st_ctime': stat_info.st_ctime
        }

        # Optional fields depending on OS
        optional_fields = ['st_ino', 'st_dev', 'st_nlink', 'st_uid', 'st_gid']
        for field in optional_fields:
            if hasattr(stat_info, field):
                stat_data[field] = getattr(stat_info, field)

        file_stat = FileStat(**stat_data)
        print(f"FileStat JSON: {file_stat.json()}")

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
        logger.error(f"An error occurred: {str(e)}", exc_info=True)  # Added exc_info=True for full traceback
    finally:
        logger.info("Exiting...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:  # event loop is already running
        logger.warning("RuntimeError detected: asyncio.run() cannot be called from a running event loop.")
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        except Exception as e:
            logger.error(f"An error occurred while handling the existing event loop: {str(e)}", exc_info=True)