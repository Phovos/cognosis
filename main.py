#!/usr/bin/env python3
import asyncio
import platform
from functools import partial
from runtime import Logger, AppBus, SymbolicKernel

App = AppBus("AppBus")

def log_error(error: Exception):
    logger.error(f"Error occurred: {error}")

logger = Logger("MainLogger")
logger.info(f"Starting main.py on {platform.system()}")

async def usermain(failure_threshold=10) -> bool:
    user_logger = Logger("UserMainLogger")

    async def do_something() -> bool:
        user_logger.info("The user had control of the application kernel.")
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

async def main():
    try:
        try:
            kb_dir = "/path/to/kb_dir"
            output_dir = "/path/to/output_dir"
            max_memory = 1024
            async with SymbolicKernel(kb_dir, output_dir, max_memory) as kernel:
                result = await kernel.process_task("Wouldn't it be cool if I could pass you python objects?")
                print(result)
                status = kernel.get_status()
                print(status)
                response = await kernel.query("Thanks for the message.")
                print(response)
        except Exception as e:
            logger.error(f"Failed to run SymbolicKernel: {e}")

        if isinstance(CurriedUsermain, partial):
            try:
                await asyncio.wait_for(CurriedUsermain(), timeout=60)
            except asyncio.TimeoutError:
                logger.error("CurriedUsermain timed out")
            except Exception as e:
                log_error(e)
        else:
            await CurriedUsermain()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        logger.info("Exiting...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        logger.warning("RuntimeError detected: asyncio.run() cannot be called from a running event loop.")
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        except Exception as e:
            logger.error(f"An error occurred while handling the existing event loop: {str(e)}", exc_info=True)
