# /src/user/usermain.py - user's (not to do with initialization logic) main file
# wrapped by /src/external/umain.py to enable user-defined dependencies
import sys
import json
import logging
import asyncio
import importlib.util
from pathlib import Path
state = {
    "pdm_installed": False,
    "virtualenv_created": False,
    "dependencies_installed": False,
    # hidden dev mode state
    # "lint_passed": False,
    # "code_formatted": False,
    # "tests_passed": False,
    # "benchmarks_run": False,
    # "pre_commit_installed": False
}

async def update_state():
    """Update the first False value in the state dictionary to True asynchronously."""
    for key in state:
        if not state[key]:
            logging.info(f"Updating state: {key}")
            state[key] = True
            await asyncio.sleep(0)  # Yield control back to the event loop
            break

async def usermain():
    try:
        # Get the absolute path of the main.py file
        main_module_path = Path(__file__).parent / 'main.py'

        if not main_module_path.exists():
            # Try to find it in the parent directory
            main_module_path = Path(__file__).parent.parent / 'main.py'

        if not main_module_path.exists():
            raise FileNotFoundError("Main script not found.")

        # Dynamically import the main module
        spec = importlib.util.spec_from_file_location("main", main_module_path)
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)

        # Update the state asynchronously
        await update_state()

        # Run usermain if it exists in the main module
        if hasattr(main, 'usermain'):
            await main.usermain()
        else:
            logging.error("No user-defined main function found in main.py")
    except (ImportError, AttributeError) as e:
        logging.error("An error occurred during execution: %s", str(e))
    finally:
        logging.info("usermain() has control of the kernel but nothing to do. Exiting...")
        json.dump(state, sys.stdout, indent=4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(usermain())
