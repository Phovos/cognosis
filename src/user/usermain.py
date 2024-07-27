# ~/usermain_script.py - Define usermain function

import sys
import json
import logging
import asyncio
import importlib.util
from pathlib import Path

# Assuming the main script is in the same directory or in a relatable path.
main_module_path = Path(__file__).parent / 'main.py'
main_module_path = Path(__file__).parent.parent.parent / 'modelmain.py' # change to main for candidate

state = {
    "pdm_installed": False,
    "virtualenv_created": False,
    "dependencies_installed": False,
    "lint_passed": False,
    "code_formatted": False,
    "tests_passed": False,
    "benchmarks_run": False,
    "pre_commit_installed": False
}

async def usermain():
    try:
        # Dynamically import the main module
        spec = importlib.util.spec_from_file_location("main", main_module_path)
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)

        # Run usermain if it exists in the main
        await main.usermain()
    except (ImportError, AttributeError) as e:
        logging.error("No user-defined main function found or an error occurred: %s", str(e))
    finally:
        logging.info("usermain() has control of the kernel but nothing to do. Exiting...")
        json.dump(state, sys.stdout, indent=4)

if __name__ == "__main__":
    asyncio.run(usermain())