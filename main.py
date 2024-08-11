import argparse
import logging
import pathlib
import asyncio
import os
import subprocess
import sys
from typing import Any, Dict
import json

from src.app.kernel import SymbolicKernel
from src.app.llama import LlamaInterface
from src.app.model import EventBus, ActionRequest, ActionResponse, Event
from src.app.atoms import ActionRequest, ActionResponse, MultiDimensionalAtom, Token
from src.usermain import usermain


class Logger:
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            file_handler = logging.FileHandler(f"{name}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logger {name} initialized.")

    def log(self, message: str, level: int = logging.INFO):
        try:
            self.logger.log(level, message)
        except Exception as e:
            logging.error(f"Failed to log message: {e}")

    def debug(self, message: str):
        self.log(message, logging.DEBUG)

    def info(self, message: str):
        self.log(message, logging.INFO)

    def warning(self, message: str):
        self.log(message, logging.WARNING)

    def error(self, message: str, exc_info=None):
        self.logger.error(message, exc_info=exc_info)

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

def run_command(command: str, check: bool = True, shell: bool = True, timeout: int = 120) -> Dict[str, Any]:
    try:
        process = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=timeout)

        if check and process.returncode != 0:
            logging.error(f"Command '{command}' failed with return code {process.returncode}")
            logging.error(f"Error output: {stderr.decode('utf-8')}")
            return {
                "return_code": process.returncode,
                "output": stdout.decode("utf-8"),
                "error": stderr.decode("utf-8")
            }

        logging.info(f"Command '{command}' completed successfully")
        logging.debug(f"Output: {stdout.decode('utf-8')}")

    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        logging.error(f"Command '{command}' timed out and was killed.")
        return {
            "return_code": -1,
            "output": stdout.decode("utf-8"),
            "error": "Command timed out"
        }
    except Exception as e:
        logging.error(f"An error occurred while running command '{command}': {str(e)}")
        return {
            "return_code": -1,
            "output": "",
            "error": str(e)
        }

    return {
        "return_code": process.returncode,
        "output": stdout.decode("utf-8"),
        "error": stderr.decode("utf-8")
    }

def setup_app(mode: str):
    """Setup the application based on the specified mode"""
    if mode == "p":
        ensure_pip_dependencies()
    else:
        if not state["pdm_installed"]:
            ensure_pdm()
        if not state["virtualenv_created"]:
            ensure_virtualenv()
        if not state["dependencies_installed"]:
            ensure_dependencies()

        if mode == "dev":
            ensure_lint()
            ensure_format()
            ensure_tests()
            ensure_benchmarks()
            ensure_pre_commit()

    introspect()

def ensure_pdm():
    """Ensure PDM is installed"""
    if not state["pdm_installed"]:
        try:
            subprocess.run("pdm --version", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            state["pdm_installed"] = True
            logging.info("PDM is already installed.")
        except subprocess.CalledProcessError:
            logging.info("PDM not found, installing PDM...")
            run_command("pip install pdm", shell=True)
            state["pdm_installed"] = True

def ensure_virtualenv():
    """Ensure the virtual environment is created"""
    if not state["virtualenv_created"]:
        if not os.path.exists(".venv"):
            run_command("pdm venv create", shell=True)
        state["virtualenv_created"] = True
        logging.info("Virtual environment already exists.")

def ensure_dependencies():
    """Install dependencies"""
    if not state["dependencies_installed"]:
        run_command("pdm install --project ./", shell=True)
        state["dependencies_installed"] = True

def ensure_pip_dependencies():
    """Install dependencies with pip"""
    if not state["dependencies_installed"]:
        run_command("pip install -r requirements.txt", shell=True)
        state["dependencies_installed"] = True

def ensure_lint():
    """Run linting tools"""
    run_command("pdm run flake8 .", shell=True)
    run_command("pdm run black --check .", shell=True)
    run_command("pdm run mypy .", shell=True)
    state["lint_passed"] = True

def ensure_format():
    """Format the code"""
    run_command("pdm run black .", shell=True)
    run_command("pdm run isort .", shell=True)
    state["code_formatted"] = True

def ensure_tests():
    """Run tests"""
    run_command("pdm run pytest", shell=True)
    state["tests_passed"] = True

def ensure_benchmarks():
    """Run benchmarks"""
    run_command("pdm run python src/bench/bench.py", shell=True)
    state["benchmarks_run"] = True

def ensure_pre_commit():
    """Install pre-commit hooks"""
    run_command("pdm run pre-commit install", shell=True)
    state["pre_commit_installed"] = True

def prompt_for_mode() -> str:
    """Prompt the user to choose between development and non-development setup"""
    while True:
        choice = input("Choose setup mode: [d]evelopment, [n]on-development or [p]ip only? ").lower()
        if choice in ["d", "n", "p"]:
            return choice
        else:
            print("Invalid choice. Please enter 'd', 'n', or 'p'.")

def introspect():
    """Introspects the installed modules"""
    logging.info("Introspecting installed modules...")
    introspect_str = ""
    for module_name in ["pdm", "transformers"]:
        try:
            module = __import__(module_name)
            introspect_str += f"Module {module_name} is installed.\n"
        except ImportError:
            introspect_str += f"Module {module_name} is not installed.\n"
    logging.info(introspect_str)

def main():
    logger = Logger(name="Main")
    mode = prompt_for_mode()
    setup_app(mode)

if __name__ == "__main__":
    main()

async def usermain():
    try:
        import main
        await main.usermain()  # Ensure usermain is run as async function
    except ImportError:
        logging.error("No user-defined main function found. Please add a main.py file and define a usermain() function.")
    finally:
        logging.info("usermain() has control of the kernel but nothing to do. Exiting...")
    json.dump(state, sys.stdout, indent=4)