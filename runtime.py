# ~/runtime.py - Runtime for Cognosis, curryable.
import argparse
import logging
import pathlib
import asyncio
import os
import platform
import subprocess
import sys
from typing import Any, Dict

class Logger:
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            for handler in [logging.StreamHandler(), logging.FileHandler(f"{name}.log")]:
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
    def log(self, message: str, level: int = logging.INFO):
        try: self.logger.log(level, message)
        except Exception as e: logging.error(f"Failed to log message: {e}")

state: Dict[str, bool] = {k: False for k in ["pdm_installed", "virtualenv_created", "dependencies_installed", "lint_passed", "code_formatted", "tests_passed", "benchmarks_run", "pre_commit_installed"]}

def run_command(
    command: str, check: bool = True, shell: bool = False, timeout: int = 120
) -> Dict[str, Any]:
    try:
        process = subprocess.Popen(
            command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=timeout)

        if check and process.returncode != 0:
            logging.error(
                f"Command '{command}' failed with return code {process.returncode}"
            )
            logging.error(f"Error output: {stderr.decode('utf-8')}")
            return {
                "return_code": process.returncode,
                "output": stdout.decode("utf-8"),
                "error": stderr.decode("utf-8"),
            }

        logging.info(f"Command '{command}' completed successfully")
        logging.debug(f"Output: {stdout.decode("utf-8")}")

    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        logging.error(f"Command '{command}' timed out and was killed.")
        return {
            "return_code": -1,
            "output": stdout.decode("utf-8"),
            "error": "Command timed out",
        }
    except Exception as e:
        logging.error(f"An error occurred while running command '{command}': {str(e)}")
        return {
            "return_code": -1,
            "output": "",
            "error": str(e),
        }

    return {
        "return_code": process.returncode,
        "output": stdout.decode("utf-8"),
        "error": stderr.decode("utf-8"),
    }

def update_path():
    home = os.path.expanduser("~")
    local_bin = os.path.join(home, ".local", "bin")
    if local_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{local_bin}:{os.environ['PATH']}"
        logging.info(f"Added {local_bin} to PATH")

def ensure_pdm():
    """Ensure pdm is installed"""
    global state
    try:
        subprocess.run(
            "pdm --version",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        state["pdm_installed"] = True
        logging.info("pdm is already installed.")
    except subprocess.CalledProcessError:
        logging.info("pdm not found, installing pdm...")
        run_command("pip install pdm", shell=True)
        state["pdm_installed"] = True

def prompt_for_mode():
    """Prompt the user to choose between development and non-development setup"""
    while True:
        choice = input(
            "Choose setup mode: [d]evelopment or [n]on-development? "
        ).lower()
        if choice in ["d", "n"]:
            return choice
        logging.info("Invalid choice, please enter 'd' or 'n'.")

def install(mode):
    """Run installation"""
    run_command("pdm install --project ./", shell=True)
    state["dependencies_installed"] = True

def lint():
    """Run linting tools"""
    global state
    run_command("pdm run flake8 .", shell=True)
    run_command("pdm run black --check .", shell=True)
    run_command("pdm run mypy .", shell=True)
    state["lint_passed"] = True

def format_code():
    """Format the code"""
    global state
    run_command("pdm run black .", shell=True)
    run_command("pdm run isort .", shell=True)
    state["code_formatted"] = True

def test():
    """Run tests"""
    global state
    run_command("pdm run pytest", shell=True)
    state["tests_passed"] = True

def bench():
    """Run benchmarks"""
    global state
    run_command("pdm run python src/bench/bench.py", shell=True)
    state["benchmarks_run"] = True

def pre_commit_install():
    """Install pre-commit hooks"""
    global state
    run_command("pdm run pre-commit install", shell=True)
    state["pre_commit_installed"] = True

def introspect():
    """Introspect the current state and print results"""
    logging.info("Introspection results:")
    for key, value in state.items():
        logging.info(f"{key}: {'✅' if value else '❌'}")

def update_shell_environment():
    if platform.system() != "Windows":
        home = os.path.expanduser("~")
        bashrc_path = os.path.join(home, ".bashrc")
        if os.path.exists(bashrc_path):
            subprocess.run(f". {bashrc_path}", shell=True, executable="/bin/bash")
            logging.info("Updated shell environment from .bashrc")
        else:
            logging.info(".bashrc not found, shell environment might not be up to date")
    else:
        logging.info("On Windows, manual PATH update might be necessary")

def create_virtualenv():
    if not os.path.exists('.venv'):
        subprocess.run(['python', '-m', 'venv', '.venv'])
    else:
        print("Virtualenv already exists.")

def ensure_virtualenv():
    """Ensure the virtual environment is created"""
    global state
    if not os.path.exists(".venv"):
        run_command("pdm venv create", shell=True)
        state["virtualenv_created"] = True
    else:
        state["virtualenv_created"] = True
        logging.info("Virtual environment already exists.")

async def usermain():
    try:
        import main
        await main.usermain()  # Ensure usermain is run as async function
    except ImportError:
        logging.error(
            "No user-defined main function found. Please add a main.py file and define a usermain() function."
        )

def main(usermain):
    ensure_pdm()
    ensure_virtualenv()
    update_shell_environment()
    update_path()

    parser = argparse.ArgumentParser(description="Setup and run cognosis project")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["dev", "non-dev"],
        help="Setup mode: 'dev' or 'non-dev'",
    )
    parser.add_argument(
        "-u",
        "--skip-user-main",
        action="store_true",
        help="Skip running the user-defined main function",
    )
    args = parser.parse_args()
    mode = args.mode
    if not mode:
        choice = prompt_for_mode()
        mode = "dev" if choice == "d" else "non-dev"

    install(mode)

    if mode == "dev":
        lint()
        format_code()
        test()
        bench()
        pre_commit_install()

    if not args.skip_user_main:
        try:
            if not asyncio.get_event_loop().is_running():
                asyncio.run(usermain())
            else:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(usermain())
        except Exception as e:
            logging.error(f"An error occurred while running usermain: {str(e)}", exc_info=True)

    introspect()
