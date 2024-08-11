#!/usr/bin/env python3
# /src/external/umain.py - usermain.py wrapper for loading the UserMain module located
# in /src/user/usermain.py to enable the UserMain to include external libraries.

import importlib.util
import os
import logging
import asyncio

class UserMainWrapper:
    def __init__(self, user_main_path=None):
        if user_main_path is None:
            user_main_path = os.path.join('src', 'user', 'usermain.py')
        self.user_main_path = os.path.abspath(user_main_path)
        self.user_main_module = None

    def load_usermain(self):
        """Load the UserMain module from /src/user/usermain.py."""
        if not os.path.exists(self.user_main_path):
            logging.error(f"UserMain file not found: {self.user_main_path}")
            raise FileNotFoundError(f"UserMain file not found: {self.user_main_path}")

        logging.info(f"Loading UserMain from: {self.user_main_path}")
        spec = importlib.util.spec_from_file_location("usermain", self.user_main_path)
        self.user_main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.user_main_module)
        logging.info("UserMain module loaded successfully.")

    async def run_usermain(self):
        """Run the user-defined main function asynchronously."""
        if not self.user_main_module:
            logging.error("UserMain module is not loaded.")
            raise RuntimeError("UserMain module is not loaded.")

        if hasattr(self.user_main_module, 'usermain'):
            logging.info("Running UserMain...")
            await self.user_main_module.usermain()
            logging.info("UserMain execution completed.")
        else:
            logging.error("UserMain function not found in the loaded module.")
            raise AttributeError("UserMain function not found in the loaded module.")

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        wrapper = UserMainWrapper()
        wrapper.load_usermain()
        asyncio.run(wrapper.run_usermain())
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
