# app.py
from atom import Atom, AtomicData, FormalTheory
from data import Event, ActionRequest, ActionResponse
from typing import Any, Dict, List, Union
import json
import logging

class AtomicBot:
    def __init__(self):
        self.formal_theory = FormalTheory()
        self.logger = logging.getLogger("AtomicBot")

    def handle_event(self, event_data: Dict[str, Any]):
        try:
            event = Event(**event_data)
            event.validate()
            # Process the event
            self.logger.info(f"Processed event: {event.id}")
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid event data: {e}")

    def handle_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action_request = ActionRequest(**action_data)
            action_request.validate()
            # Process the action
            result = self.process_action(action_request)
            response = ActionResponse(status="ok", retcode=0, data=result)
            response.validate()
            return vars(response)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid action request: {e}")
            return vars(ActionResponse(status="failed", retcode=10001, data={}, message=str(e)))

    def process_action(self, action_request: ActionRequest) -> Dict[str, Any]:
        # Implement action processing logic here
        return {"result": "Action processed successfully"}

def main():
    bot = AtomicBot()
    
    # Example usage
    event_data = {
        "id": "123",
        "type": "message",
        "message": [{"type": "text", "data": {"text": "Hello"}}]
    }
    bot.handle_event(event_data)

    action_data = {
        "action": "send_message",
        "params": {"message": "Hello, world!"}
    }
    response = bot.handle_action(action_data)
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()