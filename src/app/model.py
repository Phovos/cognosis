import asyncio
import logging
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from abc import ABC, abstractmethod

T = TypeVar('T')

# Define the Abstract Base Class for Atom
class Atom(ABC):
    @abstractmethod
    def encode(self) -> bytes:
        pass

    @abstractmethod
    def decode(self, data: bytes) -> None:
        pass

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        pass

@dataclass
class Event(Atom):
    id: str
    type: str
    detail_type: str
    message: List[Dict[str, Any]]

    def validate(self) -> bool:
        return all([
            isinstance(self.id, str),
            isinstance(self.type, str),
            isinstance(self.detail_type, str),
            isinstance(self.message, list)
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "detail_type": self.detail_type,
            "message": self.message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        return cls(
            id=data["id"],
            type=data["type"],
            detail_type=data["detail_type"],
            message=data["message"]
        )

    def encode(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    def decode(self, data: bytes) -> None:
        obj = json.loads(data.decode())
        self.id = obj['id']
        self.type = obj['type']
        self.detail_type = obj['detail_type']
        self.message = obj['message']

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        # Placeholder for user-defined implementation
        pass

@dataclass
class ActionRequest(Atom):
    action: str
    params: Dict[str, Any]
    self_info: Dict[str, Any]

    def validate(self) -> bool:
        return all([
            isinstance(self.action, str),
            isinstance(self.params, dict),
            isinstance(self.self_info, dict)
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "params": self.params,
            "self": self.self_info
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionRequest':
        return cls(
            action=data["action"],
            params=data["params"],
            self_info=data["self"]
        )

    def encode(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    def decode(self, data: bytes) -> None:
        obj = json.loads(data.decode())
        self.action = obj['action']
        self.params = obj['params']
        self.self_info = obj['self']

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        # Placeholder for user-defined implementation
        pass

@dataclass
class ActionResponse(Atom):
    status: str
    retcode: int
    data: Dict[str, Any]
    message: str = ""

    def validate(self) -> bool:
        return all([
            isinstance(self.status, str),
            isinstance(self.retcode, int),
            isinstance(self.data, dict),
            isinstance(self.message, str)
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "retcode": self.retcode,
            "data": self.data,
            "message": self.message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionResponse':
        return cls(
            status=data["status"],
            retcode=data["retcode"],
            data=data["data"],
            message=data.get("message", "")
        )

    def encode(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    def decode(self, data: bytes) -> None:
        obj = json.loads(data.decode())
        self.status = obj['status']
        self.retcode = obj['retcode']
        self.data = obj['data']
        self.message = obj['message']

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        # Placeholder for user-defined implementation
        pass

@dataclass
class ExperimentResult:
    input_data: Any
    output_data: Any
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(\n"
            f"  input_data={self.input_data},\n"
            f"  output_data={self.output_data},\n"
            f"  success={self.success},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Any], None]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]):
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    def publish(self, event_type: str, data: Any):
        for handler in self._subscribers.get(event_type, []):
            handler(data)







# ---------------------------------------------------------------
# TODO: refactor from here down
@dataclass
class ExperimentAgent(Atom):
    theory_name: str
    ttl: int
    experiment: Callable[[Any], ExperimentResult]
    termination_condition: Callable[[ExperimentResult], bool]
    initial_input: Any
    experiment_log: List[ExperimentResult] = field(default_factory=list)
    retries: int = 3
    retry_delay: float = 1.0
    max_parallel: int = 1

    async def run(self) -> Optional[ExperimentResult]:
        current_input = self.initial_input
        for iteration in range(self.ttl):
            try:
                tasks = [asyncio.create_task(self._run_experiment(current_input))
                         for _ in range(min(self.retries, self.max_parallel))]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                success_result = next((r for r in results if isinstance(r, ExperimentResult) and r.success), None)

                if success_result:
                    if self.termination_condition(success_result):
                        return success_result
                    current_input = success_result.output_data
            except Exception as e:
                logging.error(f"{self.theory_name} - Unexpected error in run method: {e}")

        return None

    async def _run_experiment(self, input_data: Any) -> Optional[ExperimentResult]:
        for attempt in range(self.retries):
            try:
                result = self.experiment(input_data)
                self.experiment_log.append(result)
                return result
            except Exception as e:
                logging.error(f"Experiment failed on attempt {attempt + 1} with error: {e}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(self.retry_delay)
        return None

    def get_experiment_log(self) -> List[ExperimentResult]:
        return self.experiment_log

    def encode(self) -> bytes:
        raise NotImplementedError("ExperimentAgent cannot be directly encoded")

    def decode(self, data: bytes) -> None:
        raise NotImplementedError("ExperimentAgent cannot be directly decoded")

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return asyncio.run(self.run())

    def __repr__(self) -> str:
        total_experiments = len(self.experiment_log)
        success_experiments = len([r for r in self.experiment_log if r.success])
        failed_experiments = total_experiments - success_experiments
        success_rate = (success_experiments / total_experiments) * 100 if total_experiments > 0 else 0

        detailed_results = "\n".join([repr(result) for result in self.experiment_log])
        return (
            f"ExperimentAgent(\n"
            f"  theory_name={self.theory_name},\n"
            f"  ttl={self.ttl},\n"
            f"  retries={self.retries},\n"
            f"  retry_delay={self.retry_delay},\n"
            f"  max_parallel={self.max_parallel},\n"
            f"  total_experiments={total_experiments},\n"
            f"  successful_experiments={success_experiments},\n"
            f"  failed_experiments={failed_experiments},\n"
            f"  success_rate={success_rate:.2f}%,\n"
            f"  detailed_results=[\n{detailed_results}\n"
            f"  ]\n"
            f")"
        )

@dataclass
class Theory:
    name: str
    hypothesis: Callable[[Any], bool]
    experiment: Callable[[Any], ExperimentResult]

    def test(self, input_data: Any) -> ExperimentResult:
        result = self.experiment(input_data)
        result.metadata['hypothesis_result'] = self.hypothesis(result.output_data)
        return result

@dataclass
class AntiTheory:
    theory: Theory

    def test(self, input_data: Any) -> ExperimentResult:
        result = self.theory.test(input_data)
        result.success = not result.success
        result.metadata['anti_hypothesis_result'] = not result.metadata['hypothesis_result']
        return result


class AtomicBotApplication:
    def __init__(self):
        self.event_bus = EventBus()
        self.theories: List[Theory] = []
        self.experiment_agents: List[ExperimentAgent] = []

    def register_theory(self, theory: Theory):
        self.theories.append(theory)

    def create_experiment_agent(self, theory: Theory, initial_input: Any, ttl: int, termination_condition: Callable[[ExperimentResult], bool]):
        agent = ExperimentAgent(
            theory_name=theory.name,
            ttl=ttl,
            experiment=theory.test,
            termination_condition=termination_condition,  # User must provide
            initial_input=initial_input
        )
        self.experiment_agents.append(agent)

    async def run_experiments(self):
        tasks = [agent.run() for agent in self.experiment_agents]
        results = await asyncio.gather(*tasks)
        return results

    def process_event(self, event: Event):
        event.execute()

    def handle_action_request_async(self, request: ActionRequest) -> ActionResponse:
        return request.execute()

    def __repr__(self) -> str:
        total_agents = len(self.experiment_agents)
        agents_results = "\n".join([repr(agent) for agent in self.experiment_agents])
        return (
            f"AtomicBotApplication(\n"
            f"  total_agents={total_agents},\n"
            f"  agents_results=[\n{agents_results}\n"
            f"  ]\n"
            f")"
        )

# -------------------------------------------------------
# end of library code




# Library users are expected to implement their own logic using the provided data structures and classes.
# Below is an example:

def sample_experiment(input_data: Any) -> ExperimentResult:
    """
    Example implementation of an experiment function.
    This function doubles the input data and returns an ExperimentResult.
    """
    try:
        output_data = input_data * 2  # Simple operation for demonstration
        success = True
        metadata = {'extra_info': 'This is a doubling operation'}
    except Exception as e:
        output_data = None
        success = False
        metadata = {'error': str(e)}
    return ExperimentResult(
        input_data=input_data,
        output_data=output_data,
        success=success,
        metadata=metadata
    )

def sample_hypothesis(output_data: Any) -> bool:
    """
    Example hypothesis function.
    This function checks if the output data is a positive integer.
    """
    return isinstance(output_data, int) and output_data > 0

def sample_termination_condition(result: ExperimentResult) -> bool:
    """
    Example termination condition function.
    This function stops the experiment if the output data is greater than 20.
    """
    return result.success and result.output_data > 20

async def main():
    """
    The main function demonstrating usage of the library.
    """
    # Track the start time
    start_time = time.time()

    # Create the application instance
    application = AtomicBotApplication()

    # Step 1: Create and register a theory
    theory_name = "SampleTheory"
    theory = Theory(name=theory_name, hypothesis=sample_hypothesis, experiment=sample_experiment)
    application.register_theory(theory)

    # Step 2: Create an experiment agent for the theory
    initial_input = 10  # Example initial input
    ttl = 5  # Time to live (number of iterations) for experiments
    application.create_experiment_agent(
        theory=theory,
        initial_input=initial_input,
        ttl=ttl,
        termination_condition=sample_termination_condition
    )

    results = await application.run_experiments()
    end_time = time.time()
    total_duration = end_time - start_time

    # Step 4: Compile detailed report
    total_experiments = len(results)
    successful_experiments = len([r for r in results if r and r.success])
    failed_experiments = total_experiments - successful_experiments
    success_rate = (successful_experiments / total_experiments) * 100 if total_experiments > 0 else 0

    print(f"\nExperiment Report for {theory_name}:")
    print(f"===================================")
    print(f"Total Experiments Run: {total_experiments}")
    print(f"Successful Experiments: {successful_experiments} ({success_rate:.2f}%)")
    print(f"Failed Experiments: {failed_experiments}")
    print(f"Total Duration: {total_duration:.2f} seconds\n")

    print("Detailed Results:")
    print("=================")
    for idx, result in enumerate(results, start=1):
        if not result:
            print(f"Experiment {idx}: Failed")
        else:
            print(f"Experiment {idx}:")
            print(f"  Input Data: {result.input_data}")
            print(f"  Output Data: {result.output_data}")
            print(f"  Success: {result.success}")
            print(f"  Hypothesis Result: {result.metadata.get('hypothesis_result', 'N/A')}")
            print(f"  Metadata: {result.metadata}")

if __name__ == '__main__':
    asyncio.run(main())