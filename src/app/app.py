from src.app.theory import *
from src.app.llama import *
from src.app.kernel import *
from src.app.model import *
from src.app.atoms import *

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

if __name__ == "__main__":
    asyncio.run(main())
    # Create a Token
    token = Token("example")
    result = token.execute()
    logging.info(f"Token result: {result}")

    # Create a FormalTheory
    formal_theory = FormalTheory()
    formal_theory.top_atom = Token("top")
    formal_theory.bottom_atom = Token("bottom")
    result = formal_theory.execute()
    logging.info(f"FormalTheory result: {result}")

    # Publish and Subscribe with the EventBus
    event_bus.subscribe("example_event", lambda e: logging.info(f"Received event: {e.to_dict()}"))

    event = Event(id="1", type="example_event", detail_type="test", message=[{"key": "value"}])
    event_bus.publish("example_event", event)
