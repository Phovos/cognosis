import json
import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Generic, TypeVar
import threading
import queue
import time
import asyncio
from atoms import *

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
    def validate(self) -> bool:
        return super().validate()

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
