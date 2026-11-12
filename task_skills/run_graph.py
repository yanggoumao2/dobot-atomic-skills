"""Input-source-independent skill DAG execution; never invokes the NL parser."""

import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .atomic_graph import AtomicNode
from .gateway.contracts import canonical
from .journal import ExecutionJournal


class MockAdapter:
    def __init__(self, cancel, step_seconds=0.001):
        self.cancel = cancel
        self.step_seconds = step_seconds
        self.calls = []
        self.lock = threading.Lock()

    def execute_atomic(self, node, scene):
        if self.cancel.is_set():
            raise InterruptedError("Execution cancelled")
        with self.lock:
            self.calls.append((node.arm, node.operation))
        seconds = node.parameters["seconds"] if node.operation == "wait" else self.step_seconds
        if self.cancel.wait(seconds):
            raise InterruptedError("Execution cancelled")
        return {"mock": True, "operation": node.operation, "hardware_connected": False}

    def snapshot_all(self):
        return {"hardware_connected": False, "mode": "mock"}

    def stop_and_verify(self):
        return {"confirmed": True, "hardware_connected": False}

    def close(self):
        pass


def run_graph(plan, adapter, scene, directory, cancel, emit, live=False):
    """Run external skill nodes with AND-success dependencies and resource exclusion.

    `emit` persists externally correlated events. The caller owns device leases
    and terminal status publication, including unconfirmed stop quarantine.
    """
    directory = Path(directory)
    journal = ExecutionJournal(directory)
    (directory / "submitted_package.json").write_text(canonical(plan.package), encoding="utf-8")
    completed, futures, active_arms = set(), {}, set()
    first_error = None
    stop_result = None
    preflight_evidence = None
    family = getattr(plan, "family", "embodied")
    nodes = {n["node_id"]: n for n in plan.package["nodes"]}

    def execute_skill(node_id):
        node = nodes[node_id]
        deadline = time.monotonic() + node.get("timeout_s", 180)

        def execute_steps():
            evidence = []
            for operation in plan.operations[node_id]:
                if cancel.is_set():
                    raise InterruptedError("Execution cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Skill exceeded its total timeout")
                operation.timeout_seconds = remaining
                journal.record("atomic_started", upstream_node_id=node_id, node=operation)
                if operation.operation == "wait":
                    seconds = operation.parameters["seconds"]
                    if cancel.wait(min(seconds, remaining)):
                        raise InterruptedError("Execution cancelled")
                    if seconds > remaining:
                        raise TimeoutError("Wait exceeded total timeout")
                    result = {"elapsed_seconds": seconds}
                else:
                    result = adapter.execute_atomic(operation, scene)
                if time.monotonic() > deadline:
                    raise TimeoutError("Skill exceeded its total timeout")
                journal.record("atomic_completed", upstream_node_id=node_id, node=operation, evidence=result)
                evidence.append(result)
            return {"internal_action_count": len(evidence), "feedback": evidence,
                    "verification": {"level": "command_trace", "physical_success_verified": False},
                    "hardware_connected": live}

        return execute_steps()

    def request_stop(force=False):
        nonlocal stop_result
        if stop_result is None or force:
            try:
                stop_result = adapter.stop_and_verify()
            except Exception as error:
                stop_result = {"confirmed": False, "error": type(error).__name__}

    try:
        if live and family == "embodied":
            from .dobot_adapter import DobotHardwareAdapter
            from .layout import physical_nova_home_degrees

            snapshots = adapter.snapshot_all()
            for arm in ("left", "right"):
                snapshot = snapshots[arm]
                if snapshot["mode"] != 5 or DobotHardwareAdapter._has_alarm(snapshot["errors"]):
                    raise RuntimeError("Controller must already be enabled, idle and alarm-free")
                measured = DobotHardwareAdapter._numeric_payload(snapshot["joints"])
                if len(measured) != 6 or not DobotHardwareAdapter._target_matches(measured, physical_nova_home_degrees(arm), 0):
                    raise RuntimeError("Both arms must start at their calibrated home pose")
                adapter.configure_robot(arm, 20)
        elif live:
            # The dobot.* atoms carry their own session and their own start-state
            # rules, so the gate is the chemistry driver's read-only preflight of
            # the whole package rather than the embodied family's home check.
            preflight_evidence = adapter.preflight(plan)
            if preflight_evidence.get("success") is not True:
                raise RuntimeError("Chemistry preflight failed: %s"
                                   % (preflight_evidence.get("error")
                                      or preflight_evidence.get("note") or "unknown reason"))
        emit("execution_started", status="running")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="dobot-skill") as pool:
            while len(completed) < len(plan.order):
                if cancel.is_set():
                    request_stop()
                if first_error is None and not cancel.is_set():
                    for node_id in plan.order:
                        if len(futures) >= 2:
                            break
                        if node_id in completed or node_id in futures.values() or not set(plan.dependencies[node_id]).issubset(completed):
                            continue
                        arm = plan.arms[node_id]
                        if arm != "system" and arm in active_arms:
                            continue
                        emit("action_started" if nodes[node_id]["node_type"] == "action" else "wait_started",
                             node_id=node_id, node_status="running")
                        active_arms.add(arm)
                        futures[pool.submit(execute_skill, node_id)] = node_id
                if not futures:
                    break
                ready, _ = wait(tuple(futures), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in ready:
                    node_id = futures.pop(future)
                    active_arms.discard(plan.arms[node_id])
                    try:
                        evidence = future.result()
                    except Exception as error:
                        is_cancel = isinstance(error, InterruptedError)
                        code = "EXECUTION_CANCELLED" if is_cancel else "MOTION_TIMEOUT" if isinstance(error, TimeoutError) else "SKILL_FAILED"
                        if not is_cancel and first_error is None:
                            first_error = {"code": code, "retryable": False, "node_id": node_id}
                        cancel.set()
                        request_stop()
                        node_status = "interrupted" if is_cancel and not stop_result.get("confirmed") else "cancelled" if is_cancel else "failed"
                        emit("action_" + node_status, node_id=node_id, node_status=node_status,
                             error={"code": code, "retryable": False},
                             payload={"exception_type": type(error).__name__,
                                      "message": str(error), "traceback": traceback.format_exc()})
                    else:
                        completed.add(node_id)
                        emit("action_completed" if nodes[node_id]["node_type"] == "action" else "wait_completed",
                             node_id=node_id, node_status="succeeded", payload=evidence)
        if cancel.is_set() or first_error:
            # All skill workers have exited; confirm the final rig state again.
            request_stop(force=True)
            return {"status": "failed" if first_error else "cancelled", "error": first_error,
                    "stop": stop_result, "verification": {"level": "command_trace", "physical_success_verified": False}}
        if len(completed) != len(plan.order):
            raise RuntimeError("DAG scheduler made no progress")
        if live:
            emit("execution_finalizing", status="finalizing")
            arms = sorted(set(plan.arms.values()) - {"system"})
            home_futures, submitted_arms = {}, set()
            home_failure, interrupted_arm = None, None
            finalization_cancelled = False
            current_arm = None

            def execute_final_home(arm):
                if cancel.is_set():
                    raise InterruptedError("Execution cancelled before final homing")
                node = AtomicNode("final:" + arm, 0, "move_safe_pose", arm, timeout_seconds=180)
                return adapter.execute_atomic(node, scene)

            if arms:
                with ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="dobot-home") as pool:
                    try:
                        for arm in arms:
                            current_arm = arm
                            if cancel.is_set():
                                finalization_cancelled = True
                                if interrupted_arm is None:
                                    interrupted_arm = arm
                                request_stop()
                                break
                            emit("arm_finalizing", payload={"arm": arm, "status": "finalizing"})
                            home_futures[pool.submit(execute_final_home, arm)] = arm
                            submitted_arms.add(arm)
                        for arm in arms:
                            if arm not in submitted_arms:
                                current_arm = arm
                                emit("arm_finalization_cancelled",
                                     payload={"arm": arm, "status": "cancelled"})

                        while home_futures:
                            if cancel.is_set():
                                finalization_cancelled = True
                                if interrupted_arm is None and home_futures:
                                    interrupted_arm = next(iter(home_futures.values()))
                                request_stop()
                            ready, _ = wait(tuple(home_futures), timeout=0.05, return_when=FIRST_COMPLETED)
                            for future in ready:
                                arm = home_futures.pop(future)
                                current_arm = arm
                                try:
                                    evidence = future.result()
                                except InterruptedError as error:
                                    finalization_cancelled = True
                                    if interrupted_arm is None:
                                        interrupted_arm = arm
                                    cancel.set()
                                    request_stop()
                                    emit("arm_finalization_cancelled",
                                         payload={"arm": arm, "status": "cancelled",
                                                  "exception_type": type(error).__name__},
                                         error={"code": "EXECUTION_CANCELLED", "retryable": False})
                                except Exception as error:
                                    code = "MOTION_TIMEOUT" if isinstance(error, TimeoutError) else "FINAL_HOME_FAILED"
                                    if home_failure is None:
                                        home_failure = {"code": code, "retryable": False, "phase": "final_home",
                                                        "arm": arm, "exception_type": type(error).__name__}
                                    cancel.set()
                                    request_stop()
                                    emit("arm_finalization_failed",
                                         payload={"arm": arm, "status": "failed",
                                                  "exception_type": type(error).__name__},
                                         error={"code": code, "retryable": False})
                                else:
                                    emit("arm_finalized", payload={"arm": arm, "status": "succeeded",
                                                                   "evidence": evidence})
                    except Exception as error:
                        # Event persistence or submission can fail while another home is active.
                        # Stop before the executor context waits for its workers to finish.
                        if home_failure is None:
                            home_failure = {"code": "FINALIZATION_EVENT_FAILED", "retryable": False,
                                            "phase": "final_home", "arm": current_arm,
                                            "exception_type": type(error).__name__}
                        cancel.set()
                        finalization_cancelled = True
                        request_stop()
                        while home_futures:
                            ready, _ = wait(tuple(home_futures), timeout=0.05, return_when=FIRST_COMPLETED)
                            for future in ready:
                                arm = home_futures.pop(future)
                                try:
                                    future.result()
                                except InterruptedError:
                                    if interrupted_arm is None:
                                        interrupted_arm = arm
                                except Exception:
                                    pass

            if cancel.is_set():
                finalization_cancelled = True
            if home_failure or finalization_cancelled:
                # Stop promptly above, then verify again after every home worker has exited.
                request_stop(force=True)
                error = home_failure or {"code": "EXECUTION_CANCELLED", "retryable": False,
                                         "phase": "final_home"}
                if not home_failure and interrupted_arm:
                    error["arm"] = interrupted_arm
                return {"status": "failed" if home_failure else "cancelled", "error": error,
                        "stop": stop_result,
                        "verification": {"level": "command_trace", "physical_success_verified": False}}
        feedback = adapter.snapshot_all()
        if live and cancel.is_set():
            request_stop()
            request_stop(force=True)
            return {"status": "cancelled",
                    "error": {"code": "EXECUTION_CANCELLED", "retryable": False, "phase": "final_home"},
                    "stop": stop_result,
                    "verification": {"level": "command_trace", "physical_success_verified": False}}
        return {"status": "succeeded", "feedback": feedback, "family": family,
                "preflight": preflight_evidence,
                "verification": {"level": "command_trace", "physical_success_verified": False},
                "hardware_connected": live}
    except Exception as error:
        cancel.set()
        request_stop()
        interrupted = isinstance(error, InterruptedError)
        return {"status": "cancelled" if interrupted else "failed",
                "error": {"code": "EXECUTION_CANCELLED" if interrupted else "MOTION_TIMEOUT" if isinstance(error, TimeoutError) else "EXECUTION_FAILED",
                          "retryable": False, "message": str(error)},
                "stop": stop_result, "exception_type": type(error).__name__,
                "traceback": traceback.format_exc()}
