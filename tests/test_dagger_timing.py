"""CPU checks of the production DAgger timing blocks; no simulator or process group.

Execute selected AST nodes to avoid the trainer's eager native simulator imports.
The optimizer uses real CPU tensors; only the clock and barrier are stand-ins.
"""
import ast
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "track_mj/learning/policy/dagger/dagger_horizon.py"
TREE = ast.parse(SOURCE.read_text())
DAGGER = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "dagger")
LOOP = next(node for node in DAGGER.body if isinstance(node, ast.For) and
            isinstance(node.target, ast.Name) and node.target.id == "step_id_train")
SECTIONS = [node for node in LOOP.body if isinstance(node, ast.With) and
            isinstance(node.items[0].context_expr, ast.Call) and
            node.items[0].context_expr.args[0].value.startswith("2.")]


def execute(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)


class Clock:
    now = 0.0

    def perf_counter(self):
        return self.now


class Policy:
    def __init__(self, clock, events, ddp=False):
        self.clock, self.events = clock, events
        self.network = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.network.weight.copy_(torch.tensor([[.25, -.5]]))
            self.network.bias.fill_(.1)
        self.model = SimpleNamespace(module=self.network) if ddp else self.network

    def compute_loss(self, batch, target, aux):
        self.events.append("loss")
        self.clock.now += 5
        return (self.network(batch["state"]) - target).square().mean(), {"aux": aux}


class Optimizer(torch.optim.AdamW):
    def __init__(self, model, clock, events):
        super().__init__(model.parameters(), lr=.001, weight_decay=.01)
        self.clock, self.events = clock, events

    def zero_grad(self, **kwargs):
        self.events.append("zero_grad")
        return super().zero_grad(**kwargs)

    def step(self, *args, **kwargs):
        self.events.append("optimizer")
        self.clock.now += 7
        return super().step(*args, **kwargs)


def scope_for(ddp):
    clock, events = Clock(), []
    clock.now = 0.0
    policy = Policy(clock, events, ddp)
    optimizer = Optimizer(policy.network, clock, events)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    original_step = scheduler.step

    def schedule():
        events.append("scheduler")
        clock.now += 11
        original_step()

    scheduler.step = schedule

    def prepare(**kwargs):
        events.append("prepare")
        clock.now += 2
        return {"example": True}

    def barrier(use_ddp):
        events.append("barrier" if use_ddp else "barrier_noop")
        if use_ddp:
            clock.now += 3

    timer_scope = dict(defaultdict=defaultdict, deque=deque, contextmanager=contextmanager,
                       time=clock, np=np)
    timer = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "PerformanceTimer")
    execute([timer], timer_scope)
    batch = torch.tensor([[1., 2.], [-.5, .25]])
    targets = torch.tensor([[.4], [-.1]])
    scope = dict(torch=torch, einops=SimpleNamespace(rearrange=lambda value, _: value.flatten(0, 1)),
                 perf_timer=timer_scope["PerformanceTimer"](), use_ddp=ddp,
                 nn_state_dict_th_list=[{"state": batch}, {"state": batch * .5}],
                 teacher_action_th_list=[targets, targets * .5], student_action_th_list=[targets, targets],
                 dones_th_list=[torch.zeros(2, dtype=torch.bool)] * 2,
                 aux_loss_info_handler=SimpleNamespace(get_aux_loss_info=prepare), auto_set_barrier=barrier,
                 student_nn=policy, optimizer=optimizer, scheduler=scheduler,
                 dagger_learning_epochs=3, max_grad_norm=.2)
    return scope, clock, events


class DaggerTimingTest(unittest.TestCase):
    def test_preparation_barrier_and_updates_keep_the_original_order_and_exact_weights(self):
        for ddp in (False, True):
            with self.subTest(ddp=ddp):
                scope, clock, events = scope_for(ddp)
                execute(SECTIONS, scope)
                expected_events = ["prepare", "barrier" if ddp else "barrier_noop"]
                expected_events += ["loss", "zero_grad", "optimizer"] * 3 + ["scheduler"]
                self.assertEqual(events, expected_events)

                # Independent reference: the original sequential updates, clipping each gradient.
                reference, _, _ = scope_for(ddp)
                inputs = torch.stack([entry["state"] for entry in reference["nn_state_dict_th_list"]], 1).flatten(0, 1)
                targets = torch.stack(reference["teacher_action_th_list"], 1).flatten(0, 1)
                policy, optimizer = reference["student_nn"], reference["optimizer"]
                for _ in range(3):
                    loss = (policy.network(inputs) - targets).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.network.parameters(), .2)
                    optimizer.step()
                reference["scheduler"].step()
                for actual, expected in zip(scope["student_nn"].network.parameters(), policy.network.parameters()):
                    self.assertTrue(torch.equal(actual, expected))
                actual_state, expected_state = scope["optimizer"].state_dict(), optimizer.state_dict()
                self.assertEqual(actual_state["param_groups"], expected_state["param_groups"])
                for param, state in actual_state["state"].items():
                    for field, value in state.items():
                        self.assertTrue(torch.equal(value, expected_state["state"][param][field]))

    def test_intervals_are_separate_and_account_for_preparation_barrier_and_update_costs(self):
        for ddp in (False, True):
            with self.subTest(ddp=ddp):
                scope, clock, _ = scope_for(ddp)
                execute(SECTIONS, scope)
                times = scope["perf_timer"].get_stats()
                expected = {"2.batch_preparation": 2, "2.rank_skew_wait": 3 if ddp else 0,
                            "2.optimizer_step": 3 * (5 + 7) + 11}
                self.assertEqual(set(times), set(expected))
                for label, duration in expected.items():
                    self.assertEqual(times[label]["count"], 1)
                    self.assertEqual(times[label]["mean"], duration)
                self.assertEqual(sum(value["mean"] for value in times.values()), clock.now)

    def test_reported_world_size_does_not_rescale_configured_optimizer_lr(self):
        log = next(node for node in DAGGER.body if isinstance(node, ast.Expr) and
                   isinstance(node.value, ast.Call) and node.value.args and
                   isinstance(node.value.args[0], ast.JoinedStr) and
                   node.value.args[0].values[0].value.startswith("[LR]"))
        optimizer = next(node for node in DAGGER.body if isinstance(node, ast.Assign) and
                         any(isinstance(target, ast.Name) and target.id == "optimizer" for target in node.targets))
        for world_size in (1, 2, 8):
            with self.subTest(world_size=world_size):
                messages = []
                lr = .0008
                scope = dict(logging=SimpleNamespace(info=messages.append), lr=lr, world_size=world_size,
                             num_envs_per_rank=10 // world_size, optim=torch.optim, weight_decay=.01,
                             student_nn=SimpleNamespace(model=torch.nn.Linear(2, 1)))
                execute([log, optimizer], scope)
                self.assertEqual(scope["optimizer"].param_groups[0]["lr"], lr)
                self.assertIn(f"lr={lr} is used as given (no world_size scaling)", messages[0])
                self.assertIn(f"world_size={world_size}", messages[0])
                self.assertIn(f"global_num_envs={(10 // world_size) * world_size}", messages[0])


if __name__ == "__main__":
    unittest.main()
