from src.kernelblaster.baseline_jobs.search import BackendConvergence, BackendScheduler


def test_scheduler_is_seventy_thirty_while_both_backends_are_active():
    scheduler = BackendScheduler()
    choices = [scheduler.next_backend() for _ in range(10)]
    assert choices.count("cuda") == 7
    assert choices.count("triton") == 3


def test_backends_converge_independently_and_survivor_continues():
    scheduler = BackendScheduler()
    for _ in range(50):
        scheduler.states["triton"].record_unrankable()
    assert not scheduler.states["triton"].active
    assert scheduler.states["cuda"].active
    assert {scheduler.next_backend() for _ in range(20)} == {"cuda"}


def test_rankable_stagnation_and_recoverable_blocked_state_are_distinct():
    state = BackendConvergence("cuda")
    state.record_rankable(100)
    for _ in range(23):
        state.record_rankable(100.9)
    assert state.active
    state.record_blocked()
    assert state.active and state.blocked_total == 1
    state.record_rankable(100.9)
    assert not state.active and state.reason == "rankable_converged"


def test_events_unavailable_terminates_gpu_search():
    scheduler = BackendScheduler()
    scheduler.stop_for_events_unavailable()
    assert scheduler.complete and scheduler.next_backend() is None
    assert {item.reason for item in scheduler.states.values()} == {"events_unavailable"}
