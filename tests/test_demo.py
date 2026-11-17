"""Demo captions and verdicts come from the run's trace (no simulator needed)."""

from adas_validation.demo import events, project, trim, verdict


def _row(t, ego_x, hz_x, hz_y=3.5, mode="follow", tracks=(), hz_vis=1.0, accel=0.0):
    return dict(t=t, ego_x=ego_x, ego_y=3.5, ego_v=36.0, hz_x=hz_x, hz_y=hz_y, hz_vx=30.0,
                hz_vis=hz_vis, mode=mode, accel_cmd=accel, dets=[], tracks=list(tracks))


def _lead_run(known: bool, collided: bool) -> dict:
    trace = [_row(0.0, 0, 40), _row(2.5, 80, 110, accel=-3.0), _row(3.0, 95, 112, mode="aeb")]
    return {"meta": {"kind": "lead_brake", "ego_speed_kph": 130, "system":
                     {"known_lead_accel": known} if known else {},
                     "params": {"initial_gap_m": 30, "lead_decel_mps2": 6.0, "brake_at_s": 2.0},
                     "result": {"collided": collided, "impact_speed_kph": 37.0,
                                "end_reason": "collision" if collided else "stopped"}},
            "trace": trace}


def test_lead_brake_events_explain_the_late_aeb():
    keys = [k for _, k, _ in events(_lead_run(known=False, collided=True))]
    assert keys == ["start", "brake", "acc", "aeb", "end"]
    ev = {k: c for _, k, c in events(_lead_run(known=False, collided=True))}
    assert "assumed the lead keeps its speed" in ev["aeb"] and "37 km/h" in ev["end"]
    ev = {k: c for _, k, c in events(_lead_run(known=True, collided=False))}
    assert "knows the lead is braking" in ev["aeb"] and ev["end"].startswith("Stopped")
    assert verdict(_lead_run(known=False, collided=True))[0].startswith("COLLISION")


def test_trim_ends_a_passed_run_after_the_pedestrian_leaves_the_path():
    trace = [_row(0.1 * k, k, 50, hz_y=3.5 - 0.2 * max(k - 20, 0),
                  mode="aeb" if k == 20 else "cruise") for k in range(80)]
    run = {"meta": {"result": {"end_reason": "passed"}}, "trace": trace}
    n = len(trim(run)["trace"])
    assert 30 < n < 80                     # |dy| > 1.78 m after ~9 frames, plus 10


def test_projection_puts_left_objects_left_and_far_objects_higher():
    near = project(10.0, 0.0, "car", 60.0)
    far = project(40.0, 0.0, "car", 60.0)
    left = project(10.0, 3.5, "car", 60.0)
    assert far[3] < near[3] and (far[2] - far[0]) < (near[2] - near[0])
    assert left[2] < near[0] + 1e-9 or left[0] < near[0]
    assert project(-1.0, 0.0, "car", 60.0) is None
