import re
from pathlib import Path

FILE_PATH = Path("line_following_v1_8_teach_and_repeat.py")
content = FILE_PATH.read_text(encoding="utf-8")

# 1. Add imports
content = re.sub(
    r"(import time\nfrom pathlib import Path)",
    r"\1\nimport detour_manager\nimport threading",
    content
)

# 2. Add detour manager initialization in main() right after control_state initialization
content = re.sub(
    r"(control_state = ControlState\(motor_armed=args\.arm\))",
    r"\1\n    detour_mgr = detour_manager.DetourManager(control_state, str(Path(__file__).parent / 'detour_script.json'))\n    threading.Thread(target=detour_manager.udp_listener, args=(detour_mgr,), daemon=True).start()",
    content
)

# 3. Rewrite activate_obstacle_detour
old_activate_obstacle_detour = """
    def activate_obstacle_detour(trigger_now):
        nonlocal center_purple_sequence_active
        nonlocal center_purple_sequence_until
        nonlocal obstacle_detour_active
        nonlocal obstacle_detour_phase_index
        nonlocal obstacle_detour_phase_until
        nonlocal obstacle_detour_phase_name
        nonlocal obstacle_detour_until
        nonlocal obstacle_line_reacquire_hits
        center_purple_sequence_active = False
        center_purple_sequence_until = -999.0
        servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.2)
        obstacle_detour_active = True
        obstacle_detour_phase_index = -1
        obstacle_detour_phase_until = trigger_now
        obstacle_detour_phase_name = "init"
        obstacle_line_reacquire_hits = 0
        obstacle_detour_until = trigger_now + max(
            OBSTACLE_DETOUR_SECONDS,
            (
                OBSTACLE_RIGHT_TURN_SECONDS
                + OBSTACLE_FORWARD_1_SECONDS
                + OBSTACLE_LEFT_TURN_SECONDS
                + 8.0
            ),
        )
        control_state.clear_manual_drive()
        control_state.set_pause(True)
"""

new_activate_obstacle_detour = """
    def activate_obstacle_detour(trigger_now):
        nonlocal center_purple_sequence_active
        nonlocal center_purple_sequence_until
        nonlocal obstacle_detour_active
        nonlocal obstacle_detour_phase_name
        center_purple_sequence_active = False
        center_purple_sequence_until = -999.0
        servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.2)
        
        obstacle_detour_active = True
        if (Path(__file__).parent / "detour_script.json").exists():
             if detour_mgr.start_playback(trigger_now):
                 obstacle_detour_phase_name = "playback"
             else:
                 detour_mgr.start_record(trigger_now)
                 obstacle_detour_phase_name = "recording"
        else:
             detour_mgr.start_record(trigger_now)
             obstacle_detour_phase_name = "recording"
             
        control_state.clear_manual_drive()
        control_state.set_pause(True)
"""

content = content.replace(old_activate_obstacle_detour.strip("\n"), new_activate_obstacle_detour.strip("\n"))

# 4. Rewrite the obstacle_detour_active handling in motor_control_loop / main loop
# The existing logic relies on phases. We'll replace it entirely.
old_obstacle_loop_regex = re.compile(
    r"if obstacle_detour_active:.*?elif now >= obstacle_detour_until:.*?control_state\.set_pause\(False\)",
    re.DOTALL
)

new_obstacle_loop = """if obstacle_detour_active:
                if obstacle_detour_phase_name == "recording":
                    if not detour_mgr.recording:
                        obstacle_detour_active = False
                        traffic_release_ignore_until = max(traffic_release_ignore_until, now + OBSTACLE_RELEASE_IGNORE_SECONDS)
                        control_state.clear_manual_drive()
                        control_state.set_pause(False)
                elif obstacle_detour_phase_name == "playback":
                    playing, left, right = detour_mgr.update_playback(now)
                    if playing:
                        control_state.set_manual_drive(left, right, now + 0.2, mode="playback")
                    else:
                        obstacle_detour_phase_name = "seek_line"
                elif obstacle_detour_phase_name == "seek_line":
                    (seek_center, _, _, _, _), _ = find_line_center(frame)
                    if seek_center is not None:
                        obstacle_line_reacquire_hits += 1
                    else:
                        obstacle_line_reacquire_hits = 0

                    control_state.set_manual_drive(OBSTACLE_SEEK_LINE_SPEED, OBSTACLE_SEEK_LINE_SPEED, now + 0.20, mode="obstacle_seek_line")
                    if obstacle_line_reacquire_hits >= OBSTACLE_LINE_REACQUIRE_HITS:
                        obstacle_detour_active = False
                        obstacle_detour_phase_name = "done"
                        obstacle_line_reacquire_hits = 0
                        control_state.clear_manual_drive()
                        traffic_release_ignore_until = max(traffic_release_ignore_until, now + OBSTACLE_RELEASE_IGNORE_SECONDS)
                        control_state.set_pause(False)"""

content = old_obstacle_loop_regex.sub(new_obstacle_loop, content)

FILE_PATH.write_text(content, encoding="utf-8")
print("Patched successfully.")
