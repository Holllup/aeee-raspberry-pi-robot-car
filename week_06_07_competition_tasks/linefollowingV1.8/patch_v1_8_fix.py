import sys
from pathlib import Path

FILE_PATH = Path("line_following_v1_8_teach_and_repeat.py")
content = FILE_PATH.read_text(encoding="utf-8")

target_str = """
            pause_active = (
                center_purple_sequence_active
                or traffic_wait_active
                or (now < music_display_until)
                or music_command_pending
                or obstacle_detour_active
                or (center_sort_counts is not None and now < sorting_display_until)
            )
            control_state.set_pause(pause_active)
""".strip("\n")

replacement_str = """
            pause_active = (
                center_purple_sequence_active
                or traffic_wait_active
                or (now < music_display_until)
                or music_command_pending
                or obstacle_detour_active
                or (center_sort_counts is not None and now < sorting_display_until)
                or detour_mgr.recording
            )
            control_state.set_pause(pause_active)
""".strip("\n")

if target_str in content:
    content = content.replace(target_str, replacement_str)
    FILE_PATH.write_text(content, encoding="utf-8")
    print("PATCH_SUCCESS")
else:
    print("PATCH_FAIL: target string not found in file")
    sys.exit(1)
