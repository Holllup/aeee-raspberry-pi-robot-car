import pathlib
import sys

p = pathlib.Path("line_following_v1_9.py")
txt = p.read_text(encoding="utf-8")

# 1. Imports
if "import football_manager" not in txt:
    txt = txt.replace("import detour_manager\n", "import detour_manager\nimport football_manager\n")

# 2. Init mgr
target_mgr = """    detour_mgr = detour_manager.DetourManager(control_state, str(Path(__file__).parent / 'detour_script.json'))
    threading.Thread(target=detour_manager.udp_listener, args=(detour_mgr,), daemon=True).start()"""
new_mgr = target_mgr + """
    football_mgr = football_manager.FootballManager(control_state, str(Path(__file__).parent / 'football_script.json'))
    threading.Thread(target=football_manager.udp_listener, args=(football_mgr,), daemon=True).start()"""
if "football_mgr =" not in txt:
    txt = txt.replace(target_mgr, new_mgr)

# 3. State vars
target_state = """    last_obstacle_detect_time = -999.0"""
new_state = target_state + """
    football_detour_active = False
    football_detour_phase_name = ""
    football_line_reacquire_hits = 0
    last_football_detect_time = -999.0"""
if "football_detour_active =" not in txt:
    txt = txt.replace(target_state, new_state)

# 4. activate function
target_activate = """        control_state.clear_manual_drive()
        control_state.set_pause(True)"""

new_activate = """        control_state.clear_manual_drive()
        control_state.set_pause(True)

    def activate_football_detour(trigger_now):
        nonlocal center_purple_sequence_active
        nonlocal center_purple_sequence_until
        nonlocal football_detour_active
        nonlocal football_detour_phase_name
        center_purple_sequence_active = False
        center_purple_sequence_until = -999.0
        servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.2)
        
        football_detour_active = True
        if (Path(__file__).parent / "football_script.json").exists():
             if football_mgr.start_playback(trigger_now):
                 football_detour_phase_name = "playback"
             else:
                 football_mgr.start_record(trigger_now)
                 football_detour_phase_name = "recording"
        else:
             football_mgr.start_record(trigger_now)
             football_detour_phase_name = "recording"
             
        control_state.clear_manual_drive()
        control_state.set_pause(True)"""
if "def activate_football_detour" not in txt:
    txt = txt.replace(target_activate, new_activate, 1)

# 5. center sequence trigger
target_trigger = """                elif sign_result is not None and sign_result.get("label") == "OBSTACLE":
                    activate_obstacle_detour(now)
                    last_obstacle_detect_time = now"""
new_trigger = target_trigger + """
                elif sign_result is not None and sign_result.get("label") == "FOOTBALL":
                    activate_football_detour(now)
                    last_football_detect_time = now"""
if "label\") == \"FOOTBALL\":" not in txt:
    txt = txt.replace(target_trigger, new_trigger)

# 6. Fallback scan trigger exclusions
target_fallback_ex = """                and not obstacle_detour_active
                and now >= sorting_post_cooldown_until"""
new_fallback_ex = """                and not obstacle_detour_active
                and not football_detour_active
                and now >= sorting_post_cooldown_until"""
if "and not football_detour_active" not in txt:
    txt = txt.replace(target_fallback_ex, new_fallback_ex)

# 7. Fallback scan trigger
target_fallback_trig = """                    if scan_sign_result.get("label") == "OBSTACLE":
                        sign_roi = scan_sign_roi
                        sign_result = scan_sign_result
                        activate_obstacle_detour(now)
                        last_obstacle_detect_time = now"""
new_fallback_trig = target_fallback_trig + """
                    elif scan_sign_result.get("label") == "FOOTBALL":
                        sign_roi = scan_sign_roi
                        sign_result = scan_sign_result
                        activate_football_detour(now)
                        last_football_detect_time = now"""
if "elif scan_sign_result.get(\"label\") == \"FOOTBALL\":" not in txt:
    txt = txt.replace(target_fallback_trig, new_fallback_trig)

# 8. Detour Processing block
target_process = """            if obstacle_detour_active:
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

new_process = target_process + """

            if football_detour_active:
                if football_detour_phase_name == "recording":
                    if not football_mgr.recording:
                        football_detour_active = False
                        traffic_release_ignore_until = max(traffic_release_ignore_until, now + OBSTACLE_RELEASE_IGNORE_SECONDS)
                        control_state.clear_manual_drive()
                        control_state.set_pause(False)
                elif football_detour_phase_name == "playback":
                    playing, left, right = football_mgr.update_playback(now)
                    if playing:
                        control_state.set_manual_drive(left, right, now + 0.2, mode="football_playback")
                    else:
                        football_detour_phase_name = "seek_line"
                elif football_detour_phase_name == "seek_line":
                    (seek_center, _, _, _, _), _ = find_line_center(frame)
                    if seek_center is not None:
                        football_line_reacquire_hits += 1
                    else:
                        football_line_reacquire_hits = 0

                    control_state.set_manual_drive(OBSTACLE_SEEK_LINE_SPEED, OBSTACLE_SEEK_LINE_SPEED, now + 0.20, mode="football_seek_line")
                    if football_line_reacquire_hits >= OBSTACLE_LINE_REACQUIRE_HITS:
                        football_detour_active = False
                        football_detour_phase_name = "done"
                        football_line_reacquire_hits = 0
                        control_state.clear_manual_drive()
                        traffic_release_ignore_until = max(traffic_release_ignore_until, now + OBSTACLE_RELEASE_IGNORE_SECONDS)
                        control_state.set_pause(False)"""
if "if football_detour_active:" not in txt:
    txt = txt.replace(target_process, new_process)

# 9. pause active conditions
target_pause = """                or obstacle_detour_active
                or (center_sort_counts is not None and now < sorting_display_until)
                or detour_mgr.recording"""
new_pause = """                or obstacle_detour_active
                or football_detour_active
                or (center_sort_counts is not None and now < sorting_display_until)
                or detour_mgr.recording
                or football_mgr.recording"""
if "or football_detour_active" not in txt:
    txt = txt.replace(target_pause, new_pause)

# 10. overlay lcd info
target_lcd = """            elif obstacle_detour_active:
                alarm_led.all_off()
                lcd_display.update("obstcal", obstacle_detour_phase_name[:16])"""
new_lcd = target_lcd + """
            elif football_detour_active:
                alarm_led.all_off()
                lcd_display.update("football", football_detour_phase_name[:16])"""
if "elif football_detour_active:" not in txt:
    txt = txt.replace(target_lcd, new_lcd)

# 11. mode label for overlay
target_mode = """            if obstacle_detour_active:
                mode = f"obstacle_{obstacle_detour_phase_name}"
            draw_overlay("""
new_mode = """            if obstacle_detour_active:
                mode = f"obstacle_{obstacle_detour_phase_name}"
            if football_detour_active:
                mode = f"football_{football_detour_phase_name}"
            draw_overlay("""
if "if football_detour_active:\n                mode = f\"football_" not in txt:
    txt = txt.replace(target_mode, new_mode)


p.write_text(txt, encoding="utf-8")
print("Huzzah! Patch applied!")
