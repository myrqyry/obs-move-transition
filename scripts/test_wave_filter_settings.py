#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WAVE_FILTER = ROOT / "move-wave-value-filter.c"
EN_US_LOCALE = ROOT / "data" / "locale" / "en-US.ini"


def main():
    source = WAVE_FILTER.read_text()

    required_tokens = [
        "VALUE_ACTION_TRANSFORM",
        "VALUE_ACTION_SETTING",
        "VALUE_ACTION_SOURCE_VISIBILITY",
        "VALUE_ACTION_FILTER_ENABLE",
        "THRESHOLD_ENABLE_OVER_DISABLE_UNDER",
        "TRANSFORM_CROP_VERTICAL",
        'obs_properties_add_list(ppts, "value_action"',
        'obs_properties_add_list(ppts, "canvas"',
        'obs_properties_add_list(ppts, "scene"',
        'obs_properties_add_list(ppts, "sceneitem"',
        'obs_properties_add_list(ppts, "source"',
        'obs_properties_add_list(ppts, "filter"',
        'obs_properties_add_list(ppts, "transform"',
        'obs_properties_add_list(ppts, "setting"',
        'obs_properties_add_list(ppts, "threshold_action"',
        'obs_properties_add_float_slider(ppts, "threshold"',
        "wave_value_action_changed",
        "wave_value_canvas_changed",
        "wave_value_scene_changed",
        "wave_value_source_changed",
        "obs_sceneitem_set_visible",
        "obs_source_set_enabled",
        "obs_sceneitem_set_crop",
        "S_WAVE_PHASE",
        "S_WAVE_INVERT",
        "S_WAVE_CLAMP_MIN",
        "S_WAVE_CLAMP_MAX",
        "S_WAVE_CLAMP_ENABLED",
        "S_WAVE_PULSE_WIDTH",
        "S_WAVE_STEPS",
        "S_WAVE_RANDOM_AMOUNT",
        "WAVE_COSINE",
        "WAVE_PULSE",
        "WAVE_SAW_UP",
        "WAVE_SAW_DOWN",
        "WAVE_ABS_SINE",
        "WAVE_STEP",
        "WAVE_RANDOM_HOLD",
        "WAVE_SMOOTH_RANDOM",
        "wave_value_type_changed",
        'obs_properties_add_float_slider(ppts, S_WAVE_PHASE',
        'obs_properties_add_bool(ppts, S_WAVE_INVERT',
        'obs_properties_add_bool(ppts, S_WAVE_CLAMP_ENABLED',
        'obs_properties_add_float(ppts, S_WAVE_CLAMP_MIN',
        'obs_properties_add_float(ppts, S_WAVE_CLAMP_MAX',
        'obs_properties_add_float_slider(ppts, S_WAVE_PULSE_WIDTH',
        'obs_properties_add_int_slider(ppts, S_WAVE_STEPS',
        'obs_properties_add_float_slider(ppts, S_WAVE_RANDOM_AMOUNT',
    ]

    missing = [token for token in required_tokens if token not in source]
    if missing:
        print("move-wave-value-filter.c is missing audio move settings/actions:")
        for token in missing:
            print(f"  - {token}")
        return 1

    if "obs_source_release(source);\n\t}\n\telse {\n\t\tsource = parent;" in source:
        print("move-wave-value-filter.c releases target source before use in wave_value_tick")
        return 1

    forbidden_visible_defaults = [
        'obs_data_set_default_double(settings, S_WAVE_CLAMP_MIN, -DBL_MAX)',
        'obs_data_set_default_double(settings, S_WAVE_CLAMP_MAX, DBL_MAX)',
        'obs_properties_add_float(ppts, S_WAVE_CLAMP_MIN, obs_module_text("ClampMin"), -DBL_MAX, DBL_MAX, 0.01)',
        'obs_properties_add_float(ppts, S_WAVE_CLAMP_MAX, obs_module_text("ClampMax"), -DBL_MAX, DBL_MAX, 0.01)',
    ]
    forbidden = [token for token in forbidden_visible_defaults if token in source]
    if forbidden:
        print("move-wave-value-filter.c exposes DBL_MAX clamp sentinel values in the UI:")
        for token in forbidden:
            print(f"  - {token}")
        return 1

    locale = EN_US_LOCALE.read_text()
    required_locale_tokens = [
        "WaveType.Cosine",
        "WaveType.Pulse",
        "WaveType.SawUp",
        "WaveType.SawDown",
        "WaveType.AbsSine",
        "WaveType.Step",
        "WaveType.RandomHold",
        "WaveType.SmoothRandom",
        "PhaseOffset",
        "Invert",
        "ClampMin",
        "ClampMax",
        "ClampEnabled",
        "PulseWidth",
        "Steps",
        "RandomAmount",
    ]
    missing_locale = [token for token in required_locale_tokens if token not in locale]
    if missing_locale:
        print("en-US.ini is missing expanded wave labels:")
        for token in missing_locale:
            print(f"  - {token}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
