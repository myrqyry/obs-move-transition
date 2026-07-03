#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NVIDIA_FILTER = ROOT / "nvidia-move-filter.c"


def main():
	source = NVIDIA_FILTER.read_text()

	required_tokens = [
		"value = filter->joint_angles.array[action->feature_number[0]].x;",
		"value = filter->joint_angles.array[action->feature_number[0]].y;",
		"value = filter->joint_angles.array[action->feature_number[0]].z;",
		"size_t old_num = filter->actions.num;\n\t\tda_resize(filter->actions, actions);\n\t\tfor (size_t i = old_num; i < actions; i++)",
		'obs_property_list_add_int(p, "eyeLookLeft", -18);',
		'obs_property_list_add_int(p, "eyeLookRight", -21);',
		'obs_property_list_add_int(p, "eyeLookSideways", -28);',
		'obs_property_list_add_int(p, "eyeLookUpDown", -29);',
	]
	missing = [token for token in required_tokens if token not in source]
	if missing:
		print("nvidia-move-filter.c is missing NVIDIA move regression fixes:")
		for token in missing:
			print(f"  - {token}")
		return 1

	swap_action_start = source.find("static void swap_action")
	swap_action_end = source.find("static bool nv_move_move_up_clicked")
	if swap_action_start == -1 or swap_action_end == -1:
		print("nvidia-move-filter.c swap_action boundaries changed")
		return 1

	swap_action = source[swap_action_start:swap_action_end]
	if '"action_%lld_canvas"' not in swap_action:
		print("nvidia-move-filter.c does not swap action canvas settings")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
