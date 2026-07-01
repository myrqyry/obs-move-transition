#include "move-transition.h"

#include <float.h>
#include <math.h>
#include <obs-module.h>
#include <obs-frontend-api.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WAVE_VALUE_FILTER_ID "move_wave_value_filter"

#define S_WAVE_TYPE "wave_type"
#define S_WAVE_FREQUENCY "wave_frequency"
#define S_WAVE_AMPLITUDE "wave_amplitude"
#define S_WAVE_PHASE "wave_phase"
#define S_WAVE_INVERT "wave_invert"
#define S_WAVE_CLAMP_ENABLED "wave_clamp_enabled"
#define S_WAVE_CLAMP_MIN "wave_clamp_min"
#define S_WAVE_CLAMP_MAX "wave_clamp_max"
#define S_WAVE_PULSE_WIDTH "wave_pulse_width"
#define S_WAVE_STEPS "wave_steps"
#define S_WAVE_RANDOM_AMOUNT "wave_random_amount"

#define WAVE_SINE 0
#define WAVE_SQUARE 1
#define WAVE_SAW_UP 2
#define WAVE_TRIANGLE 3
#define WAVE_COSINE 4
#define WAVE_PULSE 5
#define WAVE_SAW_DOWN 6
#define WAVE_ABS_SINE 7
#define WAVE_STEP 8
#define WAVE_RANDOM_HOLD 9
#define WAVE_SMOOTH_RANDOM 10

#define VALUE_ACTION_TRANSFORM 0
#define VALUE_ACTION_SETTING 1
#define VALUE_ACTION_SOURCE_VISIBILITY 2
#define VALUE_ACTION_FILTER_ENABLE 3

#define THRESHOLD_NONE 0
#define THRESHOLD_ENABLE_OVER 1
#define THRESHOLD_ENABLE_UNDER 2
#define THRESHOLD_DISABLE_OVER 3
#define THRESHOLD_DISABLE_UNDER 4
#define THRESHOLD_ENABLE_OVER_DISABLE_UNDER 5
#define THRESHOLD_ENABLE_UNDER_DISABLE_OVER 6

#define TRANSFORM_NONE 0
#define TRANSFORM_POS_X 1
#define TRANSFORM_POS_Y 2
#define TRANSFORM_ROT 3
#define TRANSFORM_SCALE 14
#define TRANSFORM_SCALE_X 4
#define TRANSFORM_SCALE_Y 5
#define TRANSFORM_BOUNDS_X 6
#define TRANSFORM_BOUNDS_Y 7
#define TRANSFORM_CROP_LEFT 8
#define TRANSFORM_CROP_TOP 9
#define TRANSFORM_CROP_RIGHT 10
#define TRANSFORM_CROP_BOTTOM 11
#define TRANSFORM_CROP_HORIZONTAL 12
#define TRANSFORM_CROP_VERTICAL 13

struct wave_value_info {
	struct move_filter move_filter;

	obs_sceneitem_t *sceneitem;
	obs_weak_source_t *target_source;
	char *setting_name;

	double phase;
	double frequency;
	double amplitude;
	double phase_offset;
	double pulse_width;
	double random_amount;
	double random_value;
	double next_random_value;
	double base_value;
	double factor;
	double threshold;
	double clamp_min;
	double clamp_max;
	long long action;
	long long threshold_action;
	long long transform;
	long long steps;
	bool invert;
	bool clamp_enabled;
	int wave_type;
};

static const char *wave_value_get_name(void *type_data)
{
	UNUSED_PARAMETER(type_data);
	return obs_module_text("MoveWaveValueFilter");
}

static double wave_output(struct wave_value_info *wave)
{
	double phase = wave->phase + wave->phase_offset;
	phase -= floor(phase);
	double t = phase * 2.0 * M_PI;
	double v;
	switch (wave->wave_type) {
	case WAVE_SINE:
		v = sin(t);
		break;
	case WAVE_COSINE:
		v = cos(t);
		break;
	case WAVE_SQUARE:
		v = phase < wave->pulse_width ? 1.0 : -1.0;
		break;
	case WAVE_PULSE:
		v = phase < wave->pulse_width ? 1.0 : 0.0;
		break;
	case WAVE_SAW_UP:
		v = 2.0 * phase - 1.0;
		break;
	case WAVE_SAW_DOWN:
		v = 1.0 - 2.0 * phase;
		break;
	case WAVE_TRIANGLE:
		v = 2.0 * fabs(2.0 * (phase - floor(phase + 0.5))) - 1.0;
		break;
	case WAVE_ABS_SINE:
		v = fabs(sin(t));
		break;
	case WAVE_STEP:
		v = wave->steps <= 1 ? 0.0 : (2.0 * floor(phase * (double)wave->steps) / (double)(wave->steps - 1)) - 1.0;
		break;
	case WAVE_RANDOM_HOLD:
		v = wave->random_value * wave->random_amount;
		break;
	case WAVE_SMOOTH_RANDOM: {
		double smooth = phase * phase * (3.0 - 2.0 * phase);
		v = (wave->random_value * (1.0 - smooth) + wave->next_random_value * smooth) * wave->random_amount;
		break;
	}
	default:
		v = sin(t);
		break;
	}
	if (wave->invert)
		v = -v;
	return wave->amplitude * v;
}

static double random_wave_value(void)
{
	return 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
}

static void wave_value_item_remove(void *data, calldata_t *call_data);
static void wave_value_source_remove(void *data, calldata_t *call_data);

static void wave_value_source_destroy(void *data, calldata_t *call_data)
{
	UNUSED_PARAMETER(call_data);
	struct wave_value_info *wave = data;
	wave->target_source = NULL;
	wave->sceneitem = NULL;
}

static void wave_value_disconnect_target(struct wave_value_info *wave)
{
	if (!wave->target_source)
		return;

	obs_source_t *source = obs_weak_source_get_source(wave->target_source);
	if (source) {
		signal_handler_t *sh = obs_source_get_signal_handler(source);
		signal_handler_disconnect(sh, "remove", wave_value_source_remove, wave);
		signal_handler_disconnect(sh, "destroy", wave_value_source_destroy, wave);
		obs_source_release(source);
	}
	obs_weak_source_release(wave->target_source);
	wave->target_source = NULL;
}

static void wave_value_disconnect_sceneitem(struct wave_value_info *wave)
{
	if (!wave->sceneitem)
		return;

	obs_scene_t *scene = obs_sceneitem_get_scene(wave->sceneitem);
	if (scene) {
		obs_source_t *scene_source = obs_scene_get_source(scene);
		signal_handler_t *sh = obs_source_get_signal_handler(scene_source);
		if (sh) {
			signal_handler_disconnect(sh, "item_remove", wave_value_item_remove, wave);
			signal_handler_disconnect(sh, "remove", wave_value_source_destroy, wave);
			signal_handler_disconnect(sh, "destroy", wave_value_source_destroy, wave);
		}
	}

	obs_source_t *item_source = obs_sceneitem_get_source(wave->sceneitem);
	if (item_source) {
		signal_handler_t *sh = obs_source_get_signal_handler(item_source);
		signal_handler_disconnect(sh, "remove", wave_value_source_destroy, wave);
		signal_handler_disconnect(sh, "destroy", wave_value_source_destroy, wave);
	}
	wave->sceneitem = NULL;
}

static void wave_value_source_remove(void *data, calldata_t *call_data)
{
	UNUSED_PARAMETER(call_data);
	struct wave_value_info *wave = data;
	wave_value_disconnect_target(wave);
	wave_value_disconnect_sceneitem(wave);
}

static void wave_value_item_remove(void *data, calldata_t *call_data)
{
	struct wave_value_info *wave = data;
	obs_scene_t *scene = NULL;
	calldata_get_ptr(call_data, "scene", &scene);
	obs_sceneitem_t *item = NULL;
	calldata_get_ptr(call_data, "item", &item);
	if (item == wave->sceneitem) {
		wave->sceneitem = NULL;
		obs_source_t *parent = obs_scene_get_source(scene);
		if (parent) {
			signal_handler_t *sh = obs_source_get_signal_handler(parent);
			if (sh) {
				signal_handler_disconnect(sh, "item_remove", wave_value_item_remove, wave);
				signal_handler_disconnect(sh, "remove", wave_value_source_destroy, wave);
				signal_handler_disconnect(sh, "destroy", wave_value_source_destroy, wave);
			}
		}
	}
}

static void wave_value_update_sceneitem(struct wave_value_info *wave, obs_data_t *settings)
{
	wave_value_disconnect_sceneitem(wave);

	const char *canvas_name = obs_data_get_string(settings, "canvas");
	const char *scene_name = obs_data_get_string(settings, "scene");
	const char *sceneitem_name = obs_data_get_string(settings, "sceneitem");
	obs_source_t *source = NULL;

	obs_canvas_t *canvas = canvas_name[0] == '\0' ? obs_get_main_canvas() : obs_get_canvas_by_name(canvas_name);
	if (canvas) {
		source = obs_canvas_get_source_by_name(canvas, scene_name);
		obs_canvas_release(canvas);
	}
	if (!source)
		source = obs_get_source_by_name(scene_name);
	if (!source)
		return;

	if (obs_source_removed(source)) {
		obs_source_release(source);
		return;
	}

	obs_scene_t *scene = obs_scene_from_source(source);
	if (!scene)
		scene = obs_group_from_source(source);
	wave->sceneitem = scene ? obs_scene_find_source_recursive(scene, sceneitem_name) : NULL;
	if (wave->sceneitem && obs_source_removed(obs_sceneitem_get_source(wave->sceneitem)))
		wave->sceneitem = NULL;

	if (wave->sceneitem) {
		signal_handler_t *sh = obs_source_get_signal_handler(source);
		if (sh) {
			signal_handler_connect(sh, "item_remove", wave_value_item_remove, wave);
			signal_handler_connect(sh, "remove", wave_value_source_destroy, wave);
			signal_handler_connect(sh, "destroy", wave_value_source_destroy, wave);
		}
		obs_source_t *item_source = obs_sceneitem_get_source(wave->sceneitem);
		if (item_source) {
			sh = obs_source_get_signal_handler(item_source);
			signal_handler_connect(sh, "remove", wave_value_source_destroy, wave);
			signal_handler_connect(sh, "destroy", wave_value_source_destroy, wave);
		}
	}
	obs_source_release(source);
}

static void wave_value_update_target(struct wave_value_info *wave, obs_data_t *settings)
{
	wave_value_disconnect_target(wave);

	obs_source_t *source = NULL;
	obs_source_t *target_source = NULL;
	if (wave->action == VALUE_ACTION_FILTER_ENABLE) {
		source = obs_get_source_by_name(obs_data_get_string(settings, "source"));
		if (source) {
			obs_source_t *filter = obs_source_get_filter_by_name(source, obs_data_get_string(settings, "filter"));
			if (filter)
				target_source = filter;
			obs_source_release(source);
		}
	} else if (wave->action == VALUE_ACTION_SETTING) {
		source = obs_get_source_by_name(obs_data_get_string(settings, "source"));
		if (source) {
			const char *filter_name = obs_data_get_string(settings, "filter");
			obs_source_t *filter = filter_name && strlen(filter_name) ?
				obs_source_get_filter_by_name(source, filter_name) : NULL;
			if (filter) {
				target_source = filter;
				obs_source_release(source);
			} else {
				target_source = source;
			}
		}
	}

	if (target_source && obs_source_removed(target_source)) {
		obs_source_release(target_source);
		target_source = NULL;
	}
	if (target_source) {
		wave->target_source = obs_source_get_weak_source(target_source);
		signal_handler_t *sh = obs_source_get_signal_handler(target_source);
		signal_handler_connect(sh, "remove", wave_value_source_remove, wave);
		signal_handler_connect(sh, "destroy", wave_value_source_destroy, wave);
		obs_source_release(target_source);
	}
}

static void wave_value_update(void *data, obs_data_t *settings)
{
	struct wave_value_info *wave = data;

	wave->action = obs_data_get_int(settings, "value_action");
	wave->transform = obs_data_get_int(settings, "transform");
	wave->base_value = obs_data_get_double(settings, "base_value");
	wave->factor = obs_data_get_double(settings, "factor");
	wave->threshold_action = obs_data_get_int(settings, "threshold_action");
	wave->threshold = obs_data_get_double(settings, "threshold") / 100.0;
	wave->wave_type = (int)obs_data_get_int(settings, S_WAVE_TYPE);
	wave->frequency = obs_data_get_double(settings, S_WAVE_FREQUENCY);
	wave->amplitude = obs_data_get_double(settings, S_WAVE_AMPLITUDE);
	wave->phase_offset = obs_data_get_double(settings, S_WAVE_PHASE) / 100.0;
	wave->invert = obs_data_get_bool(settings, S_WAVE_INVERT);
	wave->clamp_enabled = obs_data_get_bool(settings, S_WAVE_CLAMP_ENABLED);
	wave->clamp_min = obs_data_get_double(settings, S_WAVE_CLAMP_MIN);
	wave->clamp_max = obs_data_get_double(settings, S_WAVE_CLAMP_MAX);
	wave->pulse_width = obs_data_get_double(settings, S_WAVE_PULSE_WIDTH) / 100.0;
	wave->steps = obs_data_get_int(settings, S_WAVE_STEPS);
	wave->random_amount = obs_data_get_double(settings, S_WAVE_RANDOM_AMOUNT) / 100.0;

	const char *setting_name = obs_data_get_string(settings, "setting");
	if (!wave->setting_name || strcmp(wave->setting_name, setting_name) != 0) {
		bfree(wave->setting_name);
		wave->setting_name = bstrdup(setting_name);
	}

	if (wave->frequency <= 0.0)
		wave->frequency = 1.0;
	if (wave->pulse_width <= 0.0)
		wave->pulse_width = 0.5;
	if (wave->pulse_width >= 1.0)
		wave->pulse_width = 0.99;
	if (wave->steps < 2)
		wave->steps = 2;

	if (wave->action == VALUE_ACTION_TRANSFORM || wave->action == VALUE_ACTION_SOURCE_VISIBILITY)
		wave_value_update_sceneitem(wave, settings);
	else
		wave_value_disconnect_sceneitem(wave);

	if (wave->action == VALUE_ACTION_SETTING || wave->action == VALUE_ACTION_FILTER_ENABLE)
		wave_value_update_target(wave, settings);
	else
		wave_value_disconnect_target(wave);
}

static void *wave_value_create(obs_data_t *settings, obs_source_t *source)
{
	struct wave_value_info *wave = bzalloc(sizeof(struct wave_value_info));
	move_filter_init(&wave->move_filter, source, NULL);
	wave->frequency = 1.0;
	wave->amplitude = 1.0;
	wave->pulse_width = 0.5;
	wave->random_amount = 1.0;
	wave->random_value = random_wave_value();
	wave->next_random_value = random_wave_value();
	wave->clamp_min = -1000.0;
	wave->clamp_max = 1000.0;
	wave->steps = 4;
	wave->base_value = 0.0;
	wave->factor = 1.0;
	wave->wave_type = WAVE_SINE;

	wave_value_update(wave, settings);
	return wave;
}

static void wave_value_destroy(void *data)
{
	struct wave_value_info *wave = data;
	wave_value_disconnect_target(wave);
	wave_value_disconnect_sceneitem(wave);
	bfree(wave->setting_name);
	bfree(wave);
}

static void wave_value_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct wave_value_info *wave = data;
	obs_source_skip_video_filter(wave->move_filter.source);
}

static double wave_value(struct wave_value_info *wave)
{
	const double value = wave_output(wave);
	double val = value < wave->threshold ? wave->factor * wave->threshold + wave->base_value :
					      wave->factor * value + wave->base_value;
	if (wave->clamp_enabled && wave->clamp_min < wave->clamp_max) {
		if (val < wave->clamp_min)
			val = wave->clamp_min;
		if (val > wave->clamp_max)
			val = wave->clamp_max;
	}
	return val;
}

static void wave_value_tick(void *data, float seconds)
{
	struct wave_value_info *wave = data;
	if (!obs_source_enabled(wave->move_filter.source))
		return;

	double prev_phase = wave->phase;
	wave->phase += (double)seconds * wave->frequency;
	while (wave->phase >= 1.0)
		wave->phase -= 1.0;
	while (wave->phase < 0.0)
		wave->phase += 1.0;
	if (wave->phase < prev_phase) {
		wave->random_value = wave->next_random_value;
		wave->next_random_value = random_wave_value();
	}

	const double current_wave = wave_output(wave);
	const double val = wave_value(wave);

	if (wave->action == VALUE_ACTION_TRANSFORM) {
		if (!wave->sceneitem) {
			obs_data_t *settings = obs_source_get_settings(wave->move_filter.source);
			wave_value_update(wave, settings);
			obs_data_release(settings);
		}
		if (!wave->sceneitem)
			return;
		if (wave->transform == TRANSFORM_POS_X) {
			struct vec2 pos;
			obs_sceneitem_get_pos(wave->sceneitem, &pos);
			pos.x = (float)val;
			obs_sceneitem_set_pos(wave->sceneitem, &pos);
		} else if (wave->transform == TRANSFORM_POS_Y) {
			struct vec2 pos;
			obs_sceneitem_get_pos(wave->sceneitem, &pos);
			pos.y = (float)val;
			obs_sceneitem_set_pos(wave->sceneitem, &pos);
		} else if (wave->transform == TRANSFORM_ROT) {
			obs_sceneitem_set_rot(wave->sceneitem, (float)val);
		} else if (wave->transform == TRANSFORM_SCALE) {
			struct vec2 scale;
			scale.x = (float)val;
			scale.y = scale.x;
			obs_sceneitem_set_scale(wave->sceneitem, &scale);
		} else if (wave->transform == TRANSFORM_SCALE_X) {
			struct vec2 scale;
			obs_sceneitem_get_scale(wave->sceneitem, &scale);
			scale.x = (float)val;
			obs_sceneitem_set_scale(wave->sceneitem, &scale);
		} else if (wave->transform == TRANSFORM_SCALE_Y) {
			struct vec2 scale;
			obs_sceneitem_get_scale(wave->sceneitem, &scale);
			scale.y = (float)val;
			obs_sceneitem_set_scale(wave->sceneitem, &scale);
		} else if (wave->transform == TRANSFORM_BOUNDS_X) {
			struct vec2 bounds;
			obs_sceneitem_get_bounds(wave->sceneitem, &bounds);
			bounds.x = (float)val;
			obs_sceneitem_set_bounds(wave->sceneitem, &bounds);
		} else if (wave->transform == TRANSFORM_BOUNDS_Y) {
			struct vec2 bounds;
			obs_sceneitem_get_bounds(wave->sceneitem, &bounds);
			bounds.y = (float)val;
			obs_sceneitem_set_bounds(wave->sceneitem, &bounds);
		} else if (wave->transform == TRANSFORM_CROP_LEFT) {
			struct obs_sceneitem_crop crop;
			obs_sceneitem_get_crop(wave->sceneitem, &crop);
			crop.left = (int)val;
			obs_sceneitem_set_crop(wave->sceneitem, &crop);
		} else if (wave->transform == TRANSFORM_CROP_TOP) {
			struct obs_sceneitem_crop crop;
			obs_sceneitem_get_crop(wave->sceneitem, &crop);
			crop.top = (int)val;
			obs_sceneitem_set_crop(wave->sceneitem, &crop);
		} else if (wave->transform == TRANSFORM_CROP_RIGHT) {
			struct obs_sceneitem_crop crop;
			obs_sceneitem_get_crop(wave->sceneitem, &crop);
			crop.right = (int)val;
			obs_sceneitem_set_crop(wave->sceneitem, &crop);
		} else if (wave->transform == TRANSFORM_CROP_BOTTOM) {
			struct obs_sceneitem_crop crop;
			obs_sceneitem_get_crop(wave->sceneitem, &crop);
			crop.bottom = (int)val;
			obs_sceneitem_set_crop(wave->sceneitem, &crop);
		} else if (wave->transform == TRANSFORM_CROP_HORIZONTAL) {
			struct obs_sceneitem_crop crop;
			obs_sceneitem_get_crop(wave->sceneitem, &crop);
			crop.left = (int)val;
			crop.right = crop.left;
			obs_sceneitem_set_crop(wave->sceneitem, &crop);
		} else if (wave->transform == TRANSFORM_CROP_VERTICAL) {
			struct obs_sceneitem_crop crop;
			obs_sceneitem_get_crop(wave->sceneitem, &crop);
			crop.top = (int)val;
			crop.bottom = crop.top;
			obs_sceneitem_set_crop(wave->sceneitem, &crop);
		}
	} else if (wave->action == VALUE_ACTION_SOURCE_VISIBILITY) {
		if (!wave->sceneitem) {
			obs_data_t *settings = obs_source_get_settings(wave->move_filter.source);
			wave_value_update(wave, settings);
			obs_data_release(settings);
		}
		if (!wave->sceneitem)
			return;
		if ((wave->threshold_action == THRESHOLD_ENABLE_OVER ||
		     wave->threshold_action == THRESHOLD_ENABLE_OVER_DISABLE_UNDER) &&
		    current_wave >= wave->threshold) {
			obs_sceneitem_set_visible(wave->sceneitem, true);
		} else if ((wave->threshold_action == THRESHOLD_ENABLE_UNDER ||
			    wave->threshold_action == THRESHOLD_ENABLE_UNDER_DISABLE_OVER) &&
			   current_wave < wave->threshold) {
			obs_sceneitem_set_visible(wave->sceneitem, true);
		} else if ((wave->threshold_action == THRESHOLD_DISABLE_OVER ||
			    wave->threshold_action == THRESHOLD_ENABLE_UNDER_DISABLE_OVER) &&
			   current_wave >= wave->threshold) {
			obs_sceneitem_set_visible(wave->sceneitem, false);
		} else if ((wave->threshold_action == THRESHOLD_DISABLE_UNDER ||
			    wave->threshold_action == THRESHOLD_ENABLE_OVER_DISABLE_UNDER) &&
			   current_wave < wave->threshold) {
			obs_sceneitem_set_visible(wave->sceneitem, false);
		}
	} else if (wave->action == VALUE_ACTION_FILTER_ENABLE) {
		if (!wave->target_source) {
			obs_data_t *settings = obs_source_get_settings(wave->move_filter.source);
			wave_value_update(wave, settings);
			obs_data_release(settings);
		}
		if (!wave->target_source)
			return;
		obs_source_t *source = obs_weak_source_get_source(wave->target_source);
		if (!source)
			return;

		if ((wave->threshold_action == THRESHOLD_ENABLE_OVER ||
		     wave->threshold_action == THRESHOLD_ENABLE_OVER_DISABLE_UNDER) &&
		    current_wave >= wave->threshold && !obs_source_enabled(source)) {
			obs_source_set_enabled(source, true);
		} else if ((wave->threshold_action == THRESHOLD_ENABLE_UNDER ||
			    wave->threshold_action == THRESHOLD_ENABLE_UNDER_DISABLE_OVER) &&
			   current_wave < wave->threshold && !obs_source_enabled(source)) {
			obs_source_set_enabled(source, true);
		} else if ((wave->threshold_action == THRESHOLD_DISABLE_OVER ||
			    wave->threshold_action == THRESHOLD_ENABLE_UNDER_DISABLE_OVER) &&
			   current_wave >= wave->threshold && obs_source_enabled(source)) {
			obs_source_set_enabled(source, false);
		} else if ((wave->threshold_action == THRESHOLD_DISABLE_UNDER ||
			    wave->threshold_action == THRESHOLD_ENABLE_OVER_DISABLE_UNDER) &&
			   current_wave < wave->threshold && obs_source_enabled(source)) {
			obs_source_set_enabled(source, false);
		}
		obs_source_release(source);
	} else if (wave->action == VALUE_ACTION_SETTING && wave->setting_name && strlen(wave->setting_name)) {
		if (!wave->target_source) {
			obs_data_t *settings = obs_source_get_settings(wave->move_filter.source);
			wave_value_update(wave, settings);
			obs_data_release(settings);
		}
		if (!wave->target_source)
			return;
		obs_source_t *source = obs_weak_source_get_source(wave->target_source);
		if (!source)
			return;
		obs_data_t *settings = obs_source_get_settings(source);
		obs_data_item_t *setting = obs_data_item_byname(settings, wave->setting_name);
		if (setting) {
			const enum obs_data_number_type num_type = obs_data_item_numtype(setting);
			if (num_type == OBS_DATA_NUM_INT) {
				obs_data_item_set_int(&setting, (long long)val);
			} else if (num_type == OBS_DATA_NUM_DOUBLE) {
				obs_data_item_set_double(&setting, val);
			}
			obs_data_item_release(&setting);
		} else {
			obs_data_set_double(settings, wave->setting_name, val);
		}
		obs_source_update(source, settings);
		obs_data_release(settings);
		obs_source_release(source);
	}
}

static bool add_canvas_to_prop_list(void *data, obs_canvas_t *canvas)
{
	obs_property_t *p = (obs_property_t *)data;
	const char *name = obs_canvas_get_name(canvas);
	if (name && strlen(name))
		obs_property_list_add_string(p, name, name);
	return true;
}

static bool add_source_to_prop_list(void *data, obs_source_t *source)
{
	obs_property_t *p = (obs_property_t *)data;
	const char *name = obs_source_get_name(source);
	if (!name || !strlen(name))
		return true;
	size_t count = obs_property_list_item_count(p);
	size_t idx = 0;
	while (idx < count && strcmp(name, obs_property_list_item_string(p, idx)) > 0)
		idx++;
	obs_property_list_insert_string(p, idx, name, name);
	return true;
}

static bool add_group_to_prop_list(void *data, obs_source_t *source)
{
	obs_property_t *p = (obs_property_t *)data;
	if (!obs_source_is_group(source))
		return true;
	const char *name = obs_source_get_name(source);
	if (!name || !strlen(name))
		return true;
	size_t count = obs_property_list_item_count(p);
	size_t idx = 0;
	while (idx < count && strcmp(name, obs_property_list_item_string(p, idx)) > 0)
		idx++;
	obs_property_list_insert_string(p, idx, name, name);
	return true;
}

static bool wave_value_action_changed(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	UNUSED_PARAMETER(property);
	long long action = obs_data_get_int(settings, "value_action");
	obs_property_t *canvas = obs_properties_get(props, "canvas");
	obs_property_t *scene = obs_properties_get(props, "scene");
	obs_property_t *sceneitem = obs_properties_get(props, "sceneitem");
	if (action == VALUE_ACTION_TRANSFORM || action == VALUE_ACTION_SOURCE_VISIBILITY) {
		obs_property_list_clear(canvas);
		obs_property_list_clear(scene);
		obs_enum_canvases(add_canvas_to_prop_list, canvas);
		obs_enum_scenes(add_source_to_prop_list, scene);
		obs_enum_sources(add_group_to_prop_list, scene);
		obs_property_set_visible(canvas, true);
		obs_property_set_visible(scene, true);
		obs_property_set_visible(sceneitem, true);
	} else {
		obs_property_set_visible(canvas, false);
		obs_property_set_visible(scene, false);
		obs_property_set_visible(sceneitem, false);
	}

	obs_property_t *source = obs_properties_get(props, "source");
	obs_property_t *filter = obs_properties_get(props, "filter");
	if (action == VALUE_ACTION_SETTING || action == VALUE_ACTION_FILTER_ENABLE) {
		obs_property_list_clear(source);
		obs_enum_sources(add_source_to_prop_list, source);
		obs_enum_scenes(add_source_to_prop_list, source);
		obs_property_set_visible(source, true);
		obs_property_set_visible(filter, true);
	} else {
		obs_property_set_visible(source, false);
		obs_property_set_visible(filter, false);
	}

	obs_property_t *base_value = obs_properties_get(props, "base_value");
	obs_property_t *factor = obs_properties_get(props, "factor");
	if (action == VALUE_ACTION_SETTING || action == VALUE_ACTION_TRANSFORM) {
		obs_property_set_visible(base_value, true);
		obs_property_set_visible(factor, true);
	} else {
		obs_property_set_visible(base_value, false);
		obs_property_set_visible(factor, false);
	}

	obs_property_t *threshold_action = obs_properties_get(props, "threshold_action");
	if (action == VALUE_ACTION_SOURCE_VISIBILITY || action == VALUE_ACTION_FILTER_ENABLE)
		obs_property_set_visible(threshold_action, true);
	else
		obs_property_set_visible(threshold_action, false);

	obs_property_t *transform = obs_properties_get(props, "transform");
	obs_property_set_visible(transform, action == VALUE_ACTION_TRANSFORM);

	obs_property_t *setting = obs_properties_get(props, "setting");
	obs_property_set_visible(setting, action == VALUE_ACTION_SETTING);
	return true;
}

static void wave_value_update_type_visibility(obs_properties_t *props, long long wave_type)
{
	const bool pulse_visible = wave_type == WAVE_SQUARE || wave_type == WAVE_PULSE;
	const bool steps_visible = wave_type == WAVE_STEP;
	const bool random_visible = wave_type == WAVE_RANDOM_HOLD || wave_type == WAVE_SMOOTH_RANDOM;

	obs_property_set_visible(obs_properties_get(props, S_WAVE_PULSE_WIDTH), pulse_visible);
	obs_property_set_visible(obs_properties_get(props, S_WAVE_STEPS), steps_visible);
	obs_property_set_visible(obs_properties_get(props, S_WAVE_RANDOM_AMOUNT), random_visible);
}

static bool wave_value_clamp_changed(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	UNUSED_PARAMETER(property);
	const bool clamp_enabled = obs_data_get_bool(settings, S_WAVE_CLAMP_ENABLED);
	obs_property_set_visible(obs_properties_get(props, S_WAVE_CLAMP_MIN), clamp_enabled);
	obs_property_set_visible(obs_properties_get(props, S_WAVE_CLAMP_MAX), clamp_enabled);
	return true;
}

static bool wave_value_type_changed(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	UNUSED_PARAMETER(property);
	wave_value_update_type_visibility(props, obs_data_get_int(settings, S_WAVE_TYPE));
	return true;
}

static bool add_scene_to_prop_list(void *data, obs_source_t *source)
{
	obs_property_t *p = (obs_property_t *)data;
	const char *name = obs_source_get_name(source);
	if (name && strlen(name))
		obs_property_list_add_string(p, name, name);
	return true;
}

static bool wave_value_canvas_changed(void *data, obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	UNUSED_PARAMETER(data);
	UNUSED_PARAMETER(property);
	const char *canvas_name = obs_data_get_string(settings, "canvas");
	obs_property_t *scene = obs_properties_get(props, "scene");
	obs_property_list_clear(scene);

	obs_canvas_t *canvas = canvas_name[0] == '\0' ? obs_get_main_canvas() : obs_get_canvas_by_name(canvas_name);
	if (!canvas)
		return true;

	obs_canvas_enum_scenes(canvas, add_scene_to_prop_list, scene);
	obs_canvas_release(canvas);
	return true;
}

static bool add_sceneitem_to_prop_list(obs_scene_t *scene, obs_sceneitem_t *item, void *data)
{
	UNUSED_PARAMETER(scene);
	obs_property_t *p = (obs_property_t *)data;
	const obs_source_t *source = obs_sceneitem_get_source(item);
	const char *name = obs_source_get_name(source);
	if (name && strlen(name))
		obs_property_list_add_string(p, name, name);
	return true;
}

static bool wave_value_scene_changed(void *data, obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	UNUSED_PARAMETER(data);
	UNUSED_PARAMETER(property);
	const char *canvas_name = obs_data_get_string(settings, "canvas");
	obs_canvas_t *canvas = canvas_name[0] == '\0' ? obs_get_main_canvas() : obs_get_canvas_by_name(canvas_name);
	const char *scene_name = obs_data_get_string(settings, "scene");
	obs_property_t *sceneitem = obs_properties_get(props, "sceneitem");
	obs_property_list_clear(sceneitem);
	obs_source_t *source = NULL;
	if (canvas) {
		source = obs_canvas_get_source_by_name(canvas, scene_name);
		obs_canvas_release(canvas);
	}
	if (!source)
		source = obs_get_source_by_name(scene_name);
	if (!source)
		return true;
	obs_scene_t *scene = obs_scene_from_source(source);
	if (!scene)
		scene = obs_group_from_source(source);
	if (scene)
		obs_scene_enum_items(scene, add_sceneitem_to_prop_list, sceneitem);
	obs_source_release(source);
	return true;
}

static void add_filter_to_prop_list(obs_source_t *parent, obs_source_t *child, void *data)
{
	UNUSED_PARAMETER(parent);
	obs_property_t *p = (obs_property_t *)data;
	const char *name = obs_source_get_name(child);
	const char *src_id = obs_source_get_id(child);

	if (name && strlen(name) && strcmp(src_id, AUDIO_MOVE_FILTER_ID) != 0 && strcmp(src_id, WAVE_VALUE_FILTER_ID) != 0)
		obs_property_list_add_string(p, name, name);
}

static void load_properties(obs_properties_t *props_from, obs_property_t *setting_list)
{
	obs_property_t *prop_from = obs_properties_first(props_from);
	for (; prop_from != NULL; obs_property_next(&prop_from)) {
		const char *name = obs_property_name(prop_from);
		const char *description = obs_property_description(prop_from);
		if (!obs_property_visible(prop_from))
			continue;
		const enum obs_property_type prop_type = obs_property_get_type(prop_from);
		if (prop_type == OBS_PROPERTY_GROUP) {
			load_properties(obs_property_group_content(prop_from), setting_list);
		} else if (prop_type == OBS_PROPERTY_FLOAT || prop_type == OBS_PROPERTY_INT) {
			obs_property_list_add_string(setting_list, description ? description : name, name);
		}
	}
}

static bool wave_value_source_changed(void *data, obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	UNUSED_PARAMETER(property);
	UNUSED_PARAMETER(data);
	const char *source_name = obs_data_get_string(settings, "source");
	const char *filter_name = obs_data_get_string(settings, "filter");
	obs_property_t *filter = obs_properties_get(props, "filter");
	obs_property_list_clear(filter);
	obs_source_t *source = obs_get_source_by_name(source_name);
	if (!source)
		return true;
	obs_source_enum_filters(source, add_filter_to_prop_list, filter);

	obs_property_t *setting = obs_properties_get(props, "setting");
	obs_property_list_clear(setting);
	obs_properties_t *properties = NULL;
	if (filter_name && strlen(filter_name)) {
		obs_source_t *f = obs_source_get_filter_by_name(source, filter_name);
		if (f) {
			properties = obs_source_properties(f);
			obs_source_release(f);
		}
	} else {
		properties = obs_source_properties(source);
	}
	if (properties) {
		load_properties(properties, setting);
		obs_properties_destroy(properties);
	}
	obs_source_release(source);
	return true;
}

static obs_properties_t *wave_value_properties(void *data)
{
	obs_properties_t *ppts = obs_properties_create();

	obs_property_t *p = obs_properties_add_list(ppts, "value_action", obs_module_text("ValueAction"), OBS_COMBO_TYPE_LIST,
					    OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, obs_module_text("ValueAction.Transform"), VALUE_ACTION_TRANSFORM);
	obs_property_list_add_int(p, obs_module_text("ValueAction.Setting"), VALUE_ACTION_SETTING);
	obs_property_list_add_int(p, obs_module_text("ValueAction.SourceVisibility"), VALUE_ACTION_SOURCE_VISIBILITY);
	obs_property_list_add_int(p, obs_module_text("ValueAction.FilterEnable"), VALUE_ACTION_FILTER_ENABLE);
	obs_property_set_modified_callback(p, wave_value_action_changed);

	p = obs_properties_add_list(ppts, "canvas", obs_module_text("Canvas"), OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
	obs_property_set_modified_callback2(p, wave_value_canvas_changed, data);

	p = obs_properties_add_list(ppts, "scene", obs_module_text("Scene"), OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
	obs_property_set_modified_callback2(p, wave_value_scene_changed, data);

	obs_properties_add_list(ppts, "sceneitem", obs_module_text("Source"), OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);

	p = obs_properties_add_list(ppts, "source", obs_module_text("Source"), OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
	obs_property_set_modified_callback2(p, wave_value_source_changed, data);

	p = obs_properties_add_list(ppts, "filter", obs_module_text("Filter"), OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
	obs_property_set_modified_callback2(p, wave_value_source_changed, data);

	p = obs_properties_add_list(ppts, "transform", obs_module_text("Transform"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, obs_module_text("Transform.PosX"), TRANSFORM_POS_X);
	obs_property_list_add_int(p, obs_module_text("Transform.PosY"), TRANSFORM_POS_Y);
	obs_property_list_add_int(p, obs_module_text("Transform.Rotation"), TRANSFORM_ROT);
	obs_property_list_add_int(p, obs_module_text("Transform.Scale"), TRANSFORM_SCALE);
	obs_property_list_add_int(p, obs_module_text("Transform.ScaleX"), TRANSFORM_SCALE_X);
	obs_property_list_add_int(p, obs_module_text("Transform.ScaleY"), TRANSFORM_SCALE_Y);
	obs_property_list_add_int(p, obs_module_text("Transform.BoundsX"), TRANSFORM_BOUNDS_X);
	obs_property_list_add_int(p, obs_module_text("Transform.BoundsY"), TRANSFORM_BOUNDS_Y);
	obs_property_list_add_int(p, obs_module_text("Transform.CropLeft"), TRANSFORM_CROP_LEFT);
	obs_property_list_add_int(p, obs_module_text("Transform.CropTop"), TRANSFORM_CROP_TOP);
	obs_property_list_add_int(p, obs_module_text("Transform.CropRight"), TRANSFORM_CROP_RIGHT);
	obs_property_list_add_int(p, obs_module_text("Transform.CropBottom"), TRANSFORM_CROP_BOTTOM);
	obs_property_list_add_int(p, obs_module_text("Transform.CropHorizontal"), TRANSFORM_CROP_HORIZONTAL);
	obs_property_list_add_int(p, obs_module_text("Transform.CropVertical"), TRANSFORM_CROP_VERTICAL);

	obs_properties_add_list(ppts, "setting", obs_module_text("Setting"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_properties_add_float(ppts, "base_value", obs_module_text("BaseValue"), -DBL_MAX, DBL_MAX, 0.01);
	obs_properties_add_float(ppts, "factor", obs_module_text("Factor"), -DBL_MAX, DBL_MAX, 0.01);
	p = obs_properties_add_list(ppts, "threshold_action", obs_module_text("ThresholdAction"), OBS_COMBO_TYPE_LIST,
					    OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, obs_module_text("ThresholdAction.None"), THRESHOLD_NONE);
	obs_property_list_add_int(p, obs_module_text("ThresholdAction.EnableOver"), THRESHOLD_ENABLE_OVER);
	obs_property_list_add_int(p, obs_module_text("ThresholdAction.EnableUnder"), THRESHOLD_ENABLE_UNDER);
	obs_property_list_add_int(p, obs_module_text("ThresholdAction.DisableOver"), THRESHOLD_DISABLE_OVER);
	obs_property_list_add_int(p, obs_module_text("ThresholdAction.DisableUnder"), THRESHOLD_DISABLE_UNDER);
	obs_property_list_add_int(p, obs_module_text("ThresholdAction.EnableOverDisableUnder"),
			      THRESHOLD_ENABLE_OVER_DISABLE_UNDER);
	obs_property_list_add_int(p, obs_module_text("ThresholdAction.EnableUnderDisableOver"),
			      THRESHOLD_ENABLE_UNDER_DISABLE_OVER);
	obs_properties_add_float_slider(ppts, "threshold", obs_module_text("Threshold"), 0.0, 100.0, 0.01);

	p = obs_properties_add_list(ppts, S_WAVE_TYPE, obs_module_text("WaveType"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, obs_module_text("WaveType.Sine"), WAVE_SINE);
	obs_property_list_add_int(p, obs_module_text("WaveType.Cosine"), WAVE_COSINE);
	obs_property_list_add_int(p, obs_module_text("WaveType.Square"), WAVE_SQUARE);
	obs_property_list_add_int(p, obs_module_text("WaveType.Pulse"), WAVE_PULSE);
	obs_property_list_add_int(p, obs_module_text("WaveType.SawUp"), WAVE_SAW_UP);
	obs_property_list_add_int(p, obs_module_text("WaveType.SawDown"), WAVE_SAW_DOWN);
	obs_property_list_add_int(p, obs_module_text("WaveType.Triangle"), WAVE_TRIANGLE);
	obs_property_list_add_int(p, obs_module_text("WaveType.AbsSine"), WAVE_ABS_SINE);
	obs_property_list_add_int(p, obs_module_text("WaveType.Step"), WAVE_STEP);
	obs_property_list_add_int(p, obs_module_text("WaveType.RandomHold"), WAVE_RANDOM_HOLD);
	obs_property_list_add_int(p, obs_module_text("WaveType.SmoothRandom"), WAVE_SMOOTH_RANDOM);
	obs_property_set_modified_callback(p, wave_value_type_changed);

	obs_properties_add_float_slider(ppts, S_WAVE_FREQUENCY, obs_module_text("Frequency"), 0.01, 10.0, 0.01);
	obs_properties_add_float_slider(ppts, S_WAVE_AMPLITUDE, obs_module_text("Amplitude"), 0.0, 10.0, 0.01);
	obs_properties_add_float_slider(ppts, S_WAVE_PHASE, obs_module_text("PhaseOffset"), 0.0, 100.0, 0.01);
	obs_properties_add_bool(ppts, S_WAVE_INVERT, obs_module_text("Invert"));
	p = obs_properties_add_bool(ppts, S_WAVE_CLAMP_ENABLED, obs_module_text("ClampEnabled"));
	obs_property_set_modified_callback(p, wave_value_clamp_changed);
	obs_properties_add_float(ppts, S_WAVE_CLAMP_MIN, obs_module_text("ClampMin"), -1000000.0, 1000000.0, 0.01);
	obs_properties_add_float(ppts, S_WAVE_CLAMP_MAX, obs_module_text("ClampMax"), -1000000.0, 1000000.0, 0.01);
	obs_properties_add_float_slider(ppts, S_WAVE_PULSE_WIDTH, obs_module_text("PulseWidth"), 1.0, 99.0, 0.01);
	obs_properties_add_int_slider(ppts, S_WAVE_STEPS, obs_module_text("Steps"), 2, 64, 1);
	obs_properties_add_float_slider(ppts, S_WAVE_RANDOM_AMOUNT, obs_module_text("RandomAmount"), 0.0, 100.0, 0.01);
	if (data) {
		struct wave_value_info *wave = data;
		wave_value_update_type_visibility(ppts, wave->wave_type);
		obs_property_set_visible(obs_properties_get(ppts, S_WAVE_CLAMP_MIN), wave->clamp_enabled);
		obs_property_set_visible(obs_properties_get(ppts, S_WAVE_CLAMP_MAX), wave->clamp_enabled);
	} else {
		wave_value_update_type_visibility(ppts, WAVE_SINE);
		obs_property_set_visible(obs_properties_get(ppts, S_WAVE_CLAMP_MIN), false);
		obs_property_set_visible(obs_properties_get(ppts, S_WAVE_CLAMP_MAX), false);
	}
	obs_properties_add_text(ppts, "plugin_info", PLUGIN_INFO, OBS_TEXT_INFO);
	return ppts;
}

static void wave_value_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, S_WAVE_TYPE, WAVE_SINE);
	obs_data_set_default_double(settings, S_WAVE_FREQUENCY, 1.0);
	obs_data_set_default_double(settings, S_WAVE_AMPLITUDE, 1.0);
	obs_data_set_default_double(settings, S_WAVE_PHASE, 0.0);
	obs_data_set_default_bool(settings, S_WAVE_INVERT, false);
	obs_data_set_default_bool(settings, S_WAVE_CLAMP_ENABLED, false);
	obs_data_set_default_double(settings, S_WAVE_CLAMP_MIN, -1000.0);
	obs_data_set_default_double(settings, S_WAVE_CLAMP_MAX, 1000.0);
	obs_data_set_default_double(settings, S_WAVE_PULSE_WIDTH, 50.0);
	obs_data_set_default_int(settings, S_WAVE_STEPS, 4);
	obs_data_set_default_double(settings, S_WAVE_RANDOM_AMOUNT, 100.0);
	obs_data_set_default_double(settings, "base_value", 0.0);
	obs_data_set_default_double(settings, "factor", 1.0);
}

struct obs_source_info move_wave_value_filter = {
	.id = WAVE_VALUE_FILTER_ID,
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = wave_value_get_name,
	.create = wave_value_create,
	.destroy = wave_value_destroy,
	.get_properties = wave_value_properties,
	.get_defaults = wave_value_defaults,
	.video_render = wave_value_video_render,
	.video_tick = wave_value_tick,
	.update = wave_value_update,
	.load = wave_value_update,
};
