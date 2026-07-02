// MediaPipe Move Filter - mirror of nvidia-move-filter.c with MediaPipe C++ backend
// Uses MediaPipe Tasks C++ API (FaceLandmarker, PoseLandmarker) via runtime loading

#include "mediapipe-load.h"
#include "move-filter-actions.h"
#include <obs-module.h>
#include <util/threading.h>
#include <util/dstr.h>
#include <util/platform.h>
#include <util/c99defs.h>
#include "easing.h"
#include "move-transition.h"

#define BBOXES_COUNT 25
#define MAX_ACTIONS 40

// Forward declarations matching nvidia-move-filter.c pattern
static const char *mp_move_name(void *unused);
static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function);
static bool mp_move_action_get_float(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, float *v);
static void mp_move_action_get_vec2(struct mediapipe_move_info *filter, struct nvidia_move_action *action, bool easing, struct vec2 *value);
static void mp_move_update(void *data, obs_data_t *settings);
static void *mp_move_create(obs_data_t *settings, obs_source_t *context);
static void mp_move_destroy(void *data);
static void mp_move_actual_destroy(void *data);
static void mp_move_render(void *data, gs_effect_t *effect);
static obs_properties_t *mp_move_properties(void *data);
static void mp_move_fill_body_list(obs_property_t *p);
static void mp_move_fill_landmark_list(obs_property_t *p);
static void mp_move_fill_expression_list(obs_property_t *p);
static void mp_move_tick(void *data, float t);
static enum gs_color_space mp_move_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces);
static void mp_move_defaults(obs_data_t *settings);
static void swap_setting(obs_data_t *settings, char *setting1, char *setting2);
static void swap_action(obs_data_t *settings, long long a, long long b);
static bool mp_move_move_up_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_move_down_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_get_value_clicked(obs_properties_t *props, obs_property_t *property, void *data);
static bool mp_move_feature_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_landmark_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_body_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_expression_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool mp_move_actions_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
void mp_move_register(void);

// MediaPipe Move Filter struct
struct mediapipe_move_info {
	obs_source_t *source;
	DARRAY(struct nvidia_move_action) actions;
	void *face_graph;
	void *pose_graph;
	char *last_error;
	bool got_new_frame;
	bool processing_stop;
	bool target_valid;
	bool initial_render;
	bool processed_frame;
	uint32_t width;
	uint32_t height;

	// Face detection thresholds (passed to create_face_landmarker options)
	float face_detection_confidence;  // min_face_detection_confidence
	float face_presence_confidence;   // min_face_presence_confidence
	float face_tracking_confidence;   // min_tracking_confidence

	DARRAY(float) landmarks_confidence;
	DARRAY(mp_point3f) landmarks;            // 478 3D landmarks
	DARRAY(float) gaze_output_landmarks;      // unused until gaze is implemented
	DARRAY(float) bboxes_confidence;
	mp_rect_t bbox;
	DARRAY(float) expressions;                // blendshape coefficients (52 or more)
	float face_transform_matrix[16];          // 4x4 row-major transformation matrix
	DARRAY(float) keypoints_confidence;
	DARRAY(mp_point2f) keypoints;
	DARRAY(mp_point3f) keypoints3D;
	DARRAY(mp_quaternion) joint_angles;

	gs_texrender_t *render;
	gs_texrender_t *render_unorm;
	enum gs_color_space space;

	gs_effect_t *effect;
	gs_eparam_t *image_param;
	gs_eparam_t *multiplier_param;
};

static const char *mp_move_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("MediaPipeMoveFilter");
}

static bool mp_move_log_error(struct mediapipe_move_info *filter, bool success, const char *function)
{
	if (success)
		return false;
	const char *err_string = mediapipe_api.get_last_error ? mediapipe_api.get_last_error() : "unknown error";
	if (filter->last_error && strcmp(err_string, filter->last_error) == 0)
		return true;
	blog(LOG_ERROR, "[MediaPipe Move] Error in %s: %s", function, err_string);
	bfree(filter->last_error);
	filter->last_error = bstrdup(err_string);
	return true;
}

static bool mp_check_model(const char *name)
{
	char *path = obs_module_file(name);
	bool exists = os_file_exists(path);
	if (!exists) {
		blog(LOG_ERROR, "[MediaPipe] Model file missing: %s. Please download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", name);
	}
	bfree(path);
	return exists;
}

static void *mp_move_create(obs_data_t *settings, obs_source_t *context)
{
	if (!mediapipe_is_loaded()) return NULL;

	struct mediapipe_move_info *filter = (struct mediapipe_move_info *)bzalloc(sizeof(*filter));
	filter->source = context;
	filter->space = GS_CS_SRGB;

	// Default confidence thresholds
	filter->face_detection_confidence = 0.5f;
	filter->face_presence_confidence = 0.5f;
	filter->face_tracking_confidence = 0.5f;

	char *effect_path = obs_module_file("effects/unorm.effect");
	obs_enter_graphics();
	filter->effect = gs_effect_create_from_file(effect_path, NULL);
	bfree(effect_path);
	if (filter->effect) {
		filter->image_param = gs_effect_get_param_by_name(filter->effect, "image");
		filter->multiplier_param = gs_effect_get_param_by_name(filter->effect, "multiplier");
	}
	obs_leave_graphics();

	// Handles created lazily in mp_move_update based on feature_flags
	obs_source_update(context, settings);
	return filter;
}

static void mp_move_destroy(void *data)
{
	struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;
	// Release all action resources before freeing the array
	for (size_t i = 0; i < filter->actions.num; i++) {
		struct nvidia_move_action *action = filter->actions.array + i;
		obs_weak_source_release(action->target);
		action->target = NULL;
		bfree(action->name);
		action->name = NULL;
	}
	da_free(filter->actions);
	obs_queue_task(OBS_TASK_GRAPHICS, mp_move_actual_destroy, data, false);
}

static void mp_move_actual_destroy(void *data)
{
	struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;

	mp_api_t *api = mediapipe_get_api();

	// Destroy face landmarker handle if it exists
	if (filter->face_graph) {
		api->destroy_face_landmarker((mp_face_landmarker_t *)filter->face_graph);
		filter->face_graph = NULL;
	}

	// Destroy pose landmarker handle if it exists
	if (filter->pose_graph) {
		api->destroy_pose_landmarker((mp_pose_landmarker_t *)filter->pose_graph);
		filter->pose_graph = NULL;
	}

	// Free all DARRAY members
	da_free(filter->landmarks_confidence);
	da_free(filter->landmarks);
	da_free(filter->gaze_output_landmarks);
	da_free(filter->bboxes_confidence);
	da_free(filter->expressions);
	da_free(filter->keypoints_confidence);
	da_free(filter->keypoints);
	da_free(filter->keypoints3D);
	da_free(filter->joint_angles);

	// Destroy graphics resources (must be on graphics thread)
	obs_enter_graphics();
	gs_texrender_destroy(filter->render);
	filter->render = NULL;
	gs_texrender_destroy(filter->render_unorm);
	filter->render_unorm = NULL;
	if (filter->effect) {
		gs_effect_destroy(filter->effect);
		filter->effect = NULL;
	}
	obs_leave_graphics();

	bfree(filter->last_error);
	bfree(filter);
}

static void mp_move_render(void *data, gs_effect_t *effect) { UNUSED_PARAMETER(data); UNUSED_PARAMETER(effect); }
static void mp_move_tick(void *data, float t) { UNUSED_PARAMETER(data); UNUSED_PARAMETER(t); }
static enum gs_color_space mp_move_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces) { UNUSED_PARAMETER(data); UNUSED_PARAMETER(count); UNUSED_PARAMETER(preferred_spaces); return GS_CS_SRGB; }
static void mp_move_defaults(obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
	obs_data_set_default_int(settings, "actions", 1);
	obs_data_set_default_double(settings, "face_detection_confidence", 0.5);
	obs_data_set_default_double(settings, "face_presence_confidence", 0.5);
	obs_data_set_default_double(settings, "face_tracking_confidence", 0.5);
}
static obs_properties_t *mp_move_properties(void *data) { UNUSED_PARAMETER(data); return obs_properties_create(); }
static void mp_move_fill_body_list(obs_property_t *p) { UNUSED_PARAMETER(p); }
static void mp_move_fill_landmark_list(obs_property_t *p) { UNUSED_PARAMETER(p); }
static void mp_move_fill_expression_list(obs_property_t *p) { UNUSED_PARAMETER(p); }
static void swap_setting(obs_data_t *settings, char *setting1, char *setting2) { UNUSED_PARAMETER(settings); UNUSED_PARAMETER(setting1); UNUSED_PARAMETER(setting2); }
static void swap_action(obs_data_t *settings, long long a, long long b) { UNUSED_PARAMETER(settings); UNUSED_PARAMETER(a); UNUSED_PARAMETER(b); }
static bool mp_move_move_up_clicked(obs_properties_t *props, obs_property_t *property, void *data) { UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(data); return true; }
static bool mp_move_move_down_clicked(obs_properties_t *props, obs_property_t *property, void *data) { UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(data); return true; }
static bool mp_move_get_value_clicked(obs_properties_t *props, obs_property_t *property, void *data) { UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(data); return true; }
static bool mp_move_feature_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { UNUSED_PARAMETER(priv); UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(settings); return true; }
static bool mp_move_landmark_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { UNUSED_PARAMETER(priv); UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(settings); return true; }
static bool mp_move_body_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { UNUSED_PARAMETER(priv); UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(settings); return true; }
static bool mp_move_expression_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { UNUSED_PARAMETER(priv); UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(settings); return true; }
static bool mp_move_actions_changed(void *priv, obs_properties_t *props, obs_property_t *property, obs_data_t *settings) { UNUSED_PARAMETER(priv); UNUSED_PARAMETER(props); UNUSED_PARAMETER(property); UNUSED_PARAMETER(settings); return true; }
static void mp_move_update(void *data, obs_data_t *settings)
{
	struct mediapipe_move_info *filter = (struct mediapipe_move_info *)data;

	// --- Resize actions array (mirrors nv_move_update) ---
	size_t actions = (size_t)obs_data_get_int(settings, "actions");
	if (actions < filter->actions.num) {
		// Shrinking: release resources for removed actions
		for (size_t i = actions; i < filter->actions.num; i++) {
			struct nvidia_move_action *action = filter->actions.array + i;
			obs_weak_source_release(action->target);
			action->target = NULL;
			bfree(action->name);
			action->name = NULL;
		}
		da_resize(filter->actions, actions);
	} else if (actions > filter->actions.num) {
		// Growing: zero-initialize new action slots
		size_t old_num = filter->actions.num;
		da_resize(filter->actions, actions);
		for (size_t i = old_num; i < actions; i++) {
			struct nvidia_move_action *action = filter->actions.array + i;
			memset(action, 0, sizeof(struct nvidia_move_action));
		}
	}

	// --- Read confidence threshold settings ---
	filter->face_detection_confidence = (float)obs_data_get_double(settings, "face_detection_confidence");
	filter->face_presence_confidence = (float)obs_data_get_double(settings, "face_presence_confidence");
	filter->face_tracking_confidence = (float)obs_data_get_double(settings, "face_tracking_confidence");

	// --- Parse each action from settings (mirrors nv_move_update closely) ---
	uint64_t feature_flags = 0;
	struct dstr name = {0};
	for (size_t i = 1; i <= actions; i++) {
		struct nvidia_move_action *action = filter->actions.array + i - 1;
		dstr_printf(&name, "action_%lld_disabled", i);
		action->disabled = obs_data_get_bool(settings, name.array);
		dstr_printf(&name, "action_%lld_action", i);
		action->action = (uint32_t)obs_data_get_int(settings, name.array);
		obs_weak_source_release(action->target);
		action->target = NULL;
		bfree(action->name);
		action->name = NULL;

		if (action->action == ACTION_MOVE_SOURCE || action->action == ACTION_SOURCE_VISIBILITY ||
		    action->action == ACTION_ATTACH_SOURCE) {
			dstr_printf(&name, "action_%lld_canvas", i);
			const char *canvas_name = obs_data_get_string(settings, name.array);
			obs_canvas_t *canvas = canvas_name[0] == '\0' ? obs_get_main_canvas() : obs_get_canvas_by_name(canvas_name);
			dstr_printf(&name, "action_%lld_scene", i);
			const char *scene_name = obs_data_get_string(settings, name.array);
			obs_source_t *source = NULL;
			if (canvas) {
				source = obs_canvas_get_source_by_name(canvas, scene_name);
				obs_canvas_release(canvas);
			}
			if (!source)
				source = obs_get_source_by_name(scene_name);
			if (source) {
				if (obs_source_is_scene(source) || obs_source_is_group(source))
					action->target = obs_source_get_weak_source(source);
				obs_source_release(source);
			}
			dstr_printf(&name, "action_%lld_sceneitem", i);
			action->name = bstrdup(obs_data_get_string(settings, name.array));
		}

		if (action->action == ACTION_MOVE_SOURCE) {
			dstr_printf(&name, "action_%lld_sceneitem_property", i);
			action->property = (uint32_t)obs_data_get_int(settings, name.array);
		} else if (action->action == ACTION_ENABLE_FILTER || action->action == ACTION_SOURCE_VISIBILITY) {
			dstr_printf(&name, "action_%lld_enable", i);
			action->property = (uint32_t)obs_data_get_int(settings, name.array);
			dstr_printf(&name, "action_%lld_threshold", i);
			action->threshold = (float)obs_data_get_double(settings, name.array);
		} else if (action->action == ACTION_ATTACH_SOURCE) {
			dstr_printf(&name, "action_%lld_attach", i);
			action->property = (uint32_t)obs_data_get_int(settings, name.array);
		}

		if (action->action == ACTION_MOVE_VALUE || action->action == ACTION_ENABLE_FILTER) {
			dstr_printf(&name, "action_%lld_source", i);
			const char *source_name = obs_data_get_string(settings, name.array);
			obs_source_t *source = obs_get_source_by_name(source_name);
			if (source) {
				dstr_printf(&name, "action_%lld_filter", i);
				const char *filter_name = obs_data_get_string(settings, name.array);
				obs_source_t *f = obs_source_get_filter_by_name(source, filter_name);
				if (f || action->action == ACTION_ENABLE_FILTER) {
					obs_source_release(source);
					source = f;
				}
				action->target = obs_source_get_weak_source(source);

				if (action->action == ACTION_MOVE_VALUE) {
					dstr_printf(&name, "action_%lld_property", i);
					action->name = bstrdup(obs_data_get_string(settings, name.array));

					obs_properties_t *sp = obs_source_properties(source);
					obs_property_t *p = obs_properties_get(sp, action->name);
					action->is_int = (obs_property_get_type(p) == OBS_PROPERTY_INT);
					obs_properties_destroy(sp);
				}
				obs_source_release(source);
			}
		}

		if (action->action == ACTION_ATTACH_SOURCE) {
			action->feature = FEATURE_LANDMARK;
		} else {
			dstr_printf(&name, "action_%lld_feature", i);
			action->feature = (uint32_t)obs_data_get_int(settings, name.array);
		}
		feature_flags |= (1ull << action->feature);

		if (action->feature == FEATURE_BOUNDINGBOX) {
			dstr_printf(&name, "action_%lld_bounding_box", i);
			action->feature_property = (uint32_t)obs_data_get_int(settings, name.array);
			dstr_printf(&name, "action_%lld_required_confidence", i);
			action->required_confidence = (float)obs_data_get_double(settings, name.array);
		} else if (action->feature == FEATURE_LANDMARK) {
			dstr_printf(&name, "action_%lld_landmark", i);
			action->feature_property = (uint32_t)obs_data_get_int(settings, name.array);
			dstr_printf(&name, "action_%lld_landmark_1", i);
			if (obs_data_get_int(settings, name.array))
				action->feature_number[0] = (int32_t)obs_data_get_int(settings, name.array) - 1;
			dstr_printf(&name, "action_%lld_landmark_2", i);
			if (obs_data_get_int(settings, name.array))
				action->feature_number[1] = (int32_t)obs_data_get_int(settings, name.array) - 1;
			dstr_printf(&name, "action_%lld_required_confidence", i);
			action->required_confidence = (float)obs_data_get_double(settings, name.array);
		} else if (action->feature == FEATURE_POSE) {
			dstr_printf(&name, "action_%lld_pose", i);
			action->feature_property = (uint32_t)obs_data_get_int(settings, name.array);
		} else if (action->feature == FEATURE_EXPRESSION) {
			dstr_printf(&name, "action_%lld_expression_property", i);
			action->feature_property = (uint32_t)obs_data_get_int(settings, name.array);
			dstr_printf(&name, "action_%lld_expression", i);
			// Migration from old single-expression setting
			long long old = obs_data_get_int(settings, name.array);
			if (old) {
				obs_data_unset_user_value(settings, name.array);
				dstr_printf(&name, "action_%lld_expression_1", i);
				obs_data_set_int(settings, name.array, old);
			}
			dstr_printf(&name, "action_%lld_expression_1", i);
			action->feature_number[0] = (int32_t)obs_data_get_int(settings, name.array) - 1;
			dstr_printf(&name, "action_%lld_expression_2", i);
			action->feature_number[1] = (int32_t)obs_data_get_int(settings, name.array) - 1;
		} else if (action->feature == FEATURE_GAZE) {
			dstr_printf(&name, "action_%lld_gaze", i);
			action->feature_property = (int32_t)obs_data_get_int(settings, name.array);
		} else if (action->feature == FEATURE_BODY) {
			dstr_printf(&name, "action_%lld_body", i);
			action->feature_property = (int32_t)obs_data_get_int(settings, name.array);
			dstr_printf(&name, "action_%lld_body_1", i);
			action->feature_number[0] = (int32_t)obs_data_get_int(settings, name.array);
			dstr_printf(&name, "action_%lld_body_2", i);
			action->feature_number[1] = (int32_t)obs_data_get_int(settings, name.array);
			dstr_printf(&name, "action_%lld_required_confidence", i);
			action->required_confidence = (float)obs_data_get_double(settings, name.array);
		}

		// Factor/diff defaults and reads (mirrors NVIDIA pattern)
		dstr_printf(&name, "action_%lld_factor2", i);
		obs_data_set_default_double(settings, name.array, 100.0f);
		dstr_printf(&name, "action_%lld_factor", i);
		obs_data_set_default_double(settings, name.array, 100.0f);
		action->factor = (float)obs_data_get_double(settings, name.array) / 100.0f;
		dstr_printf(&name, "action_%lld_factor2", i);
		action->factor2 = (float)obs_data_get_double(settings, name.array) / 100.0f;
		dstr_printf(&name, "action_%lld_diff", i);
		action->diff = (float)obs_data_get_double(settings, name.array);
		dstr_printf(&name, "action_%lld_diff2", i);
		action->diff2 = (float)obs_data_get_double(settings, name.array);

		dstr_printf(&name, "action_%lld_easing", i);
		action->easing = ExponentialEaseOut((float)obs_data_get_double(settings, name.array) / 100.0f);

		// Pre-calculate initial values
		if (action->action != ACTION_ATTACH_SOURCE) {
			mp_move_action_get_float(filter, action, false, &action->previous_float);
			mp_move_action_get_vec2(filter, action, false, &action->previous_vec2);
		}
	}
	dstr_free(&name);

	// Clear last_error on settings update (mirror NVIDIA)
	bfree(filter->last_error);
	filter->last_error = NULL;

	// --- Feature-flag-driven handle lifecycle ---
	bool needs_face = (feature_flags & ((1ull << FEATURE_BOUNDINGBOX) |
					     (1ull << FEATURE_LANDMARK) |
					     (1ull << FEATURE_POSE) |
					     (1ull << FEATURE_EXPRESSION) |
					     (1ull << FEATURE_GAZE))) != 0;

	bool needs_body = (feature_flags & (1ull << FEATURE_BODY)) != 0;

	mp_api_t *api = mediapipe_get_api();

	// --- Face graph lifecycle ---
	if (needs_face) {
		if (!filter->face_graph) {
			// Only check model if we need face features
			if (!mp_check_model("face_landmarker.task")) {
				blog(LOG_ERROR, "[MediaPipe Move] face_landmarker.task missing, face features disabled");
			} else {
				char *model_path = obs_module_file("face_landmarker.task");
				mp_face_landmarker_options_t opts;
				memset(&opts, 0, sizeof(opts));
				opts.num_faces = 1;
				opts.min_detection_confidence = filter->face_detection_confidence;
				opts.min_presence_confidence = filter->face_presence_confidence;
				opts.min_tracking_confidence = filter->face_tracking_confidence;
				// Enable blendshapes if any action uses expressions
				opts.output_blendshapes = (feature_flags & (1ull << FEATURE_EXPRESSION)) != 0;
				// Enable transform matrix if any action uses pose or bounding box
				opts.output_transform_matrix = (feature_flags & ((1ull << FEATURE_POSE) |
										  (1ull << FEATURE_BOUNDINGBOX))) != 0;
				// No result_callback for now — using synchronous detection
				opts.result_callback = NULL;
				opts.callback_userdata = NULL;

				if (api->create_face_landmarker(&opts, model_path,
								(mp_face_landmarker_t **)&filter->face_graph)) {
					blog(LOG_INFO, "[MediaPipe Move] Created face landmarker (blendshapes=%d, transform=%d)",
					     opts.output_blendshapes, opts.output_transform_matrix);
				} else {
					mp_move_log_error(filter, false, "create_face_landmarker");
				}
				bfree(model_path);
			}
		} else {
			// Handle already exists — could recreate if thresholds changed
			// (full recreate logic can be added later; for now thresholds are read at creation time only)
		}
	} else if (filter->face_graph) {
		// No face features needed but graph exists — destroy it
		api->destroy_face_landmarker((mp_face_landmarker_t *)filter->face_graph);
		filter->face_graph = NULL;
		blog(LOG_DEBUG, "[MediaPipe Move] Destroyed face landmarker (no longer needed)");
	}

	// --- Pose (body) graph lifecycle ---
	if (needs_body) {
		if (!filter->pose_graph) {
			if (!mp_check_model("pose_landmarker.task")) {
				blog(LOG_ERROR, "[MediaPipe Move] pose_landmarker.task missing, body features disabled");
			} else {
				char *model_path = obs_module_file("pose_landmarker.task");
				if (api->create_pose_landmarker(model_path,
								(mp_pose_landmarker_t **)&filter->pose_graph)) {
					blog(LOG_INFO, "[MediaPipe Move] Created pose landmarker");
				} else {
					mp_move_log_error(filter, false, "create_pose_landmarker");
				}
				bfree(model_path);
			}
		}
	} else if (filter->pose_graph) {
		api->destroy_pose_landmarker((mp_pose_landmarker_t *)filter->pose_graph);
		filter->pose_graph = NULL;
		blog(LOG_DEBUG, "[MediaPipe Move] Destroyed pose landmarker (no longer needed)");
	}

	// --- Resize output arrays based on which features are active ---
	// Landmark arrays sized by their MediaPipe output — set here for consistency
	if (feature_flags & ((1ull << FEATURE_LANDMARK) | (1ull << FEATURE_BOUNDINGBOX) |
			     (1ull << FEATURE_POSE) | (1ull << FEATURE_GAZE))) {
		if (!filter->landmarks_confidence.num)
			da_resize(filter->landmarks_confidence, 478);
		if (!filter->landmarks.num)
			da_resize(filter->landmarks, 478);
	}
	if (feature_flags & (1ull << FEATURE_BOUNDINGBOX)) {
		if (!filter->bboxes_confidence.num)
			da_resize(filter->bboxes_confidence, BBOXES_COUNT);
	}
	if (feature_flags & (1ull << FEATURE_EXPRESSION)) {
		if (!filter->expressions.num)
			da_resize(filter->expressions, 52); // MediaPipe blendshape count
	}
	if (feature_flags & (1ull << FEATURE_BODY)) {
		if (!filter->keypoints_confidence.num)
			da_resize(filter->keypoints_confidence, 33); // COCO 17 + extra
		if (!filter->keypoints.num)
			da_resize(filter->keypoints, 33);
		if (!filter->keypoints3D.num)
			da_resize(filter->keypoints3D, 33);
		if (!filter->joint_angles.num)
			da_resize(filter->joint_angles, 33);
	}
}

struct obs_source_info mediapipe_move_filter = {
	.id = "mp_move_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
	.get_name = mp_move_name,
	.create = mp_move_create,
	.destroy = mp_move_destroy,
	.get_defaults = mp_move_defaults,
	.get_properties = mp_move_properties,
	.update = mp_move_update,
	.video_tick = mp_move_tick,
	.video_render = mp_move_render,
	.filter_video = NULL,
	.load = mp_move_update,
	.video_get_color_space = mp_move_get_color_space,
};

void mp_move_register(void)
{
	obs_register_source(&mediapipe_move_filter);
}
