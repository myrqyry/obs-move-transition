#include "mediapipe-load.h"
#include <obs-module.h>
#include <util/platform.h>
#include <string.h>

bool mediapipe_loaded = false;
mp_api_t mediapipe_api = {0};

static void *mp_handle = NULL;

bool mediapipe_load_library(void)
{
	if (mediapipe_loaded)
		return true;

	const char *lib_path = "libmediapipe_tasks.so"; // Or platform specific path
	mp_handle = os_dlopen(lib_path);
	if (!mp_handle) {
		blog(LOG_ERROR, "[MediaPipe] Failed to load library %s", lib_path);
		return false;
	}

	// Load API function pointers
#define LOAD_FUNC(name) \
	mediapipe_api.name = (void *)os_dlsym(mp_handle, #name); \
	if (!mediapipe_api.name) { \
		blog(LOG_ERROR, "[MediaPipe] Failed to load function %s", #name); \
		return false; \
	}

	LOAD_FUNC(create_face_landmarker);
	LOAD_FUNC(detect_face);
	LOAD_FUNC(destroy_face_landmarker);
	LOAD_FUNC(create_pose_landmarker);
	LOAD_FUNC(detect_pose);
	LOAD_FUNC(destroy_pose_landmarker);
	LOAD_FUNC(get_version);
	LOAD_FUNC(get_last_error);

	mediapipe_loaded = true;
	blog(LOG_INFO, "[MediaPipe] Loaded version %s", mediapipe_api.get_version());
	return true;
}

void mediapipe_unload_library(void)
{
	if (mp_handle) {
		os_dlclose(mp_handle);
		mp_handle = NULL;
	}
	memset(&mediapipe_api, 0, sizeof(mediapipe_api));
	mediapipe_loaded = false;
}

bool mediapipe_is_loaded(void)
{
	return mediapipe_loaded;
}

mp_api_t *mediapipe_get_api(void)
{
	return &mediapipe_api;
}
