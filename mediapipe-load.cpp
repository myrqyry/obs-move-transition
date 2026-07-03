#include "mediapipe-load.h"
#include <obs-module.h>
#include <util/platform.h>
#include <string.h>
#include <type_traits>
#include <dlfcn.h>

bool mediapipe_loaded = false;
mp_api_t mediapipe_api = {0};

static void *mp_handle = NULL;

extern "C" bool mediapipe_load_library(void)
{
	if (mediapipe_loaded)
		return true;

	// Use dladdr to find the plugin .so's own path, then derive the
	// directory containing libmediapipe.so (same directory).
	Dl_info info;
	memset(&info, 0, sizeof(info));
	if (!dladdr((void *)mediapipe_load_library, &info) || !info.dli_fname) {
		blog(LOG_ERROR, "[MediaPipe] Could not determine plugin module path via dladdr");
		return false;
	}

	// Copy the .so path and strip the filename to get the directory
	char *dir_path = bstrdup(info.dli_fname);
	char *last_slash = strrchr(dir_path, '/');
	if (!last_slash) {
		bfree(dir_path);
		blog(LOG_ERROR, "[MediaPipe] Could not parse plugin module path: %s", info.dli_fname);
		return false;
	}
	*(last_slash + 1) = '\0'; // keep the trailing /

	// Build libmediapipe.so path as <plugin_dir>/libmediapipe.so
	char *lib_path = (char *)bmalloc(strlen(dir_path) + 32);
	snprintf(lib_path, strlen(dir_path) + 32, "%slibmediapipe.so", dir_path);
	bfree(dir_path);

	blog(LOG_INFO, "[MediaPipe] Attempting to load libmediapipe.so from: %s", lib_path);

	mp_handle = os_dlopen(lib_path);
	if (!mp_handle) {
		// Fallback: try just "libmediapipe.so" in LD_LIBRARY_PATH
		blog(LOG_WARNING, "[MediaPipe] Failed to load from %s, trying system path", lib_path);
		bfree(lib_path);
		mp_handle = os_dlopen("libmediapipe.so");
		if (!mp_handle) {
			blog(LOG_ERROR, "[MediaPipe] Failed to load libmediapipe.so from system path either");
			return false;
		}
	} else {
		bfree(lib_path);
	}


#define LOAD_FUNC(ptr_name, sym_name) \
	do { \
		void *__sym = os_dlsym(mp_handle, sym_name); \
		memcpy(&mediapipe_api.ptr_name, &__sym, sizeof(void *)); \
		if (!mediapipe_api.ptr_name) { \
			blog(LOG_ERROR, "[MediaPipe] Failed to load symbol %s", sym_name); \
			os_dlclose(mp_handle); \
			mp_handle = NULL; \
			memset(&mediapipe_api, 0, sizeof(mediapipe_api)); \
			return false; \
		} \
	} while (0)

	// Load MpImage functions
	LOAD_FUNC(image_create_from_uint8_data, "MpImageCreateFromUint8Data");
	LOAD_FUNC(image_free, "MpImageFree");

	// Error handling
	LOAD_FUNC(error_free, "MpErrorFree");

	// Face Landmarker
	LOAD_FUNC(face_landmarker_create, "MpFaceLandmarkerCreate");
	LOAD_FUNC(face_landmarker_detect_image, "MpFaceLandmarkerDetectImage");
	LOAD_FUNC(face_landmarker_close_result, "MpFaceLandmarkerCloseResult");
	LOAD_FUNC(face_landmarker_close, "MpFaceLandmarkerClose");

	// Pose Landmarker
	LOAD_FUNC(pose_landmarker_create, "MpPoseLandmarkerCreate");
	LOAD_FUNC(pose_landmarker_detect_image, "MpPoseLandmarkerDetectImage");
	LOAD_FUNC(pose_landmarker_close_result, "MpPoseLandmarkerCloseResult");
	LOAD_FUNC(pose_landmarker_close, "MpPoseLandmarkerClose");

	mediapipe_loaded = true;
	blog(LOG_INFO, "[MediaPipe] Loaded libmediapipe.so successfully");
	return true;
}

extern "C" void mediapipe_unload_library(void)
{
	if (mp_handle) {
		os_dlclose(mp_handle);
		mp_handle = NULL;
	}
	memset(&mediapipe_api, 0, sizeof(mediapipe_api));
	mediapipe_loaded = false;
}

extern "C" bool mediapipe_is_loaded(void)
{
	return mediapipe_loaded;
}

extern "C" mp_api_t *mediapipe_get_api(void)
{
	return &mediapipe_api;
}
