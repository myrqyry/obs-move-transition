# Move Transition for OBS Studio

Move Transition is an OBS Studio plugin by
[Exeldro](https://github.com/exeldro) that animates sources, filters, and
values instead of cutting instantly between states. It started as a scene
transition plugin, but it now includes a larger set of filters for moving
sources, driving settings, reacting to audio, running actions, and connecting
motion to face or camera tracking.

Use it when you want scene changes, source transforms, filter settings, audio
levels, or OBS actions to animate over time.

## What it does

Move Transition compares the outgoing and incoming scenes during a transition.
When matching sources are found, it animates them from their old scene item
transform to their new transform. Sources that only exist in one scene can use
separate appearing and disappearing transitions.

The plugin also adds filters that let you trigger the same kind of movement
without changing scenes. These filters can move a source, swap a source, change
settings, react to audio, run OBS actions, and link movement to tracking data.

## Included tools

The plugin registers several transitions and filters in OBS. Some filters are
available only when their platform or runtime dependency is present.

- `Move Transition` animates matched scene items during a scene transition.
- `Move transition override` lets a source override the transition behavior
  used by `Move Transition`.
- `Move Source` moves a source or scene item on a trigger, hotkey, or event.
- `Move Source Swap` swaps source visibility during a movement.
- `Move Value` changes source settings, filter settings, text values, random
  values, typed values, and numeric values over time.
- `Audio Move` drives movement from audio meter values.
- `Move Action` runs actions such as hotkeys, visibility changes, media
  controls, audio changes, and frontend actions.
- `Audio Move Action` runs actions from audio meter values.
- `NVIDIA AR Move` links source movement to NVIDIA AR tracking data when the
  NVIDIA runtime is available.
- `Move Video Capture Device` controls supported DirectShow camera properties
  on Windows.

## Common use cases

Move Transition can replace several manual scene and filter setups with one
animated workflow.

- Animate a camera source from a small corner layout to a full-screen layout.
- Slide lower thirds, alerts, chat boxes, and overlays in or out of frame.
- Move matching sources between scenes without a visible cut.
- Fade, mute, pause, restart, or trigger media at the start, midpoint, or end of
  a move.
- Change filter settings gradually instead of switching them instantly.
- Drive transforms from audio levels for simple reactive layouts.
- Run OBS hotkeys or frontend actions when a source becomes visible, hidden,
  active, inactive, enabled, disabled, or receives a UDP trigger.
- Attach movement to face, pose, expression, or camera tracking data when the
  relevant tracking filter is available.

## Installation

Download the plugin from the
[Move Transition resource page on the OBS forums](https://obsproject.com/forum/resources/move-transition.913/).

On Flatpak installations, you can install the plugin from a terminal:

```bash
flatpak install com.obsproject.Studio.Plugin.MoveTransition
```

Restart OBS after installation. The plugin adds its transition and filters to
the normal OBS transition and filter menus.

## Building from source

You can build the plugin either inside the OBS Studio source tree or as a
stand-alone plugin on Linux.

### In-tree build

Use this path when you already build OBS Studio from source.

1. Build OBS Studio by following the
   [OBS build instructions](https://obsproject.com/wiki/Install-Instructions).
2. Check out this repository to `plugins/move-transition` inside the OBS source
   tree.
3. Add `add_subdirectory(move-transition)` to `plugins/CMakeLists.txt`.
4. Rebuild OBS Studio.

### Stand-alone build on Linux

Use this path when your system provides OBS development files.

1. Install the package that provides OBS Studio development headers and
   libraries.
2. Configure and build the plugin:

```bash
cmake -S . -B build -DBUILD_OUT_OF_TREE=On
cmake --build build
```

## This fork

This repository tracks
[exeldro/obs-move-transition](https://github.com/exeldro/obs-move-transition)
and keeps Exeldro's plugin as the baseline. The local branch currently adds
experimental work on top of `exeldro/master`.

- Adds `Wave Move`, a filter that drives transforms, settings, source
  visibility, and filter enable states from generated wave values.
- Adds wave shapes such as sine, cosine, square, saw, triangle, pulse, step,
  random hold, and smooth random.
- Adds a MediaPipe-backed move filter that mirrors the NVIDIA AR Move action
  model for face, pose, expression, landmark, bounding box, and transform-based
  actions.
- Loads `libmediapipe.so` at runtime and reports missing MediaPipe model files
  in the OBS log.
- Shares move-action definitions between the NVIDIA and MediaPipe filters.
- Fixes several memory-safety and resource-lifetime issues found while working
  on the new filters.
- Adds regression checks for `Wave Move` and NVIDIA Move settings behavior.

## Credits

Move Transition is created and maintained by Exeldro. Support the original
project through one of these links:

- [GitHub Sponsors](https://github.com/sponsors/exeldro)
- [Ko-fi](https://ko-fi.com/exeldro)
- [Patreon](https://www.patreon.com/Exeldro)
- [PayPal](https://www.paypal.me/exeldro)
