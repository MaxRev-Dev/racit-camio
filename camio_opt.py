import os

# Configure pyglet for headless execution and larger ALSA buffer before importing it.
os.environ.setdefault("PYGLET_SHADOW_WINDOW", "0")
if not os.getenv("DISPLAY"):
    os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("PYGLET_ALSA_BUFFER_SIZE", "4096")

import sys
import time
import json
import argparse
from collections import deque

import cv2 as cv
import numpy as np
import pyglet

pyglet.options.setdefault("audio", ("pulse", "openal", "alsa", "silent"))

import pyglet.media
from simple_camio_2d import InteractionPolicy2D, CamIOPlayer2D
from simple_camio_mp import PoseDetectorMP, SIFTModelDetectorMP



class MovementFilter:
    def __init__(self):
        self.prev_position = None
        self.BETA = 0.5

    def push_position(self, position):
        if self.prev_position is None:
            self.prev_position = position
        else:
            self.prev_position = self.prev_position*(1-self.BETA) + position*self.BETA
        return self.prev_position


class MovementMedianFilter:
    def __init__(self):
        self.MAX_QUEUE_LENGTH = 30
        self.positions = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.times = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.AVERAGING_TIME = .7

    def push_position(self, position):
        self.positions.append(position)
        now = time.time()
        self.times.append(now)
        i = len(self.times)-1
        Xs = []
        Ys = []
        Zs = []
        while i >= 0 and now - self.times[i] < self.AVERAGING_TIME:
            Xs.append(self.positions[i][0])
            Ys.append(self.positions[i][1])
            Zs.append(self.positions[i][2])
            i -= 1
        return np.array([np.median(Xs), np.median(Ys), np.median(Zs)])

class GestureDetector:
    def __init__(self):
        self.MAX_QUEUE_LENGTH = 30
        self.positions = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.times = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.DWELL_TIME_THRESH = .75
        self.X_MVMNT_THRESH = 0.95
        self.Y_MVMNT_THRESH = 0.95
        self.Z_MVMNT_THRESH = 4.0

    def push_position(self, position):
        self.positions.append(position)
        now = time.time()
        self.times.append(now)
        i = len(self.times)-1
        Xs = []
        Ys = []
        Zs = []
        while (i >= 0 and now - self.times[i] < self.DWELL_TIME_THRESH):
            Xs.append(self.positions[i][0])
            Ys.append(self.positions[i][1])
            Zs.append(self.positions[i][2])
            i -= 1
        Xdiff = max(Xs) - min(Xs)
        Ydiff = max(Ys) - min(Ys)
        Zdiff = max(Zs) - min(Zs)
        # print("(i: " + str(i) + ") X: " + str(Xdiff) + ", Y: " + str(Ydiff) + ", Z: " + str(Zdiff))
        if Xdiff < self.X_MVMNT_THRESH and Ydiff < self.Y_MVMNT_THRESH and Zdiff < self.Z_MVMNT_THRESH:
            return np.array([sum(Xs)/float(len(Xs)), sum(Ys)/float(len(Ys)), sum(Zs)/float(len(Zs))]), 'still'
        else:
            return position, 'moving'


class AmbientSoundPlayer:
    def __init__(self, soundfile):
        self.sound = pyglet.media.load(soundfile, streaming=False)
        self.player = pyglet.media.Player()
        self.player.queue(self.sound)
        self.player.eos_action = 'loop'
        self.player.loop = True

    def set_volume(self, volume):
        if 0 <= volume <= 1:
            self.player.volume = volume

    def play_sound(self):
        if not self.player.playing:
            self.player.play()

    def pause_sound(self):
        if self.player.playing:
            self.player.pause()


def draw_rect_in_image(image, sz, H):
    img_corners = np.array([[0,0],[sz[1],0],[sz[1],sz[0]],[0,sz[0]]], dtype=np.float32)
    img_corners = np.reshape(img_corners, [-1, 1, 2])
    H_inv = np.linalg.inv(H)
    pts = cv.perspectiveTransform(img_corners, H_inv)
    for pt in pts:
        image = cv.circle(image, (int(pt[0][0]), int(pt[0][1])), 3, (0, 255, 0), -1)
    return image


def throttle_loop(start_time, target_frame_time):
    if target_frame_time <= 0.0:
        return
    elapsed_time = time.time() - start_time
    remaining_time = target_frame_time - elapsed_time
    if remaining_time > 0:
        time.sleep(remaining_time)


class CommandInterface:
    KEY_MAPPINGS = {
        ord('q'): ("quit", []),
        27: ("quit", []),  # ESC
        ord('h'): ("update_homography", []),
        ord('b'): ("toggle_blips", []),
    }

    def __init__(self, headless, verbose, command_file):
        self.headless = headless
        self.verbose = verbose
        self.command_file = command_file if command_file else None
        if self.verbose and self.command_file:
            print(f"[CamIO] Command file monitoring enabled at {self.command_file}")

    def poll(self, waitkey):
        commands = []
        if waitkey is not None and waitkey != -1:
            mapped = self.KEY_MAPPINGS.get(waitkey)
            if mapped:
                commands.append(mapped)
        commands.extend(self._poll_command_file())
        return commands

    def _poll_command_file(self):
        if not self.command_file or not os.path.exists(self.command_file):
            return []
        try:
            with open(self.command_file, 'r', encoding='utf-8') as cmd_file:
                raw = cmd_file.read()
        except OSError:
            return []
        try:
            os.remove(self.command_file)
        except OSError:
            pass
        commands = []
        for line in raw.splitlines():
            entry = line.strip()
            if not entry:
                continue
            command, args = self._parse_entry(entry)
            if command:
                commands.append((command, args))
                if self.verbose:
                    print(f"[CamIO] Command received: {command} {args}")
        return commands

    @staticmethod
    def _parse_entry(entry):
        try:
            data = json.loads(entry)
        except json.JSONDecodeError:
            data = entry
        if isinstance(data, dict):
            command = data.get("command")
            args = data.get("args", [])
            if isinstance(args, str):
                args = [args]
            elif not isinstance(args, list):
                args = [str(args)]
        elif isinstance(data, str):
            tokens = data.split()
            if not tokens:
                return None, []
            command = tokens[0]
            args = tokens[1:]
        else:
            return None, []
        return command.lower(), args


def select_cam_port():
    available_ports, working_ports, non_working_ports = list_ports()
    if len(working_ports) == 1:
        return working_ports[0][0]
    elif len(working_ports) > 1:
        print("The following cameras were detected:")
        for i in range(len(working_ports)):
            print(f'{i}) Port {working_ports[i][0]}: {working_ports[i][1]} x {working_ports[i][2]}')
        cam_selection = input("Please select which camera you would like to use: ")
        return working_ports[int(cam_selection)][0]
    else:
        return 0

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 3:  # if there are more than 2 non working ports stop the testing.
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append((dev_port, h, w))
            else:
                print("Port %s for camera ( %s x %s) is present but does not read." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


# Function to load map parameters from a JSON file
def load_map_parameters(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            map_params = json.load(f)
            print("loaded map parameters from file.")
    else:
        print("No map parameters file found at " + filename)
        print("Usage: simple_camio.exe --input1 <filename>")
        print(" ")
        print("Press any key to exit.")
        _ = sys.stdin.read(1)
        exit(0)
    return map_params['model']


parser = argparse.ArgumentParser(description='Code for CamIO.')
parser.add_argument('--input1', help='Path to parameter json file.', default='models/UkraineMap/UkraineMap.json')
parser.add_argument('--headless', action='store_true',
                    help='Disable OpenCV UI windows; automatically enabled if DISPLAY is not set.')
parser.add_argument('--target-fps', type=float, default=15.0,
                    help='Desired processing FPS to throttle CPU usage on constrained devices.')
parser.add_argument('--frame-skip', type=int, default=0,
                    help='Number of frames to skip between full processing passes for performance.')
parser.add_argument('--verbose', action='store_true', help='Print verbose diagnostics to stdout.')
parser.add_argument('--pose-input-width', type=int, default=0,
                    help='Downscale frames before MediaPipe hand inference (0 keeps original width).')
parser.add_argument('--pose-max-hands', type=int, default=1,
                    help='Maximum hands to track; lowering reduces CPU load.')
parser.add_argument('--draw-pose', choices=['auto', 'on', 'off'], default='auto',
                    help='Draw pose overlay (auto = only when not headless).')
parser.add_argument('--command-file', default='/tmp/camio_command',
                    help='Optional path to a command file polled each frame; set empty string to disable.')
args = parser.parse_args()

# Load map and camera parameters
model = load_map_parameters(args.input1)

# Determine UI availability once to avoid repeated getenv calls.
headless = args.headless or not bool(os.getenv("DISPLAY"))
# Ensure pyglet knows if a display is available.
pyglet.options["headless"] = headless
if args.draw_pose == 'auto':
    draw_pose_overlay = not headless
else:
    draw_pose_overlay = args.draw_pose == 'on'

if args.verbose:
    print(f"[CamIO] Headless mode: {headless}")
    print(f"[CamIO] Pose overlay drawing: {draw_pose_overlay}")
    print(f"[CamIO] Target FPS: {args.target_fps}, frame skip: {args.frame_skip}")
    if args.command_file:
        print(f"[CamIO] External command file: {args.command_file}")
    print("[CamIO] Commands: h/update_homography, b/toggle_blips, q/quit, set-volume <heartbeat|ambient> <0-1>")

# ========================================
cam_port = select_cam_port()
# ========================================

if args.verbose:
    print(f"[CamIO] Selected camera port: {cam_port}")

# Initialize objects
if model["modelType"] == "sift_2d_mediapipe":
    model_detector = SIFTModelDetectorMP(model)
    pose_detector = PoseDetectorMP(
        model,
        max_hands=max(1, args.pose_max_hands),
        input_width=max(0, args.pose_input_width),
        verbose=args.verbose,
        draw_landmarks=draw_pose_overlay
    )
    gesture_detector = GestureDetector()
    motion_filter = MovementMedianFilter()
    interact = InteractionPolicy2D(model)
    camio_player = CamIOPlayer2D(model)
    camio_player.play_welcome()
    crickets_player = AmbientSoundPlayer(model['crickets'])
    heartbeat_player = AmbientSoundPlayer(model['heartbeat'])
else:
    raise ValueError(f"Unsupported modelType '{model['modelType']}' in configuration.")

heartbeat_player.set_volume(.05)
command_interface = CommandInterface(headless=headless, verbose=args.verbose, command_file=args.command_file)
cap = cv.VideoCapture(cam_port)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)  # set camera image height
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)  # set camera image width
cap.set(cv.CAP_PROP_FOCUS, 0)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

cv.setUseOptimized(True)
try:
    cv.setNumThreads(2)
except cv.error:
    pass

loop_has_run = False
print("Press \"h\" key to update map position in image.")

target_frame_time = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
frame_skip = max(0, args.frame_skip)
frame_counter = 0
marker_detected = False
prev_zone_id = None
prev_audio_zone = None
heartbeat_playing = False
crickets_playing = False
should_shutdown = False
# Main loop
while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("No camera image returned.")
            break

        loop_start = time.time()
        frame_counter += 1
        waitkey = -1
        if loop_has_run and not headless:
            cv.imshow('image reprojection', img_scene_color)
            try:
                waitkey = cv.waitKeyEx(1)
            except AttributeError:
                waitkey = cv.waitKey(1)
            if waitkey != -1:
                waitkey &= 0xFF
        elif loop_has_run:
            # Yield a little time back to the OS when we are headless.
            time.sleep(0.001)

        commands = command_interface.poll(waitkey)
        for command, cmd_args in commands:
            if command in ("quit", "exit", "shutdown"):
                print("Shutdown requested via command interface.")
                should_shutdown = True
                break
            if command in ("update_homography", "update_h"):
                model_detector.requires_homography = True
                if args.verbose:
                    print("[CamIO] Homography refresh requested.")
                continue
            if command in ("toggle_blips", "blips"):
                camio_player.enable_blips = not camio_player.enable_blips
                state = "enabled" if camio_player.enable_blips else "disabled"
                print(f"Blips have been {state}.")
                continue
            if command in ("set-volume", "set_volume") and cmd_args:
                target = cmd_args[0].lower()
                try:
                    value = float(cmd_args[1]) if len(cmd_args) > 1 else None
                except (ValueError, TypeError):
                    value = None
                if value is None:
                    if args.verbose:
                        print(f"[CamIO] Invalid volume command args: {cmd_args}")
                    continue
                value = max(0.0, min(1.0, value))
                if target in ("heartbeat", "hb"):
                    heartbeat_player.set_volume(value)
                    if args.verbose:
                        print(f"[CamIO] Heartbeat volume set to {value:.2f}")
                elif target in ("ambient", "crickets"):
                    crickets_player.set_volume(value)
                    if args.verbose:
                        print(f"[CamIO] Ambient volume set to {value:.2f}")
                else:
                    if args.verbose:
                        print(f"[CamIO] Unknown volume target '{target}'")
                continue
            if args.verbose:
                print(f"[CamIO] Unrecognized command '{command}' with args {cmd_args}")
        if should_shutdown:
            break

        if os.path.exists("/tmp/update_map.flag"):
            model_detector.requires_homography = True
            os.remove("/tmp/update_map.flag")

        pyglet.clock.tick()
        pyglet.app.platform_event_loop.dispatch_posted_events()

        process_frame = frame_skip == 0 or (frame_counter % (frame_skip + 1) == 1)
        if not process_frame:
            throttle_loop(loop_start, target_frame_time)
            continue

        img_scene_color = frame.copy()
        loop_has_run = True

        # load images grayscale
        img_scene_gray = cv.cvtColor(img_scene_color, cv.COLOR_BGR2GRAY)
        # Detect aruco markers for map in image
        retval, H, tvec = model_detector.detect(img_scene_gray)

        # If no  markers found, continue to next iteration
        if not retval:
            if marker_detected and args.verbose:
                print("[CamIO] Lost map alignment; switching to ambient audio.")
            marker_detected = False
            if heartbeat_playing:
                heartbeat_player.pause_sound()
                heartbeat_playing = False
                if args.verbose:
                    print("[Audio] Heartbeat paused.")
            if not crickets_playing:
                crickets_player.play_sound()
                crickets_playing = True
                if args.verbose:
                    print("[Audio] Ambient loop started.")
            throttle_loop(loop_start, target_frame_time)
            continue

        if not marker_detected and args.verbose:
            print("[CamIO] Map alignment established.")
        marker_detected = True
        if crickets_playing:
            crickets_player.pause_sound()
            crickets_playing = False
            if args.verbose:
                print("[Audio] Ambient loop paused.")

        camio_player.play_description()
        crickets_player.pause_sound()
        crickets_playing = False

        gesture_loc, gesture_status, img_scene_color = pose_detector.detect(frame, H, tvec)
        if gesture_loc is None:
            if heartbeat_playing:
                heartbeat_player.pause_sound()
                heartbeat_playing = False
                if args.verbose:
                    print("[Audio] Heartbeat paused (no gesture).")
            img_scene_color = draw_rect_in_image(img_scene_color, interact.image_map_color.shape, H)
            throttle_loop(loop_start, target_frame_time)
            continue
        gesture_loc = gesture_loc / model["pixels_per_cm"]
        if not heartbeat_playing:
            heartbeat_player.play_sound()
            heartbeat_playing = True
            if args.verbose:
                print("[Audio] Heartbeat loop playing.")

        # Determine zone from point of interest
        zone_id = interact.push_gesture(gesture_loc)

        zone_value = int(zone_id) if isinstance(zone_id, np.integer) else zone_id
        if args.verbose and zone_value != prev_zone_id:
            print(f"[CamIO] Active zone: {zone_value}")
        prev_zone_id = zone_value

        if zone_value in camio_player.hotspots and gesture_status != "moving":
            hotspot_desc = camio_player.hotspots[zone_value].get('textDescription', str(zone_value))
            if zone_value != prev_audio_zone and args.verbose:
                print(f"[Audio] Playing hotspot audio: {hotspot_desc}")
            prev_audio_zone = zone_value
        elif zone_value == -1 and prev_audio_zone is not None:
            prev_audio_zone = None
        elif zone_value not in camio_player.hotspots and prev_audio_zone is not None:
            prev_audio_zone = None

        # If the zone id is valid, play the sound for the zone
        camio_player.convey(zone_id, gesture_status)

        # Draw points in image
        img_scene_color = draw_rect_in_image(img_scene_color, interact.image_map_color.shape, H)

        throttle_loop(loop_start, target_frame_time)

    except cv.error:
        pass
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        break

cap.release()
if not headless:
    cv.destroyAllWindows()

camio_player.play_goodbye()
heartbeat_player.pause_sound()
crickets_player.pause_sound()
heartbeat_playing = False
crickets_playing = False
time.sleep(1)
