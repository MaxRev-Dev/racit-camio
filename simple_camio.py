import os
import sys
import logging
import subprocess

# Set volume to 80% for the PCM channel
subprocess.run(["amixer", "sset", "PCM", "80%"])

# Set up comprehensive logging for headless operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        #logging.FileHandler('/home/user/camio/camio_debug.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== CamIO Application Starting ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Environment DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")

# Set environment variables for headless operation before importing OpenCV
# Only remove DISPLAY if we're not already running with a virtual display
if 'DISPLAY' not in os.environ:
    logger.info("No DISPLAY environment variable found")
    # We'll let the run script handle DISPLAY
    pass
else:
    logger.info(f"DISPLAY environment variable found: {os.environ['DISPLAY']}")

# Disable Qt logging to reduce noise
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
# Disable Qt accessibility
os.environ['QT_ACCESSIBILITY'] = '0'
# Force OpenCV to use headless backend
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

logger.info("Environment variables set for headless operation")

logger.info("Importing required modules...")
import cv2 as cv
import time
import numpy as np
import json
import argparse
from collections import deque
logger.info("Basic modules imported successfully")

try:
    from simple_camio_2d import InteractionPolicy2D, CamIOPlayer2D
    logger.info("CamIO 2D modules imported successfully")
except Exception as e:
    logger.error(f"Failed to import CamIO 2D modules: {e}")
    raise

try:
    from simple_camio_mp import PoseDetectorMP, SIFTModelDetectorMP
    logger.info("CamIO MediaPipe modules imported successfully")
except Exception as e:
    logger.error(f"Failed to import CamIO MediaPipe modules: {e}")
    raise

# Check if we have a display available for GUI
DISPLAY_AVAILABLE = False
# Force headless mode to avoid Qt platform issues completely
logger.info("Forcing headless mode to avoid Qt platform issues")
print("Forcing headless mode to avoid Qt platform issues")

# Try to import pygame for audio with proper error handling
logger.info("Attempting to initialize pygame audio...")
try:
    import pygame
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
    logger.info("Audio enabled - pygame available")
    print("Audio enabled - pygame available")
except Exception as e:
    logger.warning(f"pygame not available: {e}")
    print(f"Warning: pygame not available - {e}")
    print("Audio features will be disabled")
    PYGAME_AVAILABLE = False

# Keep pyglet variables for compatibility
PYGLET_AVAILABLE = PYGAME_AVAILABLE
logger.info(f"Audio system initialized: PYGAME_AVAILABLE={PYGAME_AVAILABLE}")



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
        print("(i: " + str(i) + ") X: " + str(Xdiff) + ", Y: " + str(Ydiff) + ", Z: " + str(Zdiff))
        if Xdiff < self.X_MVMNT_THRESH and Ydiff < self.Y_MVMNT_THRESH and Zdiff < self.Z_MVMNT_THRESH:
            return np.array([sum(Xs)/float(len(Xs)), sum(Ys)/float(len(Ys)), sum(Zs)/float(len(Zs))]), 'still'
        else:
            return position, 'moving'


class AmbientSoundPlayer:
    def __init__(self, soundfile):
        logger.debug(f"Initializing AmbientSoundPlayer with {soundfile}")
        if not PYGAME_AVAILABLE:
            logger.warning(f"Audio disabled - pygame not available for {soundfile}")
            print(f"Audio disabled - pygame not available for {soundfile}")
            self.sound = None
            self.channel = None
            return
        try:
            self.sound = pygame.mixer.Sound(soundfile)
            self.channel = None
            self.volume = 1.0
            logger.info(f"Audio loaded successfully: {soundfile}")
            print(f"Audio loaded successfully: {soundfile}")
        except Exception as e:
            logger.error(f"Audio initialization failed for {soundfile}: {e}")
            print(f"Audio initialization failed for {soundfile}: {e}")
            self.sound = None
            self.channel = None

    def set_volume(self, volume):
        logger.debug(f"Setting volume to {volume}")
        if self.sound and 0 <= volume <= 1:
            self.volume = volume
            self.sound.set_volume(volume)

    def play_sound(self):
        logger.debug("Playing sound...")
        if self.sound and (not self.channel or not self.channel.get_busy()):
            self.channel = self.sound.play(loops=-1)  # Loop indefinitely
            if self.channel:
                self.channel.set_volume(self.volume)
                logger.debug("Sound started playing")

    def pause_sound(self):
        logger.debug("Pausing sound...")
        if self.channel and self.channel.get_busy():
            self.channel.stop()
            logger.debug("Sound stopped")


def draw_rect_in_image(image, sz, H):
    img_corners = np.array([[0,0],[sz[1],0],[sz[1],sz[0]],[0,sz[0]]], dtype=np.float32)
    img_corners = np.reshape(img_corners, [-1, 1, 2])
    H_inv = np.linalg.inv(H)
    pts = cv.perspectiveTransform(img_corners, H_inv)
    for pt in pts:
        image = cv.circle(image, (int(pt[0][0]), int(pt[0][1])), 3, (0, 255, 0), -1)
    return image


def select_cam_port():
    logger.info("Starting camera port selection...")
    available_ports, working_ports, non_working_ports = list_ports()
    logger.info(f"Camera scan results: {len(working_ports)} working, {len(available_ports)} available, {len(non_working_ports)} non-working")
    
    if len(working_ports) == 1:
        logger.info(f"Single camera found on port {working_ports[0][0]}")
        return working_ports[0][0]
    elif len(working_ports) > 1:
        print("The following cameras were detected:")
        for i in range(len(working_ports)):
            print(f'{i}) Port {working_ports[i][0]}: {working_ports[i][1]} x {working_ports[i][2]}')
        cam_selection = input("Please select which camera you would like to use: ")
        selected_port = working_ports[int(cam_selection)][0]
        logger.info(f"User selected camera port: {selected_port}")
        return selected_port
    else:
        logger.warning("No working cameras found, defaulting to port 0")
        return 0

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    logger.debug("Scanning for camera ports...")
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 1:  # if there are more than 1 non working ports stop the testing.
        logger.debug(f"Testing camera port {dev_port}")
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            logger.debug(f"Port {dev_port} is not working")
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                logger.info(f"Port {dev_port} working: {h} x {w}")
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append((dev_port, h, w))
            else:
                logger.warning(f"Port {dev_port} present but cannot read: {h} x {w}")
                print("Port %s for camera ( %s x %s) is present but does not read." % (dev_port, h, w))
                available_ports.append(dev_port)
        camera.release()
        dev_port += 1
    logger.info(f"Camera scan complete: found {len(working_ports)} working cameras")
    return available_ports, working_ports, non_working_ports


# Function to load map parameters from a JSON file
def load_map_parameters(filename):
    logger.info(f"Loading map parameters from: {filename}")
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            map_params = json.load(f)
            logger.info("Map parameters loaded successfully from file")
            print("loaded map parameters from file.")
    else:
        logger.error(f"No map parameters file found at {filename}")
        print("No map parameters file found at " + filename)
        print("Usage: simple_camio.exe --input <filename>")
        print(" ")
        print("Press any key to exit.")
        _ = sys.stdin.read(1)
        exit(0)
    return map_params['model']


parser = argparse.ArgumentParser(description='Code for CamIO.')
parser.add_argument('--input', help='Path to parameter json file.', default='camio/models/RACITFirstFloor/CollegeFirstFloor.json')
args = parser.parse_args()
logger.info(f"Command line arguments: {args}")

# Load map and camera parameters
logger.info("Loading map and camera parameters...")
model = load_map_parameters(args.input)
logger.info(f"Model type: {model['modelType']}")

# ========================================
logger.info("Selecting camera port...")
cam_port = select_cam_port()
logger.info(f"Selected camera port: {cam_port}")
# ========================================

# Initialize objects
logger.info("Initializing CamIO objects...")
if model["modelType"] == "sift_2d_mediapipe":
    logger.info("Creating SIFT 2D MediaPipe model detector...")
    model_detector = SIFTModelDetectorMP(model)
    logger.info("Creating MediaPipe pose detector...")
    pose_detector = PoseDetectorMP(model)
    logger.info("Creating gesture detector...")
    gesture_detector = GestureDetector()
    logger.info("Creating motion filter...")
    motion_filter = MovementMedianFilter()
    logger.info("Creating interaction policy...")
    interact = InteractionPolicy2D(model)
    logger.info("Creating CamIO player...")
    camio_player = CamIOPlayer2D(model)
    logger.info("Playing welcome message...")
    camio_player.play_welcome()
    logger.info("Creating ambient sound players...")
    crickets_player = AmbientSoundPlayer(model['crickets'])
    heartbeat_player = AmbientSoundPlayer(model['heartbeat'])
else:
    logger.error(f"Unknown model type: {model['modelType']}")
    raise ValueError(f"Unknown model type: {model['modelType']}")

logger.info("Setting heartbeat volume...")
heartbeat_player.set_volume(.05)

logger.info("Initializing camera capture...")
cap = cv.VideoCapture(cam_port)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)  # set camera image height
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)  # set camera image width
cap.set(cv.CAP_PROP_FOCUS, 0)

logger.info("Camera settings applied")
logger.info(f"Camera resolution: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT)}")

loop_has_run = False
timer = time.time()
frame_count = 0
logger.info("Starting main processing loop...")
print("Press \"h\" key to update map position in image.")
# Main loop

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret:
        logger.error("No camera image returned - breaking loop")
        print("No camera image returned.")
        break
        
    logger.debug(f"Processing frame {frame_count}")
    
    if loop_has_run:
        # Only show GUI if display is available
        if DISPLAY_AVAILABLE:
            cv.imshow('image reprojection', img_scene_color)
            waitkey = cv.waitKey(1)
        else:
            # save image
            # In headless mode, check for exit condition periodically
            waitkey = -1
            if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                logger.info(f"Headless mode - Frame {frame_count} processed at {time.strftime('%H:%M:%S')}")
                print(f"Running in headless mode... Frame {frame_count} processed at {time.strftime('%H:%M:%S')}")
        
        if waitkey == 27 or waitkey == ord('q'):
            logger.info("Exit key pressed - stopping application")
            print('Escape.')
            cap.release()
            if DISPLAY_AVAILABLE:
                cv.destroyAllWindows()
            break
        if waitkey == ord('h'):
            logger.info("Homography update requested")
            model_detector.requires_homography = True
        # with
        if os.path.exists("/tmp/update_map.flag"):
            model_detector.requires_homography = True
            os.remove("/tmp/update_map.flag")

        if waitkey == ord('b'):
            camio_player.enable_blips = not camio_player.enable_blips
            if camio_player.enable_blips:
                logger.info("Blips enabled")
                print("Blips have been enabled.")
            else:
                logger.info("Blips disabled")
                print("Blips have been disabled.")
    
    prev_time = timer
    timer = time.time()
    elapsed_time = timer - prev_time
    
    if frame_count % 100 == 0:
        fps = 1/elapsed_time if elapsed_time > 0 else 0
        logger.debug(f"Current FPS: {fps:.2f}")
    
    # No need for pyglet clock operations since we're using pygame
    img_scene_color = frame.copy()
    loop_has_run = True

    # load images grayscale
    img_scene_gray = cv.cvtColor(img_scene_color, cv.COLOR_BGR2GRAY)
    logger.debug("Converted frame to grayscale")
    
    # Detect aruco markers for map in image
    logger.debug("Detecting ArUco markers...")
    retval, H, tvec = model_detector.detect(img_scene_gray)

    # If no  markers found, continue to next iteration
    if not retval:
        logger.debug("No ArUco markers found")
        heartbeat_player.pause_sound()
        crickets_player.play_sound()
        continue

    logger.debug("ArUco markers detected successfully")
    camio_player.play_description()
    crickets_player.pause_sound()

    logger.debug("Detecting pose and gesture...")
    gesture_loc, gesture_status, img_scene_color = pose_detector.detect(frame, H, tvec)
    
    if gesture_loc is None:
        logger.debug("No gesture location detected")
        heartbeat_player.pause_sound()
        img_scene_color = draw_rect_in_image(img_scene_color, interact.image_map_color.shape, H)
        continue
        
    gesture_loc = gesture_loc / model["pixels_per_cm"]
    logger.debug(f"Gesture location: {gesture_loc}, status: {gesture_status}")
    heartbeat_player.play_sound()

    # Determine zone from point of interest
    logger.debug("Determining interaction zone...")
    zone_id = interact.push_gesture(gesture_loc)
    logger.debug(f"Zone ID: {zone_id}")

    # If the zone id is valid, play the sound for the zone
    logger.debug("Conveying zone information...")
    camio_player.convey(zone_id, gesture_status)

    # Draw points in image
    img_scene_color = draw_rect_in_image(img_scene_color, interact.image_map_color.shape, H)

logger.info("Main loop ended - cleaning up...")
camio_player.play_goodbye()
heartbeat_player.pause_sound()
crickets_player.pause_sound()
logger.info("Application shutdown complete")
time.sleep(1)
