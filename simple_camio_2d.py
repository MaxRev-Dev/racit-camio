import os, os.path
import pygame
import numpy as np
from scipy import stats
import cv2 as cv
os.environ['SDL_AUDIODRIVER']  ='alsa'
# Initialize pygame mixer
try:
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception as e:
    print(f"Warning: pygame not available - {e}")
    PYGAME_AVAILABLE = False


# The InteractionPolicy class takes the position and determines where on the
# map it is, finding the color of the zone, if any, which is decoded into
# zone ID number. This zone ID number is filtered through a ring buffer that
# returns the mode. If the position is near enough to the plane (within 2cm)
# then the zone ID number is returned.
class InteractionPolicy2D:
    def __init__(self, model):
        self.model = model
        self.image_map_color = cv.imread(model['filename'], cv.IMREAD_COLOR)
        self.ZONE_FILTER_SIZE = 10
        self.Z_THRESHOLD = 2.0
        self.zone_filter = -1 * np.ones(self.ZONE_FILTER_SIZE, dtype=int)
        self.zone_filter_cnt = 0
        

    # Sergio: we are currently returning the zone id also when the ring buffer is not full. Is this the desired behavior?
    # the impact is clearly minor, but conceptually I am not convinced that this is the right behavior.
    # Sergio (2): I have a concern about this function, I will discuss it in an email.
    def push_gesture(self, position):
        zone_color = self.get_zone(position, self.image_map_color, self.model['pixels_per_cm'])
        self.zone_filter[self.zone_filter_cnt] = self.get_dict_idx_from_color(zone_color)
        self.zone_filter_cnt = (self.zone_filter_cnt + 1) % self.ZONE_FILTER_SIZE
        zone = stats.mode(self.zone_filter).mode
        if isinstance(zone, np.ndarray):
            zone = zone[0]
        if np.abs(position[2]) < self.Z_THRESHOLD:
            return zone
        else:
            return -1

    # Retrieves the zone of the point of interest on the map
    def get_zone(self, point_of_interest, img_map, pixels_per_cm):
        x = int(point_of_interest[0] * pixels_per_cm)
        y = int(point_of_interest[1] * pixels_per_cm)
        #map_copy = img_map.copy()
        if 0 <= x < img_map.shape[1] and 0 <= y < img_map.shape[0]:
            # cv.line(map_copy, (x-1, y), (x+1, y), (255, 0, 0), 2)
            # cv.line(map_copy, (x,y-1), (x,y+1), (255, 0, 0), 2)
            # cv.circle(map_copy, (x, y), 4, (255, 0, 0), 2)
            return img_map[y, x]#, map_copy
        else:
            return [0,0,0]#, map_copy

    # Returns the key of the dictionary in the dictionary of dictionaries that matches the color given
    def get_dict_idx_from_color(self, color):
        color_idx = 256*256*color[2] + 256*color[1] + color[0]
        return color_idx


class CamIOPlayer2D:
    def __init__(self, model):
        self.model = model
        self.prev_zone_name = ''
        self.prev_zone_moving = -1
        self.curr_zone_moving = -1
        self.sound_files = {}
        self.hotspots = {}
        self.current_channel = None
        
        if not PYGAME_AVAILABLE:
            print("Audio disabled - pygame not available")
            self.blip_sound = None
            self.map_description = None
            self.welcome_message = None
            self.goodbye_message = None
        else:
            try:
                self.blip_sound = pygame.mixer.Sound(self.model['blipsound'])
                self.enable_blips = False
                if "map_description" in self.model:
                    self.map_description = pygame.mixer.Sound(self.model['map_description'])
                    self.have_played_description = False
                else:
                    self.have_played_description = True
                    self.map_description = None
                self.welcome_message = pygame.mixer.Sound(self.model['welcome_message'])
                self.goodbye_message = pygame.mixer.Sound(self.model['goodbye_message'])
            except Exception as e:
                print(f"Audio initialization failed: {e}")
                self.blip_sound = None
                self.map_description = None
                self.welcome_message = None
                self.goodbye_message = None
        
        # Load hotspot sounds
        for hotspot in self.model['hotspots']:
            key = hotspot['color'][2] + hotspot['color'][1] * 256 + hotspot['color'][0] * 256 * 256
            self.hotspots.update({key:hotspot})
            if PYGAME_AVAILABLE and os.path.exists(hotspot['audioDescription']):
                try:
                    self.sound_files[key] = pygame.mixer.Sound(hotspot['audioDescription'])
                except Exception as e:
                    print(f"warning. Could not load audio file: {hotspot['audioDescription']} - {e}")
            else:
                print("warning. file not found:" + hotspot['audioDescription'])

    def play_description(self):
        if not self.have_played_description and self.map_description and PYGAME_AVAILABLE:
            try:
                if self.current_channel and self.current_channel.get_busy():
                    self.current_channel.stop()
                self.current_channel = self.map_description.play()
                self.have_played_description = True
            except Exception as e:
                print(f"Error playing description: {e}")

    def play_welcome(self):
        if self.welcome_message and PYGAME_AVAILABLE:
            try:
                self.welcome_message.play()
            except Exception as e:
                print(f"Error playing welcome: {e}")

    def play_goodbye(self):
        if self.goodbye_message and PYGAME_AVAILABLE:
            try:
                self.goodbye_message.play()
            except Exception as e:
                print(f"Error playing goodbye: {e}")

    def convey(self, zone, status):
        if status == "moving":
            if self.curr_zone_moving != zone and self.prev_zone_moving == zone and self.enable_blips:
                if self.current_channel and self.current_channel.get_busy():
                    self.current_channel.stop()
                try:
                    if self.blip_sound and PYGAME_AVAILABLE:
                        self.current_channel = self.blip_sound.play()
                except Exception as e:
                    print(f"Exception raised. Cannot play blip sound: {e}")
                self.curr_zone_moving = zone
            self.prev_zone_moving = zone
            return
        if zone not in self.hotspots:
            self.prev_zone_name = None
            return
        zone_name = self.hotspots[zone]['textDescription']
        if self.prev_zone_name != zone_name:
            if self.current_channel and self.current_channel.get_busy():
                self.current_channel.stop()
            if zone in self.sound_files and PYGAME_AVAILABLE:
                sound = self.sound_files[zone]
                try:
                    self.current_channel = sound.play()
                except Exception as e:
                    print(f"Exception raised. Cannot play sound: {e}")
            self.prev_zone_name = zone_name

