
import os 
os.environ['SDL_AUDIODRIVER'] = 'alsa'

import pygame
pygame.mixer.pre_init(44100, -16, 2, 1024)
pygame.init()

sound = pygame.mixer.Sound("/home/user/camio/models/RACITFirstFloor/Audio/8th auditorium.wav")  # or .wav
sound.set_volume(0.8) 
sound.play()

while pygame.mixer.get_busy():
    pass