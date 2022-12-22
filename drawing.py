import pygame
from settings import *
from ray_casting import ray_casting
from map import mini_map
from  collections import deque

class Drawing:
    def __init__(self, sc, sc_map, player):
        self.player = player
        self.sc = sc
        self.sc_map = sc_map
        self.font = pygame.font.SysFont('Arial', 36, bold=True)
        self.textures = {1: pygame.image.load('D:/1.jpg').convert(),
                        2: pygame.image.load('D:/9.jpg').convert(),
                        #3: pygame.image.load('D:/johnson.jpg').convert(),
                        'S': pygame.image.load('D:/3.jpg').convert()
                        }

        # weapon parameters
        self.weapon_base_sprite = pygame.image.load('D:/sprites/weapon/base/0.png').convert_alpha()
        self.weapon_shot_animation = deque([pygame.image.load(f'D:/sprites/weapon/anim/{i}.png').convert_alpha()
                                            for i in range(5)])
        self.weapon_rect = self.weapon_base_sprite.get_rect()
        self.weapon_pos = (HALF_WIDTH - self.weapon_rect.width // 2, HEIGHT - self.weapon_rect.height)
        self.shot_length = len(self.weapon_shot_animation)
        self.shot_length_count = 0
        self.shot_animation_speed = 3
        self.shot_animation_count = 0
        self.shot_animation_trigger = True

        # sfx parameters
        self.sfx = deque([pygame.image.load(f'D:/sprites/weapon/effect/{i}.png').convert_alpha() for i in range(6)])
        self.sfx_lenght_count = 0
        self.sfx_lenght = len(self.sfx)


    def background(self, angle):
        sky_offset = -5 * math.degrees(angle) % WIDTH
        self.sc.blit(self.textures['S'], (sky_offset, 0))
        self.sc.blit(self.textures['S'], (sky_offset - WIDTH, 0))
        self.sc.blit(self.textures['S'], (sky_offset + WIDTH, 0))
        pygame.draw.rect(self.sc, DARKGRAY, (0, HALF_HEIGHT, WIDTH, HALF_HEIGHT))

    def world(self, world_objects):
        for obj in sorted(world_objects, key=lambda n: n[0], reverse=True):
            if obj[0]:
                _, object, object_pos = obj
                self.sc.blit(object, object_pos)

    def fps(self, clock):
        display_fps = str(int(clock.get_fps()))
        render = self.font.render(display_fps, 0, RED)
        self.sc.blit(render, FPS_POS)

    def mini_map(self, player):
        self.sc_map.fill(BLACK)
        map_x, map_y = player.x // MAP_SCALE, player.y // MAP_SCALE
        pygame.draw.line(self.sc_map, YELLOW, (map_x, map_y), (map_x + 12 * math.cos(player.angle),
                                                map_y + 12 * math.sin(player.angle)), 2)
        pygame.draw.circle(self.sc_map, RED, (int(map_x), int(map_y)), 5)
        for x, y in mini_map:
            pygame.draw.rect(self.sc_map, SANDY, (x, y, MAP_TILE, MAP_TILE))
        self.sc.blit(self.sc_map, MAP_POS)

    def player_weapon(self, shots):
        if self.player.shot:
            self.shot_projection = min(shots)[1] // 2
            self.bullet_sfx()
            shot_sprite = self.weapon_shot_animation[0]
            self.sc.blit(shot_sprite, self.weapon_pos)
            self.shot_animation_count += 1
            if self.shot_animation_count == self.shot_animation_speed:
                self.weapon_shot_animation.rotate(-1)
                self.shot_animation_count = 0
                self.shot_length_count += 1
                self.shot_animation_trigger = False
            if self.shot_length_count == self.shot_length:
                self.player.shot = False
                self.shot_length_count = 0
                self.sfx_lenght_count = 0
                self.shot_animation_trigger = True
        else:
            self.sc.blit(self.weapon_base_sprite, self.weapon_pos)

    def bullet_sfx(self):
        if self.sfx_lenght_count < self.sfx_lenght:
            sfx = pygame.transform.scale(self.sfx[0], (self.shot_projection, self.shot_projection))
            sfx_rect = sfx.get_rect()
            self.sc.blit(sfx, (HALF_WIDTH - sfx_rect.w // 2, HALF_HEIGHT - sfx_rect.h // 2))
            self.sfx_lenght_count += 1
            self.sfx.rotate(-1)