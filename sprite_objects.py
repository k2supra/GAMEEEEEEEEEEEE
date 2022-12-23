import pygame
from settings import *
from collections import deque
from ray_casting import mapping
from numba.core import types
from numba.typed import Dict
from numba import int32


class Sprites:
    def __init__(self):
        self.sprites_parameters = {
            'sprite_barrel': {
                'sprite': pygame.image.load('D:/sprites/barrel/barrel.png').convert_alpha(),
                'viewing_angles': None,
                'shift': 1.8,
                'side': 30,
                'scale': (0.4, 0.4),
                'animation': deque(
                    [pygame.image.load(f'D:/sprites/barrel_gold/{i}.png') for i in range(10)]),
                'is_dead': 'immortal',
                'dead_shift': -10,
                'animation_dist': 800,
                'animation_speed': 10,
                'death_animation': None,
                'blocked': True,
                'flag': 'decor',
                'obj_action': []
            },

            'normal_barrel': {
                'sprite': pygame.image.load('D:/sprites/barrel/barrel.png').convert_alpha(),
                'viewing_angles': None,
                'shift': 0,
                'scale': (0.4, 0.4),
                'side': 30,
                'animation': [],
                'animation_dist': 800,
                'animation_speed': 10,
                'death_animation': deque(
                    [pygame.image.load(f'D:/sprites/barrel/m_barrel/{i}.png').convert_alpha() for i in range(2)]),
                'is_dead': None,
                'dead_shift': 2,
                'blocked': True,
                'flag': 'enemy',
                'obj_action': []
            },

            'npc_johnson': {
                'sprite': pygame.image.load('D:/sprites/johnson_ph/0.jpg').convert_alpha(),
                'viewing_angles': None,
                'shift': 0.0,
                'scale': (0.8, 0.8),
                'side': 60,
                'animation': deque(
                    [pygame.image.load(f'D:/sprites/johnson_ph/0.jpg')]),
                'is_dead': 'immortal',
                'dead_shift': 5,
                'animation_dist': 800,
                'animation_speed': 5,
                'death_animation': None,
                'blocked': True,
                'flag': 'npc',
                'obj_action': deque(
                    [pygame.image.load(f'D:/sprites/johnson_ph/{i}.jpg').convert_alpha() for i in range(4)]),
            }
        }

        self.list_of_objects = [
            SpriteObject(self.sprites_parameters['sprite_barrel'], (7.1, 2.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (7.8, 2.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (8.8, 3.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (14.8, 1.5)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (14.8, 2.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (17.8, 3.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (2.3, 1.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (2.5, 7.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (5.6, 5.6)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (20.8, 5.1)),
            SpriteObject(self.sprites_parameters['normal_barrel'], (20.8, 6.1)),
            # SpriteObject(self.sprites_parameters['sprite_barrel'], (7.9, 2.1)),
            SpriteObject(self.sprites_parameters['npc_johnson'], (10.9, 1.5)),
        ]

    @property
    def sprite_shot(self):
        return min([obj.is_on_fire for obj in self.list_of_objects], default=(float('inf'), 0))


class SpriteObject:
    def __init__(self, parameters, pos):
        self.distance_to_sprite = None
        self.current_ray = None
        self.object = parameters['sprite'].copy()
        self.viewing_angles = parameters['viewing_angles']
        self.shift = parameters['shift']
        self.scale = parameters['scale']
        self.animation = parameters['animation'].copy()
        self.death_animation = parameters['death_animation']
        self.is_dead = parameters['is_dead']
        self.dead_shift = parameters['dead_shift']
        self.animation_dist = parameters['animation_dist']
        self.animation_speed = parameters['animation_speed']
        self.blocked = parameters['blocked']
        self.obj_action = parameters['obj_action'].copy()
        self.dead_animation_count = 0
        self.side = 40
        self.animation_count = 0
        self.flag = parameters['flag']
        self.npc_action_trigger = False
        self.x, self.y = pos[0] * TILE, pos[1] * TILE
        if self.viewing_angles:
            if len(self.object) == 8:
                self.sprite_angles = [frozenset(range(338, 361)) | frozenset(range(0, 23))] + \
                                    [frozenset(range(i, i + 36)) for i in range(23, 338, 36)]
            else:
                self.sprite_angles = [frozenset(range(348, 361)) | frozenset(range(0, 11))] + \
                                    [frozenset(range(i, i + 23)) for i in range(11, 348, 23)]
            self.sprite_positions = {angle: pos for angle, pos in zip(self.sprite_angles, self.object)}

    @property
    def is_on_fire(self):
        if CENTER_RAY - self.side // 2 < self.current_ray < CENTER_RAY + self.side // 2 and self.blocked:
            return self.distance_to_sprite, self.proj_height
        return float('inf'), None

    @property
    def pos(self):
        return self.x - self.side // 2, self.y - self.side // 2

    def object_locate(self, player):
        dx, dy = self.x - player.x, self.y - player.y
        self.distance_to_sprite = math.sqrt(dx ** 2 + dy ** 2)

        self.theta = math.atan2(dy, dx)
        gamma = self.theta - player.angle
        if dx > 0 and 180 <= math.degrees(player.angle) <= 360 or dx < 0 and dy < 0:
            gamma += DOUBLE_PI
        self.theta -= 1.4 * gamma

        delta_rays = int(gamma / DELTA_ANGLE)
        self.current_ray = CENTER_RAY + delta_rays
        self.distance_to_sprite *= math.cos(HALF_FOV - self.current_ray * DELTA_ANGLE)

        fake_ray = self.current_ray + FAKE_RAYS
        if 0 <= fake_ray <= FAKE_RAYS_RANGE and self.distance_to_sprite > 30:
            self.proj_height = min(int(PROJ_COEFF / self.distance_to_sprite), DOUBLE_HEIGHT)
            sprite_width = int(self.proj_height * self.scale[0])
            sprite_height = int(self.proj_height * self.scale[1])
            half_sprite_width = sprite_width // 2
            half_sprite_height = sprite_height // 2
            shift = half_sprite_height * self.shift

            if self.is_dead and self.is_dead != 'immortal':
                sprite_object = self.dead_animation()
                shift = half_sprite_height * self.dead_shift
                sprite_height = int(sprite_height / 1.3)
            elif self.npc_action_trigger:
                sprite_object = self.npc_in_action()
            else:
                self.object = self.visible_sprite()
                sprite_object = self.sprite_animation()

            # sprite scale and pos

            sprite_pos = (self.current_ray * SCALE - half_sprite_width, HALF_HEIGHT - half_sprite_height + shift)
            sprite = pygame.transform.scale(sprite_object, (sprite_width, sprite_height))
            return (self.distance_to_sprite, sprite, sprite_pos)
        else:
            return (False,)

    def sprite_animation(self):
        if self.animation and self.distance_to_sprite < self.animation_dist:
            sprite_object = self.animation[0]
            if self.animation_count < self.animation_speed:
                self.animation_count += 1
            else:
                self.animation.rotate()
                self.animation_count = 0
            return sprite_object
        return self.object

    def visible_sprite(self):
        if self.viewing_angles:
            if self.theta < 0:
                self.theta += DOUBLE_PI
            self.theta = 360 - int(math.degrees(self.theta))

            for angles in self.sprite_angles:
                if self.theta in angles:
                    return self.sprite_positions[angles]
        return self.object

    def dead_animation(self):
        if len(self.death_animation):
            if self.dead_animation_count < self.animation_speed:
                self.dead_sprite = self.death_animation[0]
                self.dead_animation_count += 1
            else:
                self.dead_sprite = self.death_animation.popleft()
                self.dead_animation_count = 0
        return self.dead_sprite

    def npc_in_action(self):
        sprite_object = self.obj_action[0]
        if self.animation_count < self.animation_speed:
            self.animation_count += 1
        else:
            self.obj_action.rotate()
            self.animation_count = 0
        return sprite_object