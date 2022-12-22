from player import Player
from sprite_objects import *
from ray_casting import ray_casting_walls
from drawing import Drawing
from interaction import Interaction

pygame.init()
sc = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.mouse.set_visible(False)
sc_map = pygame.Surface(MINIMAP_RES)

sprites = Sprites()
clock = pygame.time.Clock()
player = Player(sprites)
drawing = Drawing(sc, sc_map, player)
interaction = Interaction(player, sprites, drawing)


while True:

    player.movement()
    drawing.background(player.angle)
    walls, wall_shot = ray_casting_walls(player, drawing.textures)
    drawing.world(walls + [obj.object_locate(player) for obj in sprites.list_of_objects])
    drawing.fps(clock)
    drawing.mini_map(player)
    drawing.player_weapon([wall_shot, sprites.sprite_shot])

    interaction.interaction_objects()
    interaction.npc_action()

    pygame.display.flip()
    clock.tick()