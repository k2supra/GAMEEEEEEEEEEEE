from settings import *

_ = False
matrix_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, _, _, _, _, 2, 2, _, _, _, _, _, _, _, _, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, _, 2, 1, 1, 1, _, 1, 1, 1, 1, 1, 1, _, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, _, 2, 1, 1, _, _, _, 1, _, _, 1, 1, _, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, _, 2, 1, 1, _, 1, 1, 1, _, 1, 1, 1, _, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, _, 2, 1, _, _, _, _, _, _, _, _, _, _, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, _, 2, 1, 1, 1, 1, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, _, 2, 2, 2, _, _, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, _, _, _, _, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, _, _, _, _, _, _, _, _, _, _, 1],
    [1, 1, 1, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, _, _, _, _, _, _, _, _, _, _, 1],
    [1, 1, 1, _, 1, 1, 1, 1, 1, 1, 1, 1, 1, _, _, 1, 1, _, 2, 2, _, _, _, 1],
    [1, 1, _, _, _, _, _, _, _, _, _, _, 1, _, _, 1, 1, _, 2, 2, _, _, _, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, _, _, _, _, _, _, _, _, _, _, _, _, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, _, _, _, _, _, _, _, _, _, _, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

WORLD_WIDTH = len(matrix_map[0]) * TILE
WORLD_HEIGHT = len(matrix_map) * TILE
world_map = {}
mini_map = set()
for j, row in enumerate(matrix_map):
    for i, char in enumerate(row):
        if char:
            mini_map.add((i * MAP_TILE, j * MAP_TILE))
            if char == 1:
                world_map[((i * TILE, j * TILE))] = 1
            elif char == 2:
                world_map[((i * TILE, j * TILE))] = 2