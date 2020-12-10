import os
import sys
import numpy as np
import ple
import pygame
from pygame.constants import K_UP, K_DOWN, K_RIGHT, K_LEFT

MOVE_SIZE = 1
BEACON_MOVE_SIZE = 1

class PersonPlayer(pygame.sprite.Sprite):
    def __init__(self,
                SCREEN_WIDTH, SCREEN_HEIGHT,
                image_assets, rng):

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        pygame.sprite.Sprite.__init__(self)

        self.image_assets = image_assets

        self.flapped = True  # start off w/ a flap
        self.current_image = 0

        self.rng = rng
        self.image = self.image_assets
        self.rect = self.image.get_rect()

        self.width = self.image.get_width()
        self.height = self.image.get_height()
        
        self.game_tick = 0
   
        # self.scale = scale
        self.init()

        
    def init(self):
        self.pos_x = np.random.randint(0, int(self.SCREEN_WIDTH)-self.height)
        self.pos_y = np.random.randint(0, int(self.SCREEN_HEIGHT)-self.width)
        
    def draw(self, screen):
        screen.blit(self.image, self.rect.center)
    
    def update(self, dt):
        self.game_tick += 1
        self.rect.center = (self.pos_x, self.pos_y)

    def move(self, direction):
        switcher = { 
            "up": [-MOVE_SIZE, 0], 
            "down": [MOVE_SIZE, 0], 
            "right": [0, MOVE_SIZE], 
            "left": [0, -MOVE_SIZE], 
        } 

        new_x = switcher[direction][0]
        new_y = switcher[direction][1]

        if self.pos_x >= 0 and self.pos_x<self.SCREEN_WIDTH-self.width-1 and new_x == MOVE_SIZE:
            self.pos_x += new_x

        if self.pos_x > 1 and self.pos_x<=self.SCREEN_WIDTH-self.width and new_x == -MOVE_SIZE:
            self.pos_x += new_x
         
        if self.pos_y >= 0 and self.pos_y<self.SCREEN_HEIGHT-self.height-1 and new_y == MOVE_SIZE:
            self.pos_y += new_y

        if self.pos_y > 1 and self.pos_y<=self.SCREEN_HEIGHT-self.height and new_y == -MOVE_SIZE:
            self.pos_y += new_y


class Beacon(pygame.sprite.Sprite):
    def __init__(self,
                SCREEN_WIDTH, SCREEN_HEIGHT,
                image_assets, rng):

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        pygame.sprite.Sprite.__init__(self)

        self.image_assets = image_assets

        self.flapped = True  # start off w/ a flap
        self.current_image = 0

        self.rng = rng
        self.image = self.image_assets
        self.rect = self.image.get_rect()

        self.width = self.image.get_width()
        self.height = self.image.get_height()
        
        self.game_tick = 0
 
        # self.scale = scale
        self.init()

        
    def init(self):
        self.pos_x = np.random.randint(0, int(self.SCREEN_WIDTH)-self.height)
        self.pos_y = np.random.randint(0, int(self.SCREEN_HEIGHT)-self.width)
        
    def draw(self, screen):
        screen.blit(self.image, self.rect.center)
    
    def update(self, dt):
        self.game_tick += 1

        new_x = self.pos_x
        new_y = self.pos_y

        if self.game_tick % 10 == 0:
            # print("trying to update")
            new_x = np.random.randint(-BEACON_MOVE_SIZE, BEACON_MOVE_SIZE)
            new_y = np.random.randint(-BEACON_MOVE_SIZE, BEACON_MOVE_SIZE)

        if self.pos_x >= 0 and self.pos_x<self.SCREEN_WIDTH-self.width-1 and new_x == BEACON_MOVE_SIZE:
            self.pos_x += new_x

        if self.pos_x > 1 and self.pos_x<=self.SCREEN_WIDTH-self.width and new_x == -BEACON_MOVE_SIZE:
            self.pos_x += new_x
         
        if self.pos_y >= 0 and self.pos_y<self.SCREEN_HEIGHT-self.height-1 and new_y == BEACON_MOVE_SIZE:
            self.pos_y += new_y

        if self.pos_y > 1 and self.pos_y<=self.SCREEN_HEIGHT-self.height and new_y == -BEACON_MOVE_SIZE:
            self.pos_y += new_y


        self.rect.center = (self.pos_x, self.pos_y)

    def is_collided_with(self, sprite):
        return self.rect.colliderect(sprite.rect)

    


class Background():

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT,
                 image_background):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.background_image = image_background

        self.x = 0


    def draw_background(self, screen):
        screen.blit(self.background_image, (0, 0))


class MoveToBeacon(ple.games.base.PyGameWrapper):

    def __init__(self, width=128, height=128):

        self.actions = {
            "up": K_UP, 
            "down": K_DOWN,
            "right": K_RIGHT,
            "left": K_LEFT
        }

        # fps = 30

        ple.games.base.PyGameWrapper.__init__(self, width, height, actions=self.actions)

        # self.scale = 30.0 / fps

        # self.allowed_fps = 30  # restrict the fps  

        self.init_pos = (
            int(self.width * 0.2),
            int(self.height / 2)
        )
  
        # so we can preload images
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

        asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/")
        self.images = self._load_images(asset_dir)

        self.backdrop = None
        self.player = None
        self.beacon = None

        self.score = 0.0



    def getActionSet(self):
        return self.actions

    def _load_images(self, asset_dir):
        images = {}
        # preload and convert all the images so its faster when we reset
        # print(os.path.join(asset_dir, "person.png"))
        images["player"] = pygame.image.load(os.path.join(asset_dir, "person.png")).convert_alpha()

        images["goal"] = pygame.image.load(os.path.join(asset_dir, "goal.png")).convert_alpha()

        images["background"] = pygame.image.load(os.path.join(asset_dir, "background.png"))

        # images["base"] = pygame.image.load(os.path.join(asset_dir, "background.png"))
    
        
        return images

    def init(self):
    
        self.score = 0.0
        self.lives = 1
        self.game_tick = 0

        self.reset()

    def reset(self):

        if self.backdrop is None:
            self.backdrop = Background(
                self.width,
                self.height,
                self.images["background"]
            )


        if self.player is None:
            self.player = PersonPlayer(
                self.width,
                self.height,
                self.images["player"],
                self.rng)
        else:
            self.player.init()


        if self.beacon is None:
            self.beacon = Beacon(
                self.width,
                self.height,
                self.images["goal"],
                self.rng)
        else:
            self.beacon.init()
    

    def getGameState(self):
        state = {
            "player_x": self.player.pos_x,
            "player_y": self.player.pos_y,
            "beacon_x": self.beacon.pos_x,
            "beacon_y": self.beacon.pos_y           
        }

        return state

    def getScore(self):
        return self.score

    def _handle_player_events(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == self.actions['up']:
                    self.player.move('up')
                if key == self.actions['down']:
                    self.player.move('down')
                if key == self.actions['right']:
                    self.player.move('right')
                if key == self.actions['left']:
                    self.player.move('left')
                
    def game_over(self):
        # print(self.game_tick)
        return self.game_tick > 1000

    def collision(self, player, beacon):

        if beacon.is_collided_with(player):
            return True

        return False

    def step(self, dt):
        self.game_tick += 1
        dt = dt / 1000.0

        self.score += self.rewards["tick"]

        # handle player movement
        self._handle_player_events()

        self.player.update(dt)
        self.beacon.update(dt)

        if self.collision(self.player, self.beacon):
            self.score += 1.0
            self.reset()
            
        self.backdrop.draw_background(self.screen)
        self.beacon.draw(self.screen)
        self.player.draw(self.screen)