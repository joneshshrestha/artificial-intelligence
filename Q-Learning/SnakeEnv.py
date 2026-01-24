"""
CSC 580 HW#2 "SnakeEnv.py" -- Environment for the Snake Game

Inspired by "Snake Played by a Deep Reinforcement Learning Agent" by Hennie de Harder 
(https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)

Code in this file defines class SnakeEnv, an environment derived from gymnasium Env.

"""

import gymnasium as gym
import turtle
import random
import time
import math

HEIGHT = 20      # number of steps vertically from wall to wall of screen
WIDTH = 20       # number of steps horizontally from wall to wall of screen
PIXEL_H = 20*HEIGHT  # pixel height + border on both sides
PIXEL_W = 20*WIDTH   # pixel width + border on both sides

SLEEP = 0.2     # time to wait between steps

GAME_TITLE = 'Snake'
BG_COLOR = 'white'

SNAKE_SHAPE = 'square'
SNAKE_COLOR = 'black'
SNAKE_START_LOC_H = 0
SNAKE_START_LOC_V = 0

APPLE_SHAPE = 'circle'
APPLE_COLOR = 'green'

class SnakeEnv(gym.Env):

    def __init__(self, human=False, env_info={'state_space':None}):
        super(SnakeEnv, self).__init__()

        self.done = False
        self.seed()
        self.reward = 0
        self.action_space = 4
        self.state_space = 12

        self.human = human
        self.env_info = env_info
        self.total, self.maximum = 0, 0

        self.display = False  # to control game screen update

        ## GAME CREATION WITH TURTLE
        # screen/background
        self.win = turtle.Screen()
        self.win.title(GAME_TITLE)
        self.win.bgcolor(BG_COLOR)
        self.win.tracer(0)
        self.win.setup(width=PIXEL_W+32, height=PIXEL_H+32)
                
        # snake
        self.snake = turtle.Turtle()
        self.snake.shape(SNAKE_SHAPE)
        self.snake.speed(0)
        self.snake.penup()
        self.snake.color(SNAKE_COLOR)
        self.snake.goto(SNAKE_START_LOC_H, SNAKE_START_LOC_V)
        self.snake.direction = 'stop'
        # snake body, add first element (for location of snake's head)
        self.snake_body = []
        self.add_to_body()

        # apple
        self.apple = turtle.Turtle()
        self.apple.speed(0)
        self.apple.shape(APPLE_SHAPE)
        self.apple.color(APPLE_COLOR)
        self.apple.penup()
        self.move_apple(first=True)

        # distance between apple and snake
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + 
                              (self.snake.ycor()-self.apple.ycor())**2)

        # score
        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color('black')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 100)
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}", 
                         align='center', font=('Courier', 18, 'normal'))

        # keyboard control, for the 'human' mode
        if self.human:
            self.win.listen()
            self.win.onkey(self.go_up, 'Up')
            self.win.onkey(self.go_right, 'Right')
            self.win.onkey(self.go_down, 'Down')
            self.win.onkey(self.go_left, 'Left')

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def random_coordinates(self):
        #apple_x = random.randint(-WIDTH/2, WIDTH/2)
        apple_x = random.randint(-WIDTH//2, WIDTH//2)
        #apple_y = random.randint(-HEIGHT/2, HEIGHT/2)
        apple_y = random.randint(-HEIGHT//2, HEIGHT//2)
        return apple_x, apple_y
    
    def move_snake(self):
        if self.snake.direction == 'stop':
            self.reward = 0
        if self.snake.direction == 'up':
            y = self.snake.ycor()
            self.snake.sety(y + 20)
        if self.snake.direction == 'right':
            x = self.snake.xcor()
            self.snake.setx(x + 20)
        if self.snake.direction == 'down':
            y = self.snake.ycor()
            self.snake.sety(y - 20)
        if self.snake.direction == 'left':
            x = self.snake.xcor()
            self.snake.setx(x - 20)
    
    def go_up(self):
        if self.snake.direction != "down":
            self.snake.direction = "up"
    
    def go_down(self):
        if self.snake.direction != "up":
            self.snake.direction = "down"
    
    def go_right(self):
        if self.snake.direction != "left":
            self.snake.direction = "right"
    
    def go_left(self):
        if self.snake.direction != "right":
            self.snake.direction = "left"

    def move_apple(self, first=False):
        if first or self.snake.distance(self.apple) < 20:    
            while True:
                self.apple.x, self.apple.y = self.random_coordinates()
                self.apple.goto(round(self.apple.x*20), round(self.apple.y*20))
                if not self.body_check_apple():
                    break
            if not first:
                self.update_score()
                self.add_to_body()
            first = False
            return True

    def update_score(self):
        self.total += 1
        if self.total >= self.maximum:
            self.maximum = self.total
        self.score.clear()
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}", 
                         align='center', font=('Courier', 18, 'normal'))

    def reset_score(self):
        self.score.clear()
        self.total = 0
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}", 
                         align='center', font=('Courier', 18, 'normal'))

    def add_to_body(self):
        body = turtle.Turtle()
        body.speed(0)
        body.shape('square')
        body.color('black')
        body.penup()
        self.snake_body.append(body)

    def move_snakebody(self):
        if len(self.snake_body) > 0:
            for index in range(len(self.snake_body)-1, 0, -1):
                x = self.snake_body[index-1].xcor()
                y = self.snake_body[index-1].ycor()
                self.snake_body[index].goto(x, y)

            self.snake_body[0].goto(self.snake.xcor(), self.snake.ycor())
        
    def measure_distance(self):
        self.prev_dist = self.dist
        self.dist = math.sqrt((self.snake.xcor()-self.apple.xcor())**2 + 
                              (self.snake.ycor()-self.apple.ycor())**2)

    def body_check_snake(self):
        if len(self.snake_body) > 1:
            for body in self.snake_body[1:]:
                if body.distance(self.snake) < 20:
                    self.reset_score()
                    return True     

    def body_check_apple(self):
        if len(self.snake_body) > 0:
            for body in self.snake_body[:]:
                if body.distance(self.apple) < 20:
                    return True

    def wall_check(self):
        if self.snake.xcor() > 200 or self.snake.xcor() < -200 or \
           self.snake.ycor() > 200 or self.snake.ycor() < -200:
            self.reset_score()
            return True
    
    def reset(self, wrap_around = False):
        if self.human or self.display:
            time.sleep(1)
        for body in self.snake_body:
            body.goto(1000, 1000)

        self.snake_body = []
        if not wrap_around:
            self.snake.goto(SNAKE_START_LOC_H, SNAKE_START_LOC_V)
        else:
            if self.snake.xcor() > 200:
                self.snake.setx(-PIXEL_W/2)
            elif self.snake.xcor() < -200:
                self.snake.setx(PIXEL_W/2)
            
            if self.snake.ycor() > 200:
                self.snake.sety(-PIXEL_H/2)
            elif self.snake.ycor() < -200:
                self.snake.sety(PIXEL_H/2)
            
        if not wrap_around:
            self.snake.direction = 'stop'
        self.reward = 0
        self.total = 0
        self.done = False

        state = self.get_state()

        return state

    def run_step(self):
        reward_given = False
        
        if self.display:
            self.win.update()
            #print ("xcoord: {}, ycoord: {}".format(self.snake.xcor(), self.snake.ycor()))
            time.sleep(SLEEP)
            #turtle.TurtleScreen._RUNNING = True
            
        self.move_snake()
        if self.move_apple():
            self.reward = 10
            reward_given = True
        self.move_snakebody()
        self.measure_distance()
        if self.body_check_snake():
            self.reward = -100
            reward_given = True
            self.done = True
        if self.wall_check():
            self.reward = -100
            reward_given = True
            self.done = True
        if not reward_given:
            if self.dist < self.prev_dist:
                self.reward = 1
            else:
                self.reward = -1
        #time.sleep(0.1)
        if self.human:
            state = self.get_state()
            time.sleep(SLEEP)
        #state = self.get_state()
    
    # AI agent
    def step(self, action):
        if action == 0:
            self.go_up()
        if action == 1:
            self.go_right()
        if action == 2:
            self.go_down()
        if action == 3:
            self.go_left()
            
        # Make one step
        self.run_step()
        rew = self.reward
        done = self.done
        
        if self.done:
            if self.human:
                self.reset()
            else:
                self.reset(True)

        # Get the next state
        state = self.get_state()
        
        return state, rew, done, {}

    def get_state(self):
        # snake coordinates abs
        self.snake.x, self.snake.y = self.snake.xcor()/WIDTH, \
                                     self.snake.ycor()/HEIGHT   
        # snake coordinates scaled 0-1
        self.snake.xsc, self.snake.ysc = self.snake.x/WIDTH+0.5,\
                                         self.snake.y/HEIGHT+0.5
        # apple coordintes scaled 0-1 
        self.apple.xsc, self.apple.ysc = self.apple.x/WIDTH+0.5, \
                                         self.apple.y/HEIGHT+0.5

        # wall check
        if self.snake.y >= HEIGHT/2:
            wall_up, wall_down = 1, 0
        elif self.snake.y <= -HEIGHT/2:
            wall_up, wall_down = 0, 1
        else:
            wall_up, wall_down = 0, 0
        if self.snake.x >= WIDTH/2:
            wall_right, wall_left = 1, 0
        elif self.snake.x <= -WIDTH/2:
            wall_right, wall_left = 0, 1
        else:
            wall_right, wall_left = 0, 0

        # body close
        body_up = []
        body_right = []
        body_down = []
        body_left = []
        if len(self.snake_body) > 3:
            for body in self.snake_body[3:]:
                if body.distance(self.snake) == 20:
                    if body.ycor() < self.snake.ycor():
                        body_down.append(1)
                    elif body.ycor() > self.snake.ycor():
                        body_up.append(1)
                    if body.xcor() < self.snake.xcor():
                        body_left.append(1)
                    elif body.xcor() > self.snake.xcor():
                        body_right.append(1)
        
        if len(body_up) > 0: body_up = 1
        else: body_up = 0
        if len(body_right) > 0: body_right = 1
        else: body_right = 0
        if len(body_down) > 0: body_down = 1
        else: body_down = 0
        if len(body_left) > 0: body_left = 1
        else: body_left = 0

        # state: apple_up, apple_right, apple_down, apple_left, obstacle_up, 
        #        obstacle_right, obstacle_down, obstacle_left, 
        #        direction_up, direction_right, direction_down, direction_left
        if self.env_info['state_space'] == 'coordinates':
            state = [self.apple.xsc, self.apple.ysc, self.snake.xsc, self.snake.ysc,
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]
        elif self.env_info['state_space'] == 'no direction':
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    0, 0, 0, 0]
        elif self.env_info['state_space'] == 'no body knowledge':
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    wall_up, wall_right, wall_down, wall_left,
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]
        else:
            state = [int(self.snake.y < self.apple.y), int(self.snake.x < self.apple.x), int(self.snake.y > self.apple.y), int(self.snake.x > self.apple.x), \
                    int(wall_up or body_up), int(wall_right or body_right), int(wall_down or body_down), int(wall_left or body_left), \
                    int(self.snake.direction == 'up'), int(self.snake.direction == 'right'), int(self.snake.direction == 'down'), int(self.snake.direction == 'left')]
            
        # print(state)
        return state

    def close(self):
        self.win.bye()

#---------------------------------------------------------------
# [Extra] Possible reachable states (in the state-interger space)
#---------------------------------------------------------------

def generate_reachable_states():
    # apple can be 1 or 2 directions eg. right and up -- 8
    apple_positions = ["1000", "0100", "0010", "0001", "1100", "1001", "0110", "0011"]
    # obstacles could be any directions -- 16
    ob_positions = ["0000", "1000", "0100", "0010", "0001", "1100", "1010", "1001", "0110", "0101", "0011", 
                    "1110", "1101", "1011", "0111", "1111"]
    # snake direction is either one direction (or no direction) -- 5
    snake_directions = ["1000", "0100", "0010", "0001", "0000"]
    
    state_list = set()
    for apple in apple_positions:
        for ob in ob_positions:
            for snake in snake_directions:
                state_list.add(int("".join([apple, ob, snake]), 2)) # states as integers
    
    return state_list
    
# The function above can be called by
#
# states = generate_reachable_states()
# print ("#states = {}\nstates = {}".format(len(states), states))
#