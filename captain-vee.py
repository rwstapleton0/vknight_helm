import sys, time, random, pygame, math
from pygame.math import Vector2
from collections import deque
import cv2 as cv, mediapipe as mp
import os
import numpy as np

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "data")

def load_image(name, pos, scale=1):
    fullname = os.path.join(data_dir, name)
    image = pygame.image.load(fullname).convert_alpha()

    size = image.get_size()
    size = (size[0] * scale, size[1] * scale)
    image = pygame.transform.scale(image, size)
    return image, image.get_rect()

def easeOutElastic(x):
    c4 = (2 * math.pi) / 3
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return math.pow(2, -10 * x) * math.sin((x * 10 - 0.6) * c4) + 1

def easeOutBounce(x):
    n1 = 7.5625
    d1 = 2.75
    
    if x < 1 / d1:
        return n1 * x * x
    elif x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    elif x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    else:
        x -= 2.625 / d1
        return n1 * x * x + 0.984375

# def calculate_rotation(landmarks, image_width):
#     """Calculate rotation angle based on landmarks."""
#     # Example: Use landmarks for the left and right cheek for simplicity
#     left_cheek = landmarks[234]
#     right_cheek = landmarks[454]
    
#     # Calculate the angle
#     dx = right_cheek[0] * image_width - left_cheek[0] * image_width
#     dy = right_cheek[1] * image_width - left_cheek[1] * image_width
#     angle = math.atan2(dy, dx) * 180 / math.pi
#     return angle

class ExponentialSmoothingFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_value = None

    def update(self, new_value):
        if self.last_value is None:
            smoothed_value = new_value
        else:
            smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.last_value
        self.last_value = smoothed_value
        return smoothed_value

# Initialize the filter with an alpha value (0 < alpha <= 1)
filter_x = ExponentialSmoothingFilter(alpha=0.16)
filter_y = ExponentialSmoothingFilter(alpha=0.16)

class HelmetBody(pygame.sprite.Sprite):

    def __init__(self, pos):
        super().__init__()
        self.image, self.rect = load_image("helmet_body_a.png", pos, 1)
        self.orig_image = self.image

    def update(self, pos, angle):
        # self.image = pygame.transform.rotozoom(self.orig_image, -angle, 1)

        self.rect = self.image.get_rect(center=pos)
            

class FaceGuard(pygame.sprite.Sprite):

    def __init__(self, pos):
        super().__init__()
        self.image, self.frame = load_image("face_guard_a.png", pos, 1)
        self.orig_image = self.image
        
        # Set position to center
        self.pos = pos

        # Position Offset
        self.pos_offset = Vector2(84, -52)

        # Shake Variables
        self.shake_displace = 6
        self.orig_pos = Vector2(0,0)
        self.shake_pos = Vector2(0,0)

        # Rotation Variables
        self.rot_offset = Vector2(-130, 34)
        self.angle = 0
        self.duration = 2
        self.start_time = 0
        self.open_anim = False
    
    def update(self, pos, angle):
        self.image = pygame.transform.rotozoom(self.orig_image, -self.angle, 1)
        self.offset_rotated = self.rot_offset.rotate(self.angle)

        elapsed = (pygame.time.get_ticks() - self.start_time) / 1000
        self.x = min(elapsed / self.duration, 1)

        if self.open_anim:
            self.angle = 90 * easeOutElastic(self.x)
        else:
            self.angle = 90 * (1 - easeOutBounce(self.x ** 2))

        new_pos = pos + self.shake_pos + self.offset_rotated + self.pos_offset
        self.rect = self.image.get_rect(center=new_pos)
    
    def shake(self):
        ran_range_x = (-1,1)
        ran_range_y = (-1,1)
        if self.shake_pos.x >= self.orig_pos.x + self.shake_displace:
            ran_range_x = (-1, 0)
        elif self.shake_pos.x <= self.orig_pos.x - self.shake_displace:
            ran_range_x = (0, 1)
        if self.shake_pos.y >= self.orig_pos.y + self.shake_displace:
            ran_range_y = (-1, 0)
        elif self.shake_pos.y <= self.orig_pos.y - self.shake_displace:
            ran_range_y = (0, 1)
         
        self.shake_pos += (random.randint(ran_range_x[0], ran_range_x[1])  * self.shake_displace, 
            random.randint(ran_range_y[0], ran_range_y[1]) * self.shake_displace)

    def stop_shake(self):
        self.shake_pos = Vector2(self.orig_pos)

    def start_anim(self):
        self.open_anim = not self.open_anim
        self.start_time = pygame.time.get_ticks()

# Face landmark dertector
mp_face_mesh = mp.solutions.face_mesh
pygame.init()

# Initialize required elements/environment
VID_CAP = cv.VideoCapture(0)
window_size = (VID_CAP.get(cv.CAP_PROP_FRAME_WIDTH) * 2, VID_CAP.get(cv.CAP_PROP_FRAME_HEIGHT) * 2) # width by height
screen = pygame.display.set_mode(window_size)

# Helmet sprites
screen_center = (window_size[0] // 2, window_size[1] // 2)
helmet_body = HelmetBody(screen_center)
face_guard = FaceGuard(screen_center)

# Helmet sprite group
all_sprites = pygame.sprite.Group(helmet_body, face_guard)

# Game loop
game_clock = pygame.time.Clock()
game_is_running = True

pos = (0,0)
threshold = 2
target_box_size = (220, 220)  # Width, Height of the target box

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while game_is_running:
        # Set game timer to 60fps
        game_clock.tick(60)

        # Key events for exit and start face_guard animation.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_is_running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                game_is_running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                face_guard.start_anim()

        # Read the webcam input
        ret, frame = VID_CAP.read()
        if not ret:
            print("Empty frame, continuing...")
            continue

        # Fill screen with block color
        screen.fill((125, 220, 232))

        # Flip and resize the input image 
        frame = cv.flip(frame, 1)
        frame = cv.resize(frame, (int(window_size[0]), int(window_size[1])))
        # frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process face landmarks
        result = face_mesh.process(frame)

        facelandmarks = []
        angle = 0
        if result.multi_face_landmarks and len(result.multi_face_landmarks) > 0:
            face = result.multi_face_landmarks[0]

            # Face Position Variables
            smoothed_pos_x = filter_x.update(face.landmark[4].x * window_size[0]) + 70
            smoothed_pos_y = filter_y.update(face.landmark[4].y * window_size[1])

            if abs(pos[0] - smoothed_pos_x) > threshold or abs(pos[1] - smoothed_pos_y) > threshold:
                pos = (smoothed_pos_x, smoothed_pos_y)


            # Shake face_guard functions
            if face.landmark[15].y - face.landmark[11].y > 0.036: 
                face_guard.shake()
            else:
                face_guard.stop_shake()

            # Face mask
            for i in range(0, 468):
                pt1 = face.landmark[i]
                x = int(pt1.x * window_size[0])
                y = int(pt1.y * window_size[1])
                facelandmarks.append([x, y])
            
            # Rotation
            # angle = calculate_rotation(facelandmarks, frame.shape[1])

        landmarks = np.array(facelandmarks, np.int32)
        
        convexhull = cv.convexHull(landmarks)

        # Create a mask
        mask = np.zeros((int(window_size[1]), int(window_size[0])), np.uint8)
        cv.fillConvexPoly(mask, convexhull, 255)

        green_bg = np.zeros((int(window_size[1]), int(window_size[0]), 3), np.uint8)
        green_bg[:] = (0, 255, 0)  # Green in BGR format

        # Use the mask to blend the face with the green background
        for i in range(3):  # loop through each channel (BGR)
            green_bg[:, :, i] = cv.bitwise_and(frame[:, :, i], frame[:, :, i], mask=mask) + \
                                cv.bitwise_and(green_bg[:, :, i], green_bg[:, :, i], mask=~mask)

        # Convert to a format suitable for Pygame
        x, y, w, h = cv.boundingRect(convexhull)
        face_cropped = green_bg[y:y+h, x:x+w]
        face_extracted = np.rot90(face_cropped)
        face_extracted = np.flip(face_extracted, axis=0)
        face_resized = cv.resize(face_extracted, target_box_size, interpolation=cv.INTER_AREA)

        # Display in Pygame window
        surf = pygame.surfarray.make_surface(face_resized)
        face_pos = (pos[0] -150, pos[1] -120)
        screen.blit(surf, face_pos)

        # Screen Functions 
        all_sprites.update(pos, angle)
        all_sprites.draw(screen)
        pygame.display.flip()

    VID_CAP.release()
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()