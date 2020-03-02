import time
from source.image_recognition.brick import Brick, BrickArray
from time import perf_counter as clock
from source.settings.configuration import Globals as Glob
from source.image_recognition.image_tools import *
from source.image_recognition.drawing import *
from source.image_recognition.hand_button import HandButton
from OpenGL.GL import *
from source.settings.resources import Resources
from multiprocessing import SimpleQueue
from source.physics.mechanics import update_stress
import random

# strings ressources for futur languages
strings = Resources.strings['fr']


# AugmentedReality class is split for each Scene
class AugmentedReality:
    """ Main class"""

    def __init__(self, cam):
        self.cam = cam


# Main Scene : the game
class GameAR(AugmentedReality):
    """ Take and process webcam frames"""

    def __init__(self, cam, width: int, height: int, q_activate: SimpleQueue, liquid_im, liquid_grid) -> None:
        super().__init__(cam)
        Glob.death_text = ""
        # Attributes from parameters
        self.width, self.liquid_height = width, height
        self.q_activate = q_activate
        self.liquid_im = liquid_im
        self.liquid_grid = liquid_grid

        # Other attributes
        self.triggered_start, self.triggered_reset = False, False
        self.wait_time = 1
        self.liquid_height = 0.0
        self.clock_liquid = 0.0
        self.number = -1
        self.new = False

        # Create a texture handler with 6 different textures
        self.tex_handler = TextureHandler(14)
        image = cv2.imread("./ressources/Thermo1final.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        image = cv2.resize(image, (Conf.width, Conf.height))
        image = cv2.flip(image, 0)
        size = np.shape(image)[:2]
        self.tex_handler.bind_texture(12,image, size[1], size[0])

        image = cv2.imread("./ressources/Thermo2final.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        image = cv2.resize(image, (Conf.width, Conf.height))
        image = cv2.flip(image, 0)
        size = np.shape(image)[:2]
        self.tex_handler.bind_texture(11, image, size[1], size[0])

        image = cv2.imread("./ressources/Thermo3final.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        image = cv2.resize(image, (Conf.width, Conf.height))
        image = cv2.flip(image, 0)
        size = np.shape(image)[:2]
        self.tex_handler.bind_texture(10, image, size[1], size[0])

        # Create a handler for every drawing functions
        self.draw_handler = DrawingHandler(self.tex_handler, q_activate, liquid_im)

        # Create a class that will extract "bricks" from image
        self.brick_recognition = BrickRecognition(liquid_im)

        # Create button Handler and start them for image detection
        self.buttonStart, self.buttonReset = None, None
        self.init_start_buttons()

    def init_start_buttons(self):
        self.buttonStart = HandButton(0, None, 3, Conf.hand_area_1, Conf.hand_threshold_1)
        self.buttonReset = HandButton(1, None, 1, Conf.hand_area_2, Conf.hand_threshold_2)
        self.buttonStart.daemon = True
        self.buttonReset.daemon = True
        self.buttonStart.start()
        self.buttonReset.start()
        self.buttonReset.title = "RETOUR"

    def reset(self):
        """ Reset program state"""

        if self.buttonStart.number > self.draw_handler.previous_number:
            self.draw_handler.previous_number = self.buttonStart.number

        self.triggered_start, self.triggered_reset = False, False
        self.wait_time = 1

        # self.draw_handler = DrawingHandler(self.tex_handler, self.q_activate, self.liquid_im)
        self.brick_recognition = BrickRecognition(self.liquid_im)

        self.buttonReset.is_triggered, self.buttonReset.is_waiting = False, False
        Glob.brick_array.reset()

    def render(self) -> None:
        """ render the scene with OpenGL"""

        if Glob.frame is not None:
            # Set button text
            if Glob.mode == 0:
                self.draw_handler.q = 0.0
                self.number = -1
                self.buttonStart.title = "VALIDER"
            elif self.buttonStart.is_ready():
                self.buttonStart.title = "CONTINUER"
            else:
                self.buttonStart.title = "%i" % self.buttonStart.remaining_time

            if self.number == -1:
                self.buttonStart.wait_time = 1
            else:
                self.buttonStart.wait_time = 10

            # Set liquid state from buttons
            poor_liquid = False
            if self.buttonStart.is_triggered and self.number >= 0:
                poor_liquid = True

            # step 1 : draw background
            self.draw_handler.draw_frame(Glob.frame)

            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]
            # step 2 : draw molten steel
            self.draw_handler.draw_molten_steel(x0, y0, xf - x0, yf - y0, 1, 1, 1, poor_liquid)

            # step 3 : draw user interface
            self.draw_handler.draw_ui(start_button=self.buttonStart, number=max(0, self.number))

            # step 4 : draw buttons interfaces, reset button depends on the mode
            self.buttonStart.draw()
            if Glob.mode == 1 and self.buttonStart.is_ready():
                if self.new:
                    self.number += 1
                    self.draw_handler.q += Glob.brick_array.current_steel_volume()
                    self.new = False
                self.buttonReset.unpause()
                self.buttonReset.draw()
            else:
                self.buttonReset.pause()
                self.new = True

    def check_buttons(self) -> None:
        """ Update button image and read button state """

        # Set image to the newest one
        self.buttonStart.image = self.cam.image_raw
        self.buttonReset.image = self.cam.image_raw

        # Change mode with button state
        if Glob.mode == 0 and Glob.brick_array is not None:
            self.triggered_number = self.buttonStart.number

            if self.buttonStart.is_triggered:
                Glob.mode = 1
                Glob.brick_array.init_heat_eq()
            pass

            if self.buttonReset.is_triggered:
                Glob.mode = 0
                return
            pass

    def detect_brick(self):
        """ Execute brick detection tools """
        frame = self.cam.image_raw
        if frame is not None:
            image_1, image_2 = self.brick_recognition.update_bricks(frame.copy())

            # if we are calibrating print brick map on the screen (upper left corner)
            if image_1 is not None:
                if Glob.debug:
                    texture = cv2.resize(image_1, (Conf.width, Conf.height))
                    mask = np.zeros(texture.shape, dtype=np.uint8)
                    grid_color = (0, 255, 0, 255)
                    start_color = (0, 0, 255, 255)
                    reset_color = (255, 0, 0, 255)
                    mask[Conf.cam_area[0][0]:Conf.cam_area[1][0],
                    Conf.cam_area[0][1]:Conf.cam_area[1][1]] = grid_color
                    mask[Conf.hand_area_1[0][0]:Conf.hand_area_1[1][0],
                    Conf.hand_area_1[0][1]:Conf.hand_area_1[1][1]] = start_color
                    mask[Conf.hand_area_2[0][0]:Conf.hand_area_2[1][0],
                    Conf.hand_area_2[0][1]:Conf.hand_area_2[1][1]] = reset_color

                    texture = cv2.addWeighted(texture, 0.75, mask, 0.25, 0)
                    # texture = np.mean([texture, mask], axis=0, dtype=np.uint8)
                    self.tex_handler.bind_texture(0, cv2.flip(texture, 0), Conf.width, Conf.height)

                    texture = cv2.resize(image_2, (Conf.width, Conf.height))
                    self.tex_handler.bind_texture(5, cv2.flip(texture, 0), Conf.width, Conf.height)
                Glob.frame = image_1

    def lost_screen(self):
        """ Draw a message on the screen """
        self.draw_handler.draw_text_screen()


# Language Selection Scene
class CalibrationAR(AugmentedReality):
    def __init__(self, cam) -> None:
        super().__init__(cam)
        self.tex_handler = TextureHandler(3)
        self.image_path = "./ressources/calib.png"
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        self.ref = cv2.resize(image, (Conf.width, Conf.height))
        image = cv2.flip(self.ref, 0)

        glEnable(GL_TEXTURE_2D)
        self.tex_handler.bind_texture(2, image, image.shape[1], image.shape[0])
        glDisable(GL_TEXTURE_2D)

        self.timer = clock()

    def calibrate(self):
        if clock() - self.timer < 5:
            return False

        self.cam.take_frame()

        frame = cv2.cvtColor(self.cam.image, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(frame, None)
        if des1 is None or des2 is None:
            return False

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > 50:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            Glob.homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            print("Calibration succeed, %i points recognized" % len(good))
            return True
        else:
            return False

    def render(self):
        glut_print(20, 20, GLUT_BITMAP_HELVETICA_18, "LANGUAGE", 1, 0, 1)
        glEnable(GL_TEXTURE_2D)
        self.tex_handler.use_texture(2)
        draw_textured_rectangle(0, 0, Conf.width, Conf.height)
        glDisable(GL_TEXTURE_2D)



class VideoAR(AugmentedReality):
    def __init__(self, cam) -> None:
        super().__init__(cam)
        self.tex_handler = TextureHandler(3)
        self.draw_handler = DrawingHandler(self.tex_handler)
        self.video_path = "./ressources/poche.mp4"
        self.video_cap = cv2.VideoCapture(self.video_path)

        # Create a handler for every drawing functions
        self.draw_handler = DrawingHandler(self.tex_handler, None, None)
        # Create button Handler and start them for image detection
        self.buttonJeu = None
        self.init_start_buttons()
        self.next_frame = np.zeros((Conf.width, Conf.height, 4), np.uint8)
        self.tex_handler.bind_texture(1, self.next_frame, Conf.width, Conf.height)

    def next_video_frame(self):
        if self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = cv2.resize(frame, (Conf.width, Conf.height))
                self.next_frame = cv2.flip(frame, 0)

    def render_video(self):
        glut_print(20, 20, GLUT_BITMAP_HELVETICA_18, "VIDEO", 1, 0, 0)

        if self.next_frame is not None:
            glEnable(GL_TEXTURE_2D)
            self.tex_handler.bind_texture(1, self.next_frame, Conf.width, Conf.height)
            self.tex_handler.use_texture(1)
            draw_textured_rectangle(0, 0, Conf.width - 200, Conf.height - 100)
            glDisable(GL_TEXTURE_2D)

    def init_start_buttons(self):
        self.buttonJeu = HandButton(0, None, 2, Conf.hand_area_8, Conf.hand_threshold_1)
        self.buttonJeu.daemon = True
        self.buttonJeu.start()
        self.buttonJeu.title = "Passer au jeu"

    def render(self) -> None:
        """ render the scene with OpenGL"""
        # step 1 : draw buttons interfaces, reset button depends on the mode
        draw_rectangle(0, 0, Conf.width, Conf.height, 0.2, 0.2, 0.2)
        self.buttonJeu.draw()
        self.render_video()

    def check_buttons(self) -> bool:
        """ Update button image and read button state """
        # Set image to the newest one
        self.buttonJeu.image = self.cam.image_raw
        if self.buttonJeu.is_triggered:
            return True
        return False


# Difficulty Scene
class DifficultyAR(AugmentedReality):

    def __init__(self, cam) -> None:
        super().__init__(cam)

        # Create a texture handler with 6 different textures
        self.tex_handler = TextureHandler(8)

        # Create a handler for every drawing functions
        self.draw_handler = DrawingHandler(self.tex_handler, None, None)

        # Create button Handler and start them for image detection
        self.buttonCorrosion, self.buttonMecanique, self.buttonThermique, self.buttonMulti, self.buttonValider, self.jeuexplication = None, None, None, None, None, None
        self.init_start_buttons()

        self.active_button = None

    def init_start_buttons(self):
        self.buttonCorrosion = HandButton(1, None, 2, Conf.hand_area_3, Conf.hand_threshold_1)
        self.buttonMecanique = HandButton(1, None, 3, Conf.hand_area_4, Conf.hand_threshold_2)
        self.buttonThermique = HandButton(1, None, 7, Conf.hand_area_5, Conf.hand_threshold_2)
        self.buttonMulti = HandButton(1, None, 5, Conf.hand_area_6, Conf.hand_threshold_2)
        self.buttonValider = HandButton(1, None, 6, Conf.hand_area_7, Conf.hand_threshold_2)
        self.buttonjeuexplication = HandButton(1, None, 1, Conf.hand_area_9, Conf.hand_threshold_1)

        self.buttonCorrosion.daemon = True
        self.buttonMecanique.daemon = True
        self.buttonThermique.daemon = True
        self.buttonMulti.daemon = True
        self.buttonValider.daemon = True
        self.buttonjeuexplication.daemon = True

        self.buttonCorrosion.start()
        self.buttonMecanique.start()
        self.buttonThermique.start()
        self.buttonMulti.start()
        self.buttonValider.start()
        self.buttonjeuexplication.start()

        Glob.physics = []

        self.buttonCorrosion.title = "Thermique"
        self.buttonMecanique.title = "Mécanique + Thermique"
        self.buttonThermique.title = "Corrosion + Thermique"
        self.buttonMulti.title = "Corrosion + Thermique + Mécanique"
        self.buttonValider.title = "Valider"
        self.buttonjeuexplication.title = "Voir les explications"

    def render(self) -> None:
        """ render the scene with OpenGL"""
        # step 1 : draw buttons interfaces, reset button depends on the mode
        draw_rectangle(0, 0, Conf.width, Conf.height, 0.2, 0.2, 0.2)
        self.buttonCorrosion.draw()
        self.buttonMecanique.draw()
        self.buttonThermique.draw()
        self.buttonMulti.draw()
        self.buttonValider.draw()
        self.buttonjeuexplication.draw()
        glut_print(20, 650, GLUT_BITMAP_HELVETICA_18, "SELECTIONNEZ LE CAS AVEC LEQUEL VOUS VOULEZ JOUER", 1, 1, 1)

    def check_buttons(self) -> (bool, bool):
        """ Update button image and read button state """

        # Set image to the newest one
        self.buttonCorrosion.image = self.cam.image_raw
        self.buttonMecanique.image = self.cam.image_raw
        self.buttonThermique.image = self.cam.image_raw
        self.buttonMulti.image = self.cam.image_raw
        self.buttonValider.image = self.cam.image_raw
        self.buttonjeuexplication.image = self.cam.image_raw

        if self.buttonCorrosion.is_triggered:
            Glob.physics = ["Thermique"]
            if self.active_button is not None:
                self.active_button.unpause()
            self.buttonCorrosion.pause()
            self.active_button = self.buttonCorrosion

        if self.buttonMecanique.is_triggered:
            Glob.physics = ["Mécanique", "Thermique"]
            if self.active_button is not None:
                self.active_button.unpause()
            self.buttonMecanique.pause()
            self.active_button = self.buttonMecanique

        if self.buttonThermique.is_triggered:
            Glob.physics = ["Corrosion", "Thermique"]
            if self.active_button is not None:
                self.active_button.unpause()
            self.buttonThermique.pause()
            self.active_button = self.buttonThermique

        if self.buttonMulti.is_triggered:
            Glob.physics = ["Corrosion", "Thermique", "Mécanique"]
            if self.active_button is not None:
                self.active_button.unpause()
            self.buttonMulti.pause()
            self.active_button = self.buttonMulti

        print(Glob.physics)

        if self.buttonValider.is_triggered and len(Glob.physics) > 0:
            return True, True
        if self.buttonjeuexplication.is_triggered:
            return True, False

        return False, False


    # change to game


class ExplicationAR(AugmentedReality):
    def __init__(self, cam) -> None:
        super().__init__(cam)
        self.tex_handler = TextureHandler(6)
        self.draw_handler = DrawingHandler(self.tex_handler)
        self.video_path = "./ressources/poche.mp4"
        self.video_cap = cv2.VideoCapture(self.video_path)

        # Create a handler for every drawing functions
        self.draw_handler = DrawingHandler(self.tex_handler, None, None)
        # Create button Handler and start them for image detection
        self.buttonJeu = None
        self.init_start_buttons()
        self.next_frame = np.zeros((Conf.width, Conf.height, 4), np.uint8)
        self.tex_handler.bind_texture(1, self.next_frame, Conf.width, Conf.height)

    def next_video_frame(self):
        if self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = cv2.resize(frame, (Conf.width, Conf.height))
                self.next_frame = cv2.flip(frame, 0)

    def render_video(self):
        glut_print(20, 20, GLUT_BITMAP_HELVETICA_18, "Explication", 1, 0, 1)

        if self.next_frame is not None:
            glEnable(GL_TEXTURE_2D)
            self.tex_handler.bind_texture(1, self.next_frame, Conf.width, Conf.height)
            self.tex_handler.use_texture(1)
            draw_textured_rectangle(0, 0, Conf.width - 200, Conf.height - 100)
            glDisable(GL_TEXTURE_2D)

    def init_start_buttons(self):
        self.buttonJeu = HandButton(1, None, 2, Conf.hand_area_8, Conf.hand_threshold_1)
        self.buttonJeu.daemon = True
        self.buttonJeu.start()
        self.buttonJeu.title = "Passer au jeu"

    def render(self) -> None:
        """ render the scene with OpenGL"""
        # step 1 : draw buttons interfaces, reset button depends on the mode
        draw_rectangle(0, 0, Conf.width, Conf.height, 0.2, 0.2, 0.2)
        self.buttonJeu.draw()
        self.render_video()

    def check_buttons(self) -> bool:
        """ Update button image and read button state """
        # Set image to the newest one
        self.buttonJeu.image = self.cam.image_raw
        if self.buttonJeu.is_triggered:
            return True
        return False


class BrickRecognition:
    """ Detect bricks from webcam frame """

    def __init__(self, liquid_im):
        # keep dynamic array in class
        self.liquid_im = liquid_im
        self.frame_count = 0

    def update_bricks(self, image_raw) -> (np.ndarray, list):
        """ Process the current raw frame and return a texture to draw and the  detected bricks"""
        image, im = None, None
        self.frame_count += 1
        # update each n frames
        if Glob.mode == 0 and self.frame_count % Conf.refresh_rate == 0:  # building mode
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGBA)  # convert to RGBA
            # image_raw = adjust_gamma(image_raw.copy(), 1 if calibration else 2)  # enhance gamma of the image

            if False:  # future work : no grid mode
                bricks, image = self.discretise_area(image_raw)
            else:
                im = image_raw.copy()
                # Set almost white to white
                image_raw[np.std(image_raw[:, :, :3], axis=2) < 12.0] = [255, 255, 255, 255]

                contours, image = find_contours(image_raw)  # find contours in the center of the image
                bricks = self.isolate_bricks(contours, image)  # detect bricks

            # Make image opaque
            image[:, :, 3] = 255 * np.ones((image.shape[0], image.shape[1]))

            # Initialise brick array if needed
            if Glob.brick_array is None or len(Glob.brick_array.array) == 0:
                Glob.brick_array = BrickArray(bricks, self.liquid_im)

            else:
                # Update array with new bricks if needed
                Glob.brick_array.invalidate()
                for nb in bricks:
                    for index in nb.indexes:
                        if index == [0, 0]:
                            Glob.brick_array.set(0, 0, Brick.void(nb.indexes))
                            continue

                        prev_b = Glob.brick_array.get(index[0], index[1])
                        if prev_b is not None and prev_b.is_almost(nb):
                            prev_b.replace()

                        else:
                            # print("Brick added: " + str(nb.indexes[0]) + " (%s)" % nb.material.color)
                            Glob.brick_array.set(index[0], index[1], nb)

                Glob.brick_array.clear_invalid()

        return im, image

    @staticmethod
    def discretise_area(image):
        """ not implemented yet, no grid mode """

        # raise NotImplementedError

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh_rgb = blurred.copy()

        # for each color channel
        for index, channel in enumerate(cv2.split(thresh_rgb)):
            # make channel binary with an adaptive threshold
            thresh = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 103, 8)
            # Structure the channel with Rectangle shapes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # merge channels
            thresh_rgb[:, :, index] = thresh

        # convert into Gray scale(luminance)
        thresh_gray = cv2.cvtColor(thresh_rgb, cv2.COLOR_RGB2GRAY)
        # zoom in the center
        thresh_gray = crop_cam_area(thresh_gray)
        image = crop_cam_area(image)
        # invert black/white
        thresh_gray = cv2.bitwise_not(thresh_gray)

        kernel = np.ones((5, 5), np.uint8)
        thresh_gray = cv2.erode(thresh_gray, kernel, iterations=10)

        bricks = []

        step_x = image.shape[1] / Conf.dim_grille[0]
        step_y = image.shape[0] / Conf.dim_grille[1]

        # import matplotlib.pyplot as plt
        # plt.subplots(Conf.dim_grille[1], Conf.dim_grille[0])

        image_hls = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HLS)

        for x in range(Conf.dim_grille[0]):
            for y in range(Conf.dim_grille[1]):
                crop = image_hls[int(y * step_y):int((y + 1) * step_y),
                       int(x * step_x):int((x + 1) * step_x)]
                mean = 2 * np.mean(crop, axis=(0, 1))[0]

                if mean > 20:
                    colors = list(Conf.color_dict.keys())
                    closest_colors = sorted(colors, key=lambda color: np.abs(color - mean))
                    closest_color = closest_colors[0]

                    b = Brick.new([[x * step_x, y * step_y], [step_x, step_y], 0], closest_color)
                    bricks.append(b)

                    c_rgb = cv2.cvtColor(np.uint8([[[0.5 * closest_color, 128, 255]]]), cv2.COLOR_HLS2RGB)[0][0]
                    image = cv2.rectangle(image,
                                          (int(x * step_x), int(y * step_y)),
                                          (int((x + 1) * step_x), int((y + 1) * step_y)),
                                          tuple([int(n) for n in c_rgb]), thickness=10)
                    # cv2.putText(image, "%.2f" % mean, (int(x * step_x), int((y + .5) * step_y)),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1.5, (0, 0, 0), thickness=10, lineType=10)

        # plt.show()

        return bricks, image

    @staticmethod
    def isolate_bricks(contours: list, image: np.ndarray) -> list:
        """ create bricks from contours"""
        bricks = []

        # loop over the contours
        for index, c in enumerate(contours):
            # filter with Area
            area = cv2.contourArea(c)
            if area > Conf.max_brick_size or area < Conf.min_brick_size:
                continue

            # compute the center of the contour
            m = cv2.moments(c)
            if m["m00"] == 0:
                break
            c_x = int((m["m10"] / m["m00"]))
            c_y = int((m["m01"] / m["m00"]))

            # get a rotated rectangle from the contour
            rect = cv2.minAreaRect(c)
            # get vertices
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Create a mask with the rectangle
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [box], -1, (255, 255, 255, 255), -1)

            # Shrink the mask to make sure that we are only inside the rectangle
            # kernel = np.ones((10, 10), np.uint8)
            # mask = cv2.erode(mask, kernel, iterations=4)

            # convert image in Hue Lightness Saturation space, better for taking mean color.
            image_hls = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HLS)
            # take the mean hue
            mean = 2 * cv2.mean(image_hls, mask=mask)[0]

            # Compare it to the colors dict and find the closest one
            colors = list(Conf.color_dict.keys())
            closest_colors = sorted(colors, key=lambda color: np.abs(color - mean))
            closest_color = closest_colors[0]
            # name_color = Conf.color_dict[closest_color]

            # draw a bright color with the same hue
            c_rgb = cv2.cvtColor(np.uint8([[[0.5 * closest_color, 128, 255]]]), cv2.COLOR_HLS2RGB)[0][0]
            cv2.drawContours(image, [box], 0, tuple([int(n) for n in c_rgb]), thickness=cv2.FILLED)

            # create a brick and add it to the array
            b = Brick.new(rect, closest_color)
            cv2.putText(image, "%0.0f" % mean, (c_x - 20, c_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), thickness=10, lineType=10)
            cv2.putText(image, "%i%i" % (tuple(b.indexes[0])), (c_x - 20, c_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 0), thickness=10, lineType=10)

            bricks.append(b)

        return bricks


class DrawingHandler:
    """ Handle all drawing with OpenGL
        control shaders and textures
    """

    def __init__(self, texture_handler, q_activate: SimpleQueue = None, liquid_image=None):

        self.tex_handler = texture_handler
        self.offset_x, self.offset_y = 0, 0
        # self.shader_handler_molten_steel = ShaderHandler("./shader/LavaShader",
        #                                                  {"myTexture": None, "offset_x": float, "offset_y": float,
        #                                                   "delta_t": float, "height": float})
        if q_activate is not None:
            self.q_activate = q_activate
            with liquid_image.get_lock():
                self.liquid_image = np.frombuffer(liquid_image.get_obj())

            self.shader_handler_brick = ShaderHandler("./shader/TemperatureShader",
                                                      {"Corrosion": float, "temp_buffer": GL_SHADER_STORAGE_BUFFER,
                                                       "brick_dim": np.ndarray, "grid_pos": int, "step": float,
                                                       "border": float, "color_meca": np.ndarray})

        self.steel_flow_dir_x, self.steel_flow_dir_y = 1, 1
        self.shader_clock = clock()
        self.q, self.old_q = 0, 0
        self.previous_number = 0

        k_factor = 3
        self.kernel = np.ones((k_factor, k_factor), np.float32) / (k_factor * k_factor)

    def draw_molten_steel(self, x_s: int, y_s: int, w: int, h: int,
                          r: float, g: float, b: float, active) -> None:
        """ draw "molten_steel" from a texture"""

        draw_rectangle(x_s - 20, y_s, w + 20, h, 0.1, 0.1, 0.1)
        if Glob.mode == 0:
            draw_rectangle(x_s, y_s, w, h, 1, 1, 1)

        # self.shader_handler_molten_steel.bind(data={"myTexture": None, "offset_x": self.offset_x,
        #                                             "offset_y": self.offset_y, "delta_t": delta_t, "height": 0})
        # update offset from clock to scroll the texture
        glEnable(GL_TEXTURE_2D)
        self.tex_handler.use_texture(2)
        glColor(r, g, b, 1)

        if self.q_activate.empty():
            self.q_activate.put(active)

        if Glob.mode == 1:
            image = np.reshape(self.liquid_image.copy(), (10 * Conf.dim_grille[1], 10 * (Conf.dim_grille[0] + 1), 4))

            # smooth water image
            image[:, :, :] = cv2.filter2D(image[:, :, :], -1, self.kernel)
            image = cv2.flip(image, 0)

            # not needed, opengl can do it by itself
            # image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            # glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0],
            #                 GL_RGBA, GL_FLOAT, cv2.flip(image.astype(np.float32), 0))

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0],
                         0, GL_RGBA, GL_FLOAT, cv2.flip(image.astype(np.float32), 0))
            # draw_textured_rectangle(0, 0, image.shape[1], image.shape[0])
            draw_textured_rectangle(x_s, y_s, (Conf.dim_grille[0] + 1) * w / Conf.dim_grille[0], h)

        glDisable(GL_TEXTURE_2D)

        # self.shader_handler_molten_steel.unbind()

    @staticmethod
    def draw_thermal_diffusivity(x_s: int, y_s: int) -> None:
        glut_print(0, 100, GLUT_BITMAP_HELVETICA_18, "Resistance thermique", *Conf.text_color)
        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]
        step_x = (xf - x0) / Conf.dim_grille[0]
        step_y = (y0 - 10) / Conf.dim_grille[1]
        for c in range(Conf.dim_grille[0]):
            for l in range(Conf.dim_grille[1]):
                b_xy = Glob.brick_array.get(c, l)
                t = b_xy.material.diffusivity * 1E6
                if not b_xy.is_void:
                    color = b_xy.material.color
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, *color)
                else:
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, 0, 0, 0)

                if b_xy is not None:
                    if not b_xy.is_void:
                        dict_values = {0: "+ + +", 1.2: "+ +", 2.4: "+", 3.6: "-", 4.8: "- -", 6: "- - -"}
                        tmp = list(dict_values.keys())
                        closest_value = sorted(tmp, key=lambda v: np.abs(v - t))
                        txt = dict_values[closest_value[0]]
                        glut_print(x_s + c * step_x + .5 * step_x - 2.5 * len(txt),
                                   (Conf.dim_grille[1] - l - 1) * step_y + .3 * step_y, GLUT_BITMAP_HELVETICA_18, txt,
                                   1, 1, 1)

    @staticmethod
    def draw_thermal_diffusivity_corr(x_s: int, y_s: int) -> None:
        glut_print(0, 100, GLUT_BITMAP_HELVETICA_18, "Resistances", *Conf.text_color)
        glut_print(0, 80, GLUT_BITMAP_HELVETICA_18, "thermique / corrosion", *Conf.text_color)
        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]
        step_x = (xf - x0) / Conf.dim_grille[0]
        step_y = (y0 - 10) / Conf.dim_grille[1]
        for c in range(Conf.dim_grille[0]):
            for l in range(Conf.dim_grille[1]):
                b_xy = Glob.brick_array.get(c, l)
                r_diffus = b_xy.material.diffusivity * 1E6
                if not b_xy.is_void:
                    color = b_xy.material.color
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, *color)
                else:
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, 0, 0, 0)

                if b_xy is not None:
                    if not b_xy.is_void:
                        dict_values = {0: "+ + +", 1.2: " + + ", 2.4: "  +  ", 3.6: "  -  ", 4.8: " - - ", 6: "- - -"}
                        tmp = list(dict_values.keys())
                        closest_value = sorted(tmp, key=lambda v: np.abs(v - r_diffus))
                        txt = dict_values[closest_value[0]]
                        dict_values = {0: "- - -", 0.2: " - - ", 0.4: "  -  ", 0.6: "  +  ", 0.8: " + + ", 1: "+ + +"}
                        tmp = list(dict_values.keys())
                        closest_value = sorted(tmp, key=lambda v: np.abs(v - b_xy.material.r_cor))
                        txt += " / " + dict_values[closest_value[0]]
                        glut_print(x_s + c * step_x + .5 * step_x - 3 * len(txt),
                                   (Conf.dim_grille[1] - l - 1) * step_y + .3 * step_y, GLUT_BITMAP_HELVETICA_18, txt,
                                   1, 1, 1)

    @staticmethod
    def draw_resistance_corr(x_s: int, y_s: int) -> None:
        glut_print(0, 100, GLUT_BITMAP_HELVETICA_18, "Resistances à la", *Conf.text_color)
        glut_print(0, 80, GLUT_BITMAP_HELVETICA_18, "corrosion", *Conf.text_color)
        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]
        step_x = (xf - x0) / Conf.dim_grille[0]
        step_y = (y0 - 10) / Conf.dim_grille[1]
        for c in range(Conf.dim_grille[0]):
            for l in range(Conf.dim_grille[1]):
                b_xy = Glob.brick_array.get(c, l)
                t = b_xy.material.r_cor if b_xy is not None else np.nan
                if not np.isnan(t) and not b_xy.is_void:
                    color = b_xy.material.color
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, *color)
                else:
                    draw_rectangle(x_s + c * step_x, (Conf.dim_grille[1] - l - 1) * step_y, step_x, step_y, 0, 0, 0)

                if b_xy is not None and not b_xy.drowned and not b_xy.is_void:
                    if b_xy.material.r_cor != -1:
                        dict_values = {0: "- - -", 0.2: "- -", 0.4: "-", 0.6: "+", 0.8: "+ +", 1: "+ + +"}
                        tmp = list(dict_values.keys())
                        closest_value = sorted(tmp, key=lambda v: np.abs(v - b_xy.material.r_cor))
                        txt = dict_values[closest_value[0]]

                        # txt = "%0.0f%%" % (100.0 * b_xy.material.r_cor)

                        glut_print(x_s + c * step_x + .5 * step_x - 2.5 * len(txt),
                                   (Conf.dim_grille[1] - l - 1) * step_y + .3 * step_y, GLUT_BITMAP_HELVETICA_18, txt,
                                   1, 1, 1)

    def draw_temperatures(self, x_s: int, y_s: int, bricks, start_button: HandButton) -> None:
        glut_print(0, 100, GLUT_BITMAP_HELVETICA_18, "Temperatures", *Conf.text_color)

        _range = 2 if Glob.mode == 1 else 1
        for i in range(_range):
            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]
            h = 20
            a = 1

            if i == 1:
                w, h = xf - x0, (yf - y0) / Conf.dim_grille[1]
                x_s, y_s = x0, y0
                a = 0

            # request temperature buffer update
            self.shader_handler_brick.invalidate_buffer()

            step = int((xf - x0) / Conf.dim_grille[0])
            for index_c in range(Conf.dim_grille[0]):
                for index_l in range(Conf.dim_grille[1]):
                    b_xy: Brick = bricks.get(index_c, index_l)
                    if (not b_xy.is_void and not b_xy.material.is_broken) or i == 0:
                        index = b_xy.indexes[0]
                        temp_array = np.array(Glob.brick_array.T, dtype=ctypes.c_int32)
                        pos = (index[1]) * Glob.brick_array.step_x * (temp_array.shape[1]) + index[
                            0] * Glob.brick_array.step_y

                        border = 0
                        if index[0] < Conf.dim_grille[0] - 1:
                            border += 1
                        if index[1] < Conf.dim_grille[1] - 1:
                            border += 2

                        my_color_meca = [1, 1, 1, 1]

                        corrosion = b_xy.material.health if "Corrosion" in Glob.physics else 1
                        self.shader_handler_brick.bind({"Corrosion": corrosion if i == 1 else 1,
                                                        "temp_buffer": temp_array.flatten(),
                                                        "brick_dim": [Glob.brick_array.step_x, Glob.brick_array.step_y],
                                                        "grid_pos": pos, "step": Glob.brick_array.nx,
                                                        "border": border, "color_meca": my_color_meca})

                        # gradient
                        draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h, step, h)

                        self.shader_handler_brick.unbind()

                        # do not update buffer in the next iterations
                        self.shader_handler_brick.fix_buffer()

                        # draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h, step, h, 1, 1, 1)
                        # Affiche la tempe sur les briques
                        # message = "%0.0f °C" % temp
                        # glut_print(x_s + index_c * step + .5 * step - 2.5 * len(message),
                        # y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5,
                        # GLUT_BITMAP_HELVETICA_12,
                        # message, 1, 1, 1)

                        """glut_print(x_s + index_c * step + .5 * step - 2.5 * len(message),
                                    y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5,
                                    GLUT_BITMAP_HELVETICA_12,
                                    message, *text_color)"""
                        t = []
                        for j in range(Conf.dim_grille[1]):
                            t.append((np.max(bricks.get_temp(Conf.dim_grille[0] - 1, j))) - 273)
                        if np.max(t) <= 0:
                            glut_print(Conf.width - 100, 600, GLUT_BITMAP_HELVETICA_18, "%0.2f °C" % (np.max(t) + 273), 1, 1, 1)
                        else:
                            glut_print(Conf.width - 100, 600, GLUT_BITMAP_HELVETICA_18, "%0.2f °C" % np.max(t), 1, 1, 1)

                        glut_print(900, 625, GLUT_BITMAP_HELVETICA_18, "T° extérieure " , 1, 1, 1)

                        if not start_button.is_ready() and "Mécanique" in Glob.physics:
                            stress = update_stress(b_xy.indexes[0][1])
                            if stress >= 3.06:
                                my_color_meca = [1, 0, 0, 1]
                                self.shader_handler_brick.bind({"Corrosion": b_xy.material.health if i == 1 else 1,
                                                                "temp_buffer": temp_array.flatten(),
                                                                "brick_dim": [Glob.brick_array.step_x,
                                                                              Glob.brick_array.step_y],
                                                                "grid_pos": pos, "step": Glob.brick_array.nx,
                                                                "border": border,
                                                                "color_meca": my_color_meca})

                                # gradient
                                draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h, step,
                                               h)

                                self.shader_handler_brick.unbind()



                            elif 1 <= stress < 3.06:
                                my_color_meca = [1, 0.5, 0, 1]
                                self.shader_handler_brick.bind({"Corrosion": b_xy.material.health if i == 1 else 1,
                                                                "temp_buffer": temp_array.flatten(),
                                                                "brick_dim": [Glob.brick_array.step_x,
                                                                              Glob.brick_array.step_y],
                                                                "grid_pos": pos, "step": Glob.brick_array.nx,
                                                                "border": border,
                                                                "color_meca": my_color_meca
                                                                })

                                # gradient
                                draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h, step,
                                               h)

                                self.shader_handler_brick.unbind()

                            elif 0 <= stress < 1:
                                my_color_meca = [0, 1, 0, 1]
                                self.shader_handler_brick.bind({"Corrosion": b_xy.material.health if i == 1 else 1,
                                                                "temp_buffer": temp_array.flatten(),
                                                                "brick_dim": [Glob.brick_array.step_x,
                                                                              Glob.brick_array.step_y],
                                                                "grid_pos": pos, "step": Glob.brick_array.nx,
                                                                "border": border, "color_meca": my_color_meca})

                                # gradient
                                draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h, step,
                                               h)

                                self.shader_handler_brick.unbind()

                        temp = Glob.brick_array.get_temp(index[0], index[1]) - 273
                        try:
                            temp = np.mean(temp)
                        except Exception as e:
                            temp = 0

                        size = 60, 15
                        if "Thermique" in Glob.physics:
                            text_color = (0, 0, 0)
                            if temp < 500:
                                #message = "froid"

                                #glut_print(x_s + index_c * step + .5 * step - 2.5 * len(message),
                                #          y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5,
                                #          GLUT_BITMAP_HELVETICA_12,
                                #         message, *text_color)
                                glEnable(GL_TEXTURE_2D)
                                self.tex_handler.use_texture(12)
                                draw_textured_rectangle(x_s + index_c * step + .5 * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5, size[0], size[1])
                                glDisable(GL_TEXTURE_2D)

                            elif 500 < temp < 1400:
                                #message = "chaud"

                                glEnable(GL_TEXTURE_2D)
                                self.tex_handler.use_texture(11)
                                draw_textured_rectangle(x_s + index_c * step + .5 * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5, size[0], size[1])
                                glDisable(GL_TEXTURE_2D)

                                #glut_print(x_s + index_c * step + .5 * step - 2.5 * len(message),
                                #           y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5,
                                #           GLUT_BITMAP_HELVETICA_12,
                                #          message, *text_color)

                            else:
                                #message = "brulant"


                                #glut_print(x_s + index_c * step + .5 * step - 2.5 * len(message),
                                #           y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5,
                                #           GLUT_BITMAP_HELVETICA_12,
                                #          message, *text_color)
                                glEnable(GL_TEXTURE_2D)
                                self.tex_handler.use_texture(10)
                                draw_textured_rectangle(x_s + index_c * step + .5 * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h + .5 * h - 5, size[0], size[1])
                                glDisable(GL_TEXTURE_2D)


        else:
            draw_rectangle(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h,
                           step, h, 0, 0, 0, a if b_xy.drowned else 1)

        # draw_rectangle_empty(x_s + index_c * step, y_s + (Conf.dim_grille[1] - index_l - 1) * h,
        #                      step, h, 0, 0, 0, 1)

    def draw_texture(self, tex_loc: int, _x: int, _y: int, l: int, h: int) -> None:
        glEnable(GL_TEXTURE_2D)
        self.tex_handler.use_texture(tex_loc)
        draw_textured_rectangle(_x, _y, l, h)
        glDisable(GL_TEXTURE_2D)

    @staticmethod
    def draw_frame(image: np.ndarray) -> None:
        """ Draw the frame with OpenGL as a texture in the entire screen"""
        if type(image) != int:
            # draw background
            draw_rectangle(0, 0, Conf.width, Conf.height, 0.2, 0.2, 0.2)

            y0, x0 = Conf.cam_area[0]
            yf, xf = Conf.cam_area[1]

            if Glob.mode == 1:
                draw_rectangle(x0, y0, xf - x0, yf - y0, 0, 0, 0)

    def draw_ui(self, start_button, number) -> None:
        """ Draw user interface and decoration"""

        # draw board informations
        x_start = 2.5 * Conf.width / 10

        title = strings['title_build']
        subtitle = strings['sub_title_build']
        if Glob.brick_array is not None:
            if Glob.mode == 0:
                if Conf.swap:
                    if clock() % (2 * Conf.swap_time) <= Conf.swap_time:
                        self.draw_thermal_diffusivity(Conf.cam_area[0][1], 0)
                    else:
                        self.draw_resistance_corr(Conf.cam_area[0][1], 0)
                else:
                    self.draw_thermal_diffusivity_corr(Conf.cam_area[0][1], 0)

            else:
                if number != -1:
                    print(number)
                    self.draw_temperatures(Conf.cam_area[0][1], 0, Glob.brick_array, start_button)
                    if start_button.is_ready():
                        title = strings['title_test']
                        subtitle = strings['sub_title_test1i'] % number
                        subtitle += "   Quantite d'acier: %0.2f tonne%s " % (self.q, "s" if self.q > 1 else "")
                    elif start_button.is_triggered:
                        title = "Coulée en cours"
                        subtitle = ""
                    else:
                        title = "Evacuation de l'acier liquide"
                        subtitle = ""

        # glut_print(x_start, Conf.height - 30, GLUT_BITMAP_HELVETICA_18,
        #            title + "    %0.0f fps" % (1 / Glob.delta_t), *Conf.text_color)

        glut_print(x_start, Conf.height - 30, GLUT_BITMAP_HELVETICA_18, title, *Conf.text_color)
        glut_print(x_start, Conf.height - 60, GLUT_BITMAP_HELVETICA_18, subtitle, *Conf.text_color)

        y0, x0 = Conf.cam_area[0]
        yf, xf = Conf.cam_area[1]

        # brick map
        if Glob.debug:
            ratio = .25, .2
            glEnable(GL_TEXTURE_2D)
            self.draw_texture(0, 0, (1 - ratio[1]) * Conf.height, ratio[0] * Conf.width, ratio[1] * Conf.height)
            self.draw_texture(5, (1 - ratio[0]) * Conf.width, (1 - ratio[1]) * Conf.height, ratio[0] * Conf.width,
                              ratio[1] * Conf.height)

        # draw grid
        if Glob.mode == 0:
            step_i = (xf - x0) / Conf.dim_grille[0]
            step_j = (yf - y0) / Conf.dim_grille[1]
            for i in range(Conf.dim_grille[0]):
                for j in range(Conf.dim_grille[1]):
                    pass
                    draw_rectangle_empty(x0 + i * step_i, y0 + j * step_j, step_i, step_j, 0.2, 0.2, 0.2, 2)

        glut_print(0, 400, GLUT_BITMAP_HELVETICA_18, "Pression hydrostatique", 1, 1, 1)
        glut_print(0, 300, GLUT_BITMAP_HELVETICA_18, "Vert = Pression faible", 0, 1, 0)
        glut_print(0, 350, GLUT_BITMAP_HELVETICA_18, "Orange = Pression", 1, 0.5, 0)
        glut_print(0, 325, GLUT_BITMAP_HELVETICA_18, "         moyenne", 1, 0.5, 0)
        glut_print(0, 375, GLUT_BITMAP_HELVETICA_18, "Rouge = Pression forte", 1, 0, 0)
        glut_print(0, 250, GLUT_BITMAP_HELVETICA_18, "Attention la température ", 1, 1, 1)
        glut_print(0, 225, GLUT_BITMAP_HELVETICA_18, "extérieure ne doit pas ", 1, 1, 1)
        glut_print(0, 200, GLUT_BITMAP_HELVETICA_18, "être supérieure à 400°C", 1, 1, 1)
        glut_print(200, 650, GLUT_BITMAP_HELVETICA_18, "Sens de coulée ", 1, 1, 1)
        glut_print(200, 615, GLUT_BITMAP_HELVETICA_18, "    V V V      ", 1, 1, 1)

    def draw_text_screen(self):
        texture = np.zeros((Conf.height, Conf.width, 4), np.uint8)
        texture[..., 3] = 240

        if self.q != 0:
            self.old_q = self.q
            self.q = 0

        text_1 = "Fin du test %s" % Glob.death_text
        text_2 = "Quantite d'acier : %0.2f tonne%s " % (self.old_q, 's' if self.old_q > 1 else '')
        scale = 1
        thickness = 4

        text_size_1 = cv2.getTextSize(text_1, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        text_size_2 = cv2.getTextSize(text_2, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]

        text_x_1 = (texture.shape[1] - text_size_1[0]) // 2
        text_y_1 = (texture.shape[0] + text_size_1[1]) // 2
        texture = cv2.putText(texture, text_1, (text_x_1, text_y_1),
                              cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255, 255), thickness=thickness, lineType=10)

        text_x_2 = (texture.shape[1] - text_size_2[0]) // 2
        text_y_2 = (texture.shape[0] + text_size_2[1]) // 2 + 2 * text_size_1[1]
        texture = cv2.putText(texture, text_2, (text_x_2, text_y_2),
                              cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255, 255), thickness=thickness, lineType=10)

        glEnable(GL_TEXTURE_2D)
        self.tex_handler.bind_texture(4, cv2.flip(texture, 0), Conf.width, Conf.height)
        self.tex_handler.use_texture(4)
        self.draw_texture(4, 0, 0, Conf.width, Conf.height)


class Camera:
    """ Control webcam and send webcam image"""

    def __init__(self, width: int, height: int) -> None:
        self.capture = cv2.VideoCapture(Conf.cam_number)
        if not self.capture.isOpened():
            raise IOError("Webcam not plugged or wrong webcam index")
        time.sleep(.5)
        # self.capture.set(3, width)
        # self.capture.set(4, height)
        # self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, .25)
        # self.capture.set(cv2.CAP_PROP_EXPOSURE, -2)
        # self.capture.set(cv2.CAP_PROP_BRIGHTNESS, -10)
        _, self.image_raw = self.capture.read()

    def take_frame(self) -> None:
        """ Update current raw frame in BGR format"""
        if Glob.homography is None:
            ret, self.image = self.capture.read()
            self.image_raw = cv2.resize(self.image, (Conf.width, Conf.height))

        else:
            ret, image = self.capture.read()
            cv2.imwrite("rawframe.png", image)
            M = Glob.homography
            self.image_raw = cv2.warpPerspective(image, np.linalg.inv(M), (Conf.width, Conf.height),
                                                 borderValue=(255, 255, 255))
            cv2.imwrite("lastframe.png", self.image_raw)

        if not ret:
            print("camera capture failed")
