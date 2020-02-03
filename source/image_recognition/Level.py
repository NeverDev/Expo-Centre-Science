
    def __init__(self, cam, width: int, height: int, q_activate: SimpleQueue, liquid_im, liquid_grid) -> None:
        super().__init__(cam)
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
        self.tex_handler = TextureHandler(6)

        # Create a handler for every drawing functions
        self.draw_handler = DrawingHandler(self.tex_handler, q_activate, liquid_im)

        # Create a class that will extract "bricks" from image
        self.brick_recognition = BrickRecognition(liquid_im)

        # Create button Handler and start them for image detection
        self.buttonStart, self.buttonReset = None, None
        self.init_start_buttons()

    def init_start_buttons(self):
        self.buttonStart = HandButton(0, None, 3, Conf.hand_area_3, Conf.hand_threshold_1)
        self.buttonReset = HandButton(1, None, 1, Conf.hand_area_4, Conf.hand_threshold_2)
        self.buttonStart.daemon = True
        self.buttonReset.daemon = True
        self.buttonStart.start()
        self.buttonReset.start()
        self.buttonReset.title = "Corrosion"

        self.buttonStart = HandButton(0, None, 3, Conf.hand_area_4, Conf.hand_threshold_1)
        self.buttonReset = HandButton(1, None, 1, Conf.hand_area_5, Conf.hand_threshold_2)
        self.buttonStart.daemon = True
        self.buttonReset.daemon = True
        self.buttonStart.start()
        self.buttonReset.start()
        self.buttonReset.title = "Thermique"

        self.buttonStart = HandButton(0, None, 3, Conf.hand_area_1, Conf.hand_threshold_1)
        self.buttonReset = HandButton(1, None, 1, Conf.hand_area_2, Conf.hand_threshold_2)
        self.buttonStart.daemon = True
        self.buttonReset.daemon = True
        self.buttonStart.start()
        self.buttonReset.start()
        self.buttonReset.title = "Mécanique"

    def reset(self):
        """ Reset program state"""

        if self.buttonStart.number > self.draw_handler.previous_number:
            self.draw_handler.previous_number = self.buttonStart.number

        self.triggered_start, self.triggered_reset = False, False
        self.wait_time = 1


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

