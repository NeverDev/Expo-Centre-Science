
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
        # self.brick_recognition = BrickRecognition(liquid_im)

        # Create button Handler and start them for image detection
        self.buttonCorrosion, self.buttonMecanique, self.buttonThermique = None, None, None
        self.init_start_buttons()

    def init_start_buttons(self):
        self.buttonCorrosion = HandButton(0, None, 3, Conf.hand_area_3, Conf.hand_threshold_1)
        self.buttonMecanique = HandButton(1, None, 1, Conf.hand_area_4, Conf.hand_threshold_2)
        self.buttonThermique = HandButton(1, None, 1, Conf.hand_area_5, Conf.hand_threshold_2)

        self.buttonCorrosion.daemon = True
        self.buttonMecanique.daemon = True
        self.buttonThermique.daemon = True

        self.buttonCorrosion.start()
        self.buttonMecanique.start()
        self.buttonThermique.start()

        self.buttonCorrosion.title = "Corrosion"
        self.buttonMecanique.title = "Mécanique"
        self.buttonThermique.title = "Thermique"


    def reset(self):
        """ Reset program state"""

        if self.buttonStart.number > self.draw_handler.previous_number:
            self.draw_handler.previous_number = self.buttonStart.number

        self.triggered_start, self.triggered_reset = False, False
        self.wait_time = 1

        self.buttonMecanique.is_triggered, self.buttonMecanique.is_waiting = False, False
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
        self.buttonCorrosion.image = self.cam.image_raw
        self.buttonMecanique.image = self.cam.image_raw
        self.buttonThermique.image = self.cam.image_raw

        # Change mode with button state
        if Glob.mode == 0 and Glob.brick_array is not None:
            self.triggered_number = self.buttonStart.number

            if self.buttonCorrosion.is_triggered:
                Glob.mode = 3
                Glob.brick_array.init_heat_eq()
            pass

            if self.buttonMecanique.is_triggered:
                Glob.mode = 3
                return
            pass