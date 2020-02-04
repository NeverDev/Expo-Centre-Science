
    def __init__(self, cam, width: int, height: int, q_activate: SimpleQueue, liquid_im, liquid_grid) -> None:
        super().__init__(cam)

        # Create a texture handler with 6 different textures
        self.tex_handler = TextureHandler(6)

        # Create a handler for every drawing functions
        self.draw_handler = DrawingHandler(self.tex_handler, None, None)

        # Create button Handler and start them for image detection
        self.buttonCorrosion, self.buttonMecanique, self.buttonThermique, self.buttonValider = None, None, None, None
        self.init_start_buttons()

    def init_start_buttons(self):
        self.buttonCorrosion = HandButton(0, None, 3, Conf.hand_area_3, Conf.hand_threshold_1)
        self.buttonMecanique = HandButton(1, None, 1, Conf.hand_area_4, Conf.hand_threshold_2)
        self.buttonThermique = HandButton(1, None, 1, Conf.hand_area_5, Conf.hand_threshold_2)
        self.buttonValider = HandButton(1, None, 1, Conf.hand_area_6, Conf.hand_threshold_2)

        self.buttonCorrosion.daemon = True
        self.buttonMecanique.daemon = True
        self.buttonThermique.daemon = True
        self.buttonValider.daemon = True

        self.buttonCorrosion.start()
        self.buttonMecanique.start()
        self.buttonThermique.start()
        self.buttonValider.start()

        self.buttonCorrosion.title = "Corrosion"
        self.buttonMecanique.title = "Mécanique"
        self.buttonThermique.title = "Thermique"
        self.buttonValider.title = "Valider"


    def render(self) -> None:
            """ render the scene with OpenGL"""
            # step 1 : draw buttons interfaces, reset button depends on the mode
            self.buttonCorrosion.draw()
            self.buttonMecanique.draw()
            self.buttonThermique.draw()
            self.buttonValider.draw()

            # + interface draw if needed


    def check_buttons(self) -> None:
        """ Update button image and read button state """

        # Set image to the newest one
        self.buttonCorrosion.image = self.cam.image_raw
        self.buttonMecanique.image = self.cam.image_raw
        self.buttonThermique.image = self.cam.image_raw
        self.buttonValider.image = self.cam.image_raw

        if self.buttonCorrosion.is_triggered and not "Corrosion" in Glob.physics:
            Glob.physics.append("Corrosion")
        if self.buttonMecanique.is_triggered and not "Mecanique" in Glob.physics:
            Glob.physics.append("Mecanique")
        if self.buttonThermique.is_triggered and not "Thermique" in Glob.physics:
            Glob.physics.append("Thermique")
        if self.buttonValider.is_triggered and len(Glob.physics) > 0:
            # change to game