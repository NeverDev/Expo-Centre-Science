"""code pour la video a remettre si besoin apres la selection de difficulté

A mettre dans augmented reality pour créer la fenêtre de la video explicative

class ExplicationAR(AugmentedReality):
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
        self.buttonJeu = HandButton(0, self.tex_handler, 2, Conf.hand_area_8, Conf.hand_threshold_1)
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









A mettre dans le main

def idle_explication(self):

    self.current_activity.cam.take_frame()
    self.current_activity.next_video_frame()

    if self.current_activity.check_buttons():
        self.current_activity = GameAR(self.cam, Conf.width, Conf.height, self.q_activate,
                                           self.liquid_im, self.liquid_grid)


def idle_difficulty(self):

    self.current_activity.cam.take_frame()

    if self.current_activity.check_buttons():
        self.current_activity = GameAR(self.cam, Conf.width, Conf.height, self.q_activate,
                                        self.liquid_im, self.liquid_grid)

Mais là je ne sais pas quoi mettre pour qu il me prenne en compte que qd le bouton "voir les explications" est activé il passe
sur la fenetre ExplicationAR









A rajouter dans difficultyAR pour ajouter ce bouton

def __init__(self, cam) -> None:
    super().__init__(cam)

    # Create a texture handler with 6 different textures
    self.tex_handler = TextureHandler(8)

    # Create a handler for every drawing functions
    self.draw_handler = DrawingHandler(self.tex_handler, None, None)

    # Create button Handler and start them for image detection
    self.buttonjeuexplication = None
    self.init_start_buttons()

    self.active_button = None

def init_start_buttons(self):
    self.buttonjeuexplication = HandButton(1, self.tex_handler, 2, Conf.hand_area_3, Conf.hand_threshold_1)

    self.buttonjeuexplication.daemon = True

    self.buttonjeuexplication.start()

    Glob.physics = []

    self.buttonjeuexplication.title = "Voir les explications"

def render(self) -> None:
    """ render the scene with OpenGL"""
    # step 1 : draw buttons interfaces, reset button depends on the mode
    draw_rectangle(0, 0, Conf.width, Conf.height, 0.2, 0.2, 0.2)
    self.buttonjeuexplication.draw()

def check_buttons(self) -> bool:
    """ Update button image and read button state """
    # Set image to the newest one
    self.buttonjeuexplication.image = self.cam.image_raw

là je sais pas comment faire pour qu il le prenne en compte comme une possibilité de changement de page donc j ai
repris comme codé pour le bouton de passage au jeu mais ça doit pas être ça

    if self.buttonjeuexplication.is_triggered:
        return True

    return False

#Ajouter dans config et configuration
hand_area_9: [[0, 250], [100, 350]]

hand_area_9 = section["hand_area_9"]


"""