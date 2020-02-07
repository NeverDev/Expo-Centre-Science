from OpenGL.GL import *
from OpenGL.GLUT import *
from source.physics.liquid_equation import *
from source.image_recognition.augmented_reality import AugmentedReality, GameAR, DifficultyAR, VideoAR, LanguageAR, \
    Camera

from OpenGL.GLU import *
from source.settings.configuration import Config as Conf, Globals as Glob
import sys

from multiprocessing import SimpleQueue, Array, freeze_support
from OpenGL.arrays import numpymodule

numpymodule.NumpyHandler.ERROR_ON_COPY = True

screen_number = 1

sys.path += ['.']
OpenGL.ERROR_ON_COPY = True

import warnings

warnings.simplefilter("error")


class MainProgram:
    """ Main Class of the program, initiates everything and implements OpenGL environment"""

    def __init__(self):
        self.timer = None
        self.init_opengl()

        self.animation_clock = None
        self.lost_leak = False
        self.p_liquid = None

        # Memory shared variables
        self.q_activate, self.rst, self.lost = SimpleQueue(), SimpleQueue(), SimpleQueue()
        self.liquid_grid = Array(ctypes.c_double, (Conf.dim_grille[0] + 1) * Conf.dim_grille[1])
        self.liquid_im = Array(ctypes.c_double, 10 * (Conf.dim_grille[0] + 1) * 10 * Conf.dim_grille[1] * 4)

        self.run_liquid_process()

        self.cam = Camera(Conf.width, Conf.height)

        # Main utility class
        self.current_activity = GameAR(self.cam, Conf.width, Conf.height, self.q_activate,
                                      self.liquid_im, self.liquid_grid)
        self.current_activity = VideoAR(self.cam)

        # execute OpenGL loop forever
        self.loop()

    def init_opengl(self):
        # OpenGL setup
        glutInit(sys.argv)
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(Conf.width, Conf.height)
        glutInitWindowPosition(screen_number * 1366, 0)  # main window dim + 1
        glutCreateWindow("Poche AR")
        glutFullScreen()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutIdleFunc(self.idle)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def run_liquid_process(self):
        # Liquid control Process
        self.p_liquid = Liquid(self.liquid_im, self.q_activate, self.rst, self.lost, self.liquid_grid)
        self.p_liquid.daemon = True
        self.p_liquid.start()

    def idle(self):
        """ Opengl routine function, called each loop iteration"""

        Glob.delta_t = clock() - Glob.t_ref
        Glob.t_ref = clock()

        # swap between scene idle function
        if type(self.current_activity) == LanguageAR:
            self.idle_language()
        elif type(self.current_activity) == VideoAR:
            self.idle_video()
        elif type(self.current_activity) == DifficultyAR:
            self.idle_difficulty()
        elif type(self.current_activity) == GameAR:
            self.idle_game()

        # tell OpenGL to redraw as soon as possible
        glutPostRedisplay()


    def idle_video(self):

        # Timer Example
        if self.timer is None:
            self.timer = 5  # Seconds
        elif self.timer > 0:
            self.timer -= Glob.delta_t

        else:
            self.timer = None
            # Back to game scene when difficulty is over
            self.current_activity = DifficultyAR(self.cam)
        pass

    def idle_difficulty(self):

        if self.current_activity.check_buttons():
            self.current_activity = GameAR(self.cam, Conf.width, Conf.height, self.q_activate,
                                           self.liquid_im, self.liquid_grid)

        """
        # Timer Example
        if self.timer is None:
            self.timer = 10  # Seconds
        elif self.timer > 0:
            self.timer -= Glob.delta_t
            pass
        else:
            self.timer = None
            # Back to game scene when difficulty is over
            self.current_activity = GameAR(self.cam, Conf.width, Conf.height, self.q_activate,
                                           self.liquid_im, self.liquid_grid)
        """
        pass

    def idle_game(self):
        if Glob.mode == 3:
            # Change scene
            self.current_activity = DifficultyAR(self.cam)
            Glob.mode = 0

        elif Glob.mode != 2:
            # update frame from webcam
            self.current_activity.cam.take_frame()

            if Glob.mode == 0:
                # build mode, update bricks
                self.current_activity.detect_brick()

            self.current_activity.check_buttons()

            if Glob.mode == 1:
                # Testing structure mode
                if Conf.cooling:
                    Conf.t_chamber = Conf.temperature if (self.current_activity.buttonStart.is_triggered
                                                          and self.current_activity.number) > 0 \
                        else max(Conf.t_chamber - Conf.cooling_factor * Glob.delta_t, 293)
                    Glob.brick_array.update(not self.current_activity.buttonStart.is_ready()
                                            and self.current_activity.number >= 0)

                else:
                    Glob.brick_array.update(not self.current_activity.buttonStart.is_ready()
                                            and self.current_activity.number >= 0)

                lost_struct = False  # Glob.brick_array.test_loose()
                if not self.lost.empty():
                    self.lost_leak = self.lost.get()

                if (self.lost_leak or lost_struct) or self.current_activity.buttonReset.is_triggered:
                    print("reset")
                    if self.current_activity.buttonReset.is_triggered:
                        print("(from rst button)")
                    elif self.lost_leak:
                        print("(from liquid)")
                    elif lost_struct:
                        print("(from struct)")

                    self.rst.put(True)
                    self.current_activity.buttonStart.is_triggered = False
                    Glob.t_ref, Glob.delta_t = clock(), 0
                    Glob.hand_text = None

                    Glob.updating = False
                    Glob.update_timer = 0

                    self.current_activity.buttonReset.is_triggered = False
                    self.current_activity.reset()

                    Glob.mode = 2
                    self.animation_clock = clock()
                    self.current_activity.buttonStart.is_waiting = True

                    self.lost_leak = False

            # update brick grid in liquid simulation
            if Glob.brick_array is not None:
                with self.liquid_grid.get_lock():  # synchronize access
                    arr = np.frombuffer(self.liquid_grid.get_obj())  # no data copying
                    arr[:Conf.dim_grille[0] * Conf.dim_grille[1]] = Glob.brick_array.get_grid().flatten()

    def display(self):
        """ Opengl drawing function, called when an update is needed"""
        # Set Projection Matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, Conf.width, 0, Conf.height)

        # Switch to Model View Matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # swap between scene display
        if type(self.current_activity) == LanguageAR:
            self.display_language()
            pass
        elif type(self.current_activity) == VideoAR:
            self.display_video()
            pass
        elif type(self.current_activity) == DifficultyAR:
            self.display_difficulty()
            pass
        elif type(self.current_activity) == GameAR:
            self.display_Game()

        glutSwapBuffers()

    def display_language(self):
        # TODO render language screen here
        self.current_activity: LanguageAR
        self.current_activity.render()  # look at LanguageAR.render()

    def display_video(self):
        # TODO render video screen here
        self.current_activity: VideoAR
        self.current_activity.render()  # look at VideoAR.render()

    def display_difficulty(self):
        # TODO render difficulty screen here
        self.current_activity: DifficultyAR
        self.current_activity.render()  # look at DifficultyAR.render()

    def display_Game(self):
        self.current_activity: GameAR
        self.current_activity.render()  # look at GameAR.render()
        if Glob.mode == 2:
            # reset mode
            self.current_activity.buttonStart.pause()
            self.current_activity.lost_screen()

            if clock() - self.animation_clock > 5:
                Glob.mode = 3
        else:
            self.current_activity.buttonStart.unpause()
        pass

    @staticmethod
    def reshape(w, h):
        """ OpenGl function, change windows property when window's size changes"""
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if w <= h:
            glOrtho(-Glob.nRange, Glob.nRange, -Glob.nRange * h / w, Glob.nRange * h / w, -Glob.nRange, Glob.nRange)
        else:
            glOrtho(-Glob.nRange * w / h, Glob.nRange * w / h, -Glob.nRange, Glob.nRange, -Glob.nRange, Glob.nRange)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def keyboard(self, key, x, y):
        """ Opengl function, add action to keyboard input"""

        print("key_code =%s" % key)

        if key == b'\x1b':
            # kill program when escape is pressed
            self.p_liquid.terminate()
            self.p_liquid.join()
            glutLeaveMainLoop()

        if key == b'd':
            # change debug mode with d key
            Glob.debug = not Glob.debug

    def loop(self):
        """ in case we handle the loop differently """

        glutMainLoop()  # OpenGL loop, handle idle/display/reshape/keyboard calls


if __name__ == '__main__':
    freeze_support()  # needed for multiprocessing
    try:
        MainProgram()
    except Exception as e:
        raise e
    finally:
        print("Program closed")
