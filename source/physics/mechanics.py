"""
#creation du tableau à mettre dans brick.py

       def get_stress(self, i, j) -> np.ndarray:
           #Get stress of a brick
           return np.flipud(self.stress[int(j * self.step_x): int((j + 1) * self.step_x),
                            int(i * self.step_y): int((i + 1) * self.step_y)])


"""
"""

import numpy as np

class Mechanics (object)

    def __init__(self, stress_value, dx, dy):

        #:param stress_value: (eg. MPa)
        #:param dx et dy: float spatial interval (eg. meters) between row elements.

        self.stress_value = stress
        self.dx = float(dx)
        self.dy = float(dy)

    def stress_calculation (self)

        if get_stress(1, j) est en contact avec du liquide or get_stress (n, j) est en contact avec du liquide:
            _stress[j, i] = 3,06

        elsif get_stress (i, 1) est en contact avec du liquide or get_stress (i, n) est en contact avec du liquide:
            _stress[j, i] = 0,7 * cordonnée en y du centre de la brique par rapport à le surface

"""