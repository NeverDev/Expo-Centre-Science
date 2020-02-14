# code pour les textes à mettre sur la page de la grille au milieu à gauche

glut_print(300, 100, GLUT_BITMAP_HELVETICA_14, "Pression hydrostatique", 0, 0, 0)
glut_print(320, 100, GLUT_BITMAP_HELVETICA_14, "Vert = Pression faible", 0, 1, 0)
glut_print(340, 100, GLUT_BITMAP_HELVETICA_14, "Orange = Pression moyenne", 1, 0.5, 0)
glut_print(360, 100, GLUT_BITMAP_HELVETICA_14, "Rouge = Pression forte", 1, 0, 0)
glut_print(400, 100, GLUT_BITMAP_HELVETICA_14, "Attention la température extérieure", 0, 0, 0)
glut_print(420, 100, GLUT_BITMAP_HELVETICA_14, "ne doit pas être supérieure à 400°C", 0, 0, 0)


# code pour la variable de température variable
 temperature_ext = (max(self.get_temp(Conf.dim_grille[0]-1, i)))-273
glut_print(300, 800, GLUT_BITMAP_HELVETICA_18, temperature_ext, 0, 0, 0)