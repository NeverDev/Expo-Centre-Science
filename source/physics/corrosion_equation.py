

def update_corrosion(health, dt, r_cor):
    return max(0.0, health - .1 * dt * (1-r_cor))
