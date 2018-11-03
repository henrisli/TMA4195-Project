# The physical interpretation of the boundary conditions is that we infuse a
# one-dimensional oil-reservoir with water from the left. u represents the 
# saturation of water. We therefore set u = 1 (100% water) at the left of
# our domain.
def inflow(h, n=0):
    if n == 0:
        h[0] = 0
    else:
        h[0:n] = 0
    return h