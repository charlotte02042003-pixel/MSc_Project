from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, tricontour
import numpy as np

# homogenous Neumann boundary conditions

mesh = UnitSquareMesh(10, 10)
# first-order polynomial/CG
V = FunctionSpace(mesh, "P", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)

f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f, v) * dx
u = Function(V)
solve(a == L, u)

fig, axes = plt.subplots()
colors = tripcolor(u, axes=axes)
fig.colorbar(colors)
plt.savefig("helmholtz_homogeneous_neumann_heat.png")

fig, axes = plt.subplots()
contours = tricontour(u, axes=axes)
fig.colorbar(contours)
fig.savefig("helmholtz_homogeneous_neumann_contour.png")

# errors and convergence
ns = [10,20,30,40,50,60,70,80,90,100]   # mesh refinements           
hs = []
errs = []

for n in ns:
    mesh = UnitSquareMesh(n, n)
    # first-order polynomial/CG
    V = FunctionSpace(mesh, "P", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y = SpatialCoordinate(mesh)

    # exact solution
    u_exact_expression = cos(2*pi*x) * cos(2*pi*y)

    # f
    f = Function(V)
    f.interpolate((1+8*pi*pi) * u_exact_expression)

    # Weak form
    a = (inner(grad(u), grad(v)) + inner(u,v)) * dx
    L = inner(f,v) * dx

    # Solve
    uh = Function(V)
    solve(a == L, uh)

    # comparison
    u_exact = Function(V)
    u_exact.interpolate(u_exact_expression)

    # L2 error
    err = np.sqrt(assemble((uh - u_exact)**2 * dx))

    h = 1.0 / n  # mesh size
    hs.append(h)
    errs.append(err)

print("  n       h          L2 error       rate of convergence")
for i, n in enumerate(ns):
    h = hs[i]
    e = errs[i]
    if i == 0:
        q_str = "NaN"
    else:
        q = np.log(errs[i-1]/e) / np.log(hs[i-1]/h)
        q_str = f"{q:.3f}"
    print(f"{n:4d}   {h:9.6f}   {e:14.6e}   {q_str}")

# plot
plt.figure()
plt.loglog(hs, errs, marker="o")
plt.xlabel("h (= 1/n)")
plt.ylabel(r"$\|u_h - u_{\mathrm{exact}}\|_{L^2}$")
plt.title(f"Helmholtz L2 error convergence (homo Neumann)")
plt.grid(True, which="both")
plt.show()