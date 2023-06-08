import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_cl


def beale(x1, x2):
    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2**2) ** 2
        + (2.625 - x1 + x1 * x2**3) ** 2
    )


def dbeale_dx(x1, x2):
    dfdx1 = (
        2 * (1.5 - x1 + x1 * x2) * (x2 - 1)
        + 2 * (2.25 - x1 + x1 * x2**2) * (x2**2 - 1)
        + 2 * (2.625 - x1 + x1 * x2**3) * (x2**3 - 1)
    )
    dfdx2 = (
        2 * (1.5 - x1 + x1 * x2) * x1
        + 2 * (2.25 - x1 + x1 * x2**2) * (2 * x1 * x2)
        + 2 * (2.625 - x1 + x1 * x2**3) * (3 * x1 * x2**2)
    )
    return dfdx1, dfdx2


step_x1, step_x2 = 0.2, 0.2
x1, x2 = np.meshgrid(
    np.arange(-5, 5 + step_x1, step_x1), np.arange(-5, 5 + step_x2, step_x2)
)
Y = beale(x1, x2)


def gd_plot(x_traj=None):
    plt.figure()
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.contour(
        x1, x2, Y, levels=np.logspace(0, 6, 30), norm=plt_cl.LogNorm(), cmap=plt.cm.jet
    )
    plt.title("2D Contour Plot of Beale Function (Momentum)")
    plt.xlabel("$X_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.plot(3, 0.5, "k*", markersize=10)
    if x_traj is not None:
        x_traj = np.array(x_traj)
        plt.plot(x_traj[:, 0], x_traj[:, 1], "k-")


def SGD(x_traj, lr):
    x1, x2 = x_traj[-1]
    dfdx1, dfdx2 = dbeale_dx(x1, x2)
    x1 -= lr * dfdx1
    x2 -= lr * dfdx2
    x_traj.append((x1, x2))
    return x_traj


def Momentum(x_traj, lr, momentum, v):
    x1, x2 = x_traj[-1]
    dfdx1, dfdx2 = dbeale_dx(x1, x2)
    v1, v2 = v
    v1 = momentum * v1 - lr * dfdx1
    v2 = momentum * v2 - lr * dfdx2
    x1 += v1
    x2 += v2
    x_traj.append((x1, x2))
    return x_traj, (v1, v2)


def AdaGrad(x_traj, lr, epsilon, G):
    x1, x2 = x_traj[-1]
    dfdx1, dfdx2 = dbeale_dx(x1, x2)
    g1, g2 = G
    g1 += dfdx1**2
    g2 += dfdx2**2
    G = (g1, g2)
    x1 -= lr * dfdx1 / (g1 ** (1 / 2) + epsilon)
    x2 -= lr * dfdx2 / (g2 ** (1 / 2) + epsilon)
    x_traj.append((x1, x2))
    return x_traj, (g1, g2)


def RMSProp(x_traj, lr, momentum, epsilon, v):
    x1, x2 = x_traj[-1]
    dfdx1, dfdx2 = dbeale_dx(x1, x2)
    v1, v2 = v
    v1 = momentum * v1 + (1 - momentum) * dfdx1**2
    v2 = momentum * v2 + (1 - momentum) * dfdx2**2
    x1 -= lr * dfdx1 / (v1 ** (1 / 2) + epsilon)
    x2 -= lr * dfdx2 / (v2 ** (1 / 2) + epsilon)
    x_traj.append((x1, x2))
    return x_traj, (v1, v2)


def Adam(x_traj, lr, beta1, beta2, epsilon, m, v):
    x1, x2 = x_traj[-1]
    dfdx1, dfdx2 = dbeale_dx(x1, x2)
    m1, m2 = m
    v1, v2 = v
    m1 = beta1 * m1 + (1 - beta1) * dfdx1
    m2 = beta1 * m2 + (1 - beta1) * dfdx2
    m1 /= 1 - beta1
    m2 /= 1 - beta1
    v1 = beta2 * v1 + (1 - beta2) * dfdx1**2
    v2 = beta2 * v2 + (1 - beta2) * dfdx2**2
    v1 /= 1 - beta2
    v2 /= 1 - beta2
    x1 -= lr * m1 / (v1 ** (1 / 2) + epsilon)
    x2 -= lr * m2 / (v2 ** (1 / 2) + epsilon)
    x_traj.append((x1, x2))
    return x_traj, m, v


x_traj = [(step_x1, step_x2)]
v = (0, 0)
m = (0, 0)

for _ in range(10000):
    x_traj, m, v = Adam(
        x_traj=x_traj, lr=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8, m=m, v=v
    )
print(x_traj[-1])
gd_plot(x_traj)
plt.show()
