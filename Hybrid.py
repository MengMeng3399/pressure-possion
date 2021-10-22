
import taichi as ti
# ti.init(arch=ti.cpu, debug=True,excepthook=True)
ti.init(arch=ti.gpu)

dim = 2
n_particles =4000
n_grid =64
dx = 1 / n_grid
inv_dx = 1 / dx
# dt = 2.0e-3

dt=0.004
epsilon=1e-12
rh0=1000


#使用MAC，u:水平速度，默认方向水平向右。v:竖直方向，垂直向上.

x = ti.Vector.field(dim, dtype=float, shape=n_particles) # position
v = ti.Vector.field(dim, dtype=float, shape=n_particles) # velocity

p=ti.field(dtype=float, shape=(n_grid, n_grid))  #压强

grid_u = ti.field(dtype=float, shape=(n_grid+1, n_grid)) # 水平速度
grid_v = ti.field(dtype=float, shape=(n_grid, n_grid+1)) # 竖直速度

grid_m_u = ti.field(dtype=ti.f32, shape=(n_grid+1, n_grid))
grid_m_v = ti.field(dtype=ti.f32, shape=(n_grid, n_grid+1))


#标记该网格为流体网格，还是空气网格，如果是空气网格，则设为-1，流体网格为1，固体边界为-2

Voxelized=ti.field(dtype=int, shape=(n_grid, n_grid))


@ti.func
def is_validx(i, j):
    return i >= 0 and i < n_grid+1 and j >= 0 and j < n_grid


@ti.func
def is_solidx(i, j):
    return is_validx(i, j) and Voxelized[i, j] == -2


@ti.func
def is_validy(i, j):
    return i >= 0 and i < n_grid and j >= 0 and j < n_grid+1


@ti.func
def is_solidy(i, j):
    return is_validy(i, j) and Voxelized[i, j] == -2

@ti.kernel
def enforce_boundary():
    for i, j in grid_u:
        # print(i,j)
        if i-1>=0 and i<n_grid:
            if is_solidx(i - 1, j) or is_solidx(i, j):
                grid_u[i, j] = 0.0

    for i, j in grid_v:
        if j-1>=0 and j<n_grid:
            if is_solidy(i, j - 1) or is_solidy(i, j):
                grid_v[i, j] = 0.0



@ti.func
def confine_position_to_boundary(p):
    bmin =dx
    bmax =1.0-dx
    for i in ti.static(range(dim)):
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax <= p[i]:
            p[i] = bmax - epsilon * ti.random()
    return p


@ti.kernel
def init_state():
    for i in range(n_particles):
        #粒子的初始位置范围（0.2-0.8），速度设为0.0
        x[i] = [ti.random() * 0.6 + 0.2, ti.random() * 0.6+0.2 ]

def clear_grid():
    for s in range(10):
        grid_u.fill(0)
        grid_m_u.fill(0)
        grid_v.fill(0)
        grid_m_v.fill(0)

@ti.kernel
def ParticlesToGrid():
    #分别转化 U,V。先尝试用Quadratic B-spline进行插值
    for p in x:
        # print(v[p][0])
        base_u = (x[p] * inv_dx -ti.Vector([0.0,0.5]) ).cast(int)
        base_v=(x[p] * inv_dx -ti.Vector([0.5,0.0]) ).cast(int)
        fx_u = x[p] * inv_dx - base_u.cast(float)
        fx_v = x[p] * inv_dx - base_v.cast(float)
        # Quadratic B-spline
        w_u = [0.5 * (1.5 - fx_u) ** 2, 0.75 - (fx_u - 1) ** 2, 0.5 * (fx_u - 0.5) ** 2]
        w_v = [0.5 * (1.5 - fx_v) ** 2, 0.75 - (fx_v - 1) ** 2, 0.5 * (fx_v - 0.5) ** 2]
    #     # 分别处理 U，V
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight_u = w_u[i][0] * w_u[j][1]
                weight_v = w_v[i][0] * w_v[j][1]
                index_u = base_u + ti.Vector([i, j])
                index_v = base_v + ti.Vector([i, j])
                if index_u[0] <= n_grid and  index_u[1] <= n_grid - 1:
                    grid_u[index_u] += weight_u * v[p][0]
                    grid_m_u[index_u] += weight_u

                if index_v[0] <= n_grid - 1 and index_v[1] <= n_grid:
                    grid_v[index_v] += weight_v * v[p][1]
                    grid_m_v[index_v] += weight_v

    for i, j in grid_m_u:
        if grid_m_u[i, j] > 0:
            inv_m = 1 / grid_m_u[i, j]
            grid_u[i, j] = inv_m * grid_u[i, j]

    for i, j in grid_m_v:
        if grid_m_v[i, j] > 0:
            inv_v = 1 / grid_m_v[i, j]
            grid_v[i, j] = inv_v * grid_v[i, j]



@ti.kernel
def GridOp():
    #给该系统向下的加速度，做自由落体运动,只关心竖直方向的速度.
    for i, j in grid_v:
            grid_v[i, j] -= dt * 9.8

@ti.kernel
def GridOp2():
    scale=dt/(rh0*dx)

    for i,j in ti.ndrange(n_grid, n_grid):
        if i-1>=0:
            if (Voxelized[i - 1, j] == 1 or Voxelized[i, j] == 1):
                if Voxelized[i - 1, j] == -2 or Voxelized[i, j] == -2:
                    grid_u[i, j] = 0
                else:
                    grid_u[i, j] -= scale * (p[i, j] - p[i - 1, j])
                    # print(scale)
                    # print(i,j,p[i,j],grid_u[i,j])

        if j - 1 >=0:
            if (Voxelized[i, j - 1] == 1 or Voxelized[i, j] == 1):
                if Voxelized[i, j - 1] == -2 or Voxelized[i, j] == -2:
                    grid_v[i, j] = 0
                else:
                    grid_v[i, j] -= scale * (p[i, j] - p[i, j - 1])



@ti.kernel
def GridToParticles():
    for p in x:
        base_u = (x[p] * inv_dx - ti.Vector([0.0, 0.5])).cast(int)
        base_v = (x[p] * inv_dx - ti.Vector([0.5, 0.0])).cast(int)
        new_u = 0.0
        new_v = 0.0
        fx_u = x[p] * inv_dx - base_u.cast(float)
        fx_v = x[p] * inv_dx - base_v.cast(float)
        # Quadratic B-spline
        w_u = [0.5 * (1.5 - fx_u) ** 2, 0.75 - (fx_u - 1) ** 2, 0.5 * (fx_u - 0.5) ** 2]
        w_v = [0.5 * (1.5 - fx_v) ** 2, 0.75 - (fx_v - 1) ** 2, 0.5 * (fx_v - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight_u = w_u[i][0] * w_u[j][1]
                weight_v = w_v[i][0] * w_v[j][1]
                index_u = base_u + ti.Vector([i, j])
                index_v = base_v + ti.Vector([i, j])
                if index_u[0] <= n_grid and index_u[1] <= n_grid - 1:
                    g_u = grid_u[index_u]
                    new_u += weight_u * g_u
                if index_v[0] <= n_grid - 1 and index_v[1] <= n_grid:
                    g_v = grid_v[index_v]
                    new_v += weight_v * g_v

        v[p][0] = new_u
        v[p][1] = new_v


@ti.kernel
def init_vox():
    bound = 1
    for i,j in Voxelized:
        if (n_grid-1)-bound>= i >=bound and (n_grid-1)-bound>= j >=bound:
            Voxelized[i,j] = -1
        else:
            Voxelized[i, j] = -2
    #
    for k in x:
        cell=(x[k]*inv_dx).cast(int)
        if (n_grid-1)-bound>= cell[0] >=bound and (n_grid-1)-bound>= cell[1]  >=bound:
            Voxelized[cell] = 1


@ti.data_oriented
class CGSolver:
    #输入的参数依次为，矩阵大小m n
    def __init__(self, m, n, u, v, cell_type):
        self.m = m
        self.n = n
        self.u = u
        self.v = v
        self.cell_type = cell_type
        # 右侧的线性系统：
        self.b = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        # 左侧的线性系统
        self.Adiag = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ax = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ay = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        # cg需要的参数
        #p:x
        self.p = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #r:残差
        self.r = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #s: d
        self.s = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #As:Ad
        self.As = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #sum:rTr
        self.sum = ti.field(dtype=ti.f32, shape=())
        #alpha:往一个方向走的距离大小
        self.alpha = ti.field(dtype=ti.f32, shape=())
        #beta：
        self.beta = ti.field(dtype=ti.f32, shape=())


    @ti.kernel
    def system_init_kernel(self, scale_A: ti.f32, scale_b: ti.f32):
        # 右边线性系统
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] ==1:
                self.b[i,j] = -1 * scale_b * (self.u[i + 1, j] - self.u[i, j] +
                                              self.v[i, j + 1] - self.v[i, j])

        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1 and i-1>=0 and j-1>=0 and i+1<=self.m-1 and j+1<=self.n-1:
                if self.cell_type[i - 1, j] == -2:
                    self.b[i, j] -= scale_b * (self.u[i, j] - 0)
                if self.cell_type[i + 1, j] == -2:
                    self.b[i, j] += scale_b * (self.u[i + 1, j] - 0)
                if self.cell_type[i, j - 1] == -2:
                    self.b[i, j] -= scale_b * (self.v[i, j] - 0)
                if self.cell_type[i, j + 1] == -2:
                    self.b[i, j] += scale_b * (self.v[i, j + 1] - 0)

        #左侧线性系统：
        for i, j in ti.ndrange(self.m, self.n):
            #因为对称，在这里只关心 右 ，上 方向
            if self.cell_type[i, j] ==1 and i-1>=0 and j-1>=0 and i+1<=self.m-1 and j+1<=self.n-1 :
                if self.cell_type[i - 1, j] == 1:
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i + 1, j] == 1:
                    self.Adiag[i, j] += scale_A
                    self.Ax[i, j] = -scale_A
                elif self.cell_type[i + 1, j] == -1:
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i, j - 1] == 1:
                    self.Adiag[i, j] += scale_A
                if self.cell_type[i, j + 1] == 1:
                    self.Adiag[i, j] += scale_A
                    self.Ay[i, j] = -scale_A
                elif self.cell_type[i, j + 1] == -1:
                    self.Adiag[i, j] += scale_A


    def system_init(self, scale_A, scale_b):
        self.b.fill(0.0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)
        self.system_init_kernel(scale_A, scale_b)
        #
    def solve(self, max_iters):

        tol =1e-6
        self.p.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)

        #该系统从原点出发
        self.r.copy_from(self.b)
        self.reduce(self.r, self.r)
        init_rTr = self.sum[None]

        # print("init rTr = {}".format(init_rTr))
        if init_rTr < tol:

            print("Converged: init rtr = {}".format(init_rTr))
            # print("zhixing")

        else:

            self.s.copy_from(self.r)
            old_rTr = init_rTr

            for i in range(max_iters):
                # alpha = rTr / sAs
                #As=A*d
                self.compute_As()
                #dTq
                self.reduce(self.s, self.As)
                sAs = self.sum[None]
                if sAs==0:
                    break
                self.alpha[None] = old_rTr / sAs
                # p = p + alpha * s
                self.update_p()
                # r = r - alpha * As
                self.update_r()

                # 检查收敛性
                self.reduce(self.r, self.r)
                rTr = self.sum[None]
                if rTr < init_rTr * tol:
                    break
                new_rTr = rTr
                self.beta[None] = new_rTr / old_rTr
                # s = r + beta * s
                self.update_s()
                old_rTr = new_rTr
                i+=1

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.0
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1:
                self.sum[None] += p[i, j] * q[i, j]

    @ti.kernel
    def compute_As(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1 and i-1>=0 and j-1>=0 and i+1<=self.m-1 and j+1<=self.n-1:
                self.As[i, j] = self.Adiag[i, j] * self.s[i, j] + self.Ax[
                    i - 1, j] * self.s[i - 1, j] + self.Ax[i, j] * self.s[
                                    i + 1, j] + self.Ay[i, j - 1] * self.s[
                                    i, j - 1] + self.Ay[i, j] * self.s[i, j + 1]
    @ti.kernel
    def update_p(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] ==1:
                self.p[i, j] = self.p[i, j] + self.alpha[None] * self.s[i, j]

    @ti.kernel
    def update_r(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1:
                self.r[i, j] = self.r[i, j] - self.alpha[None] * self.As[i, j]

    @ti.kernel
    def update_s(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.cell_type[i, j] == 1:
                self.s[i, j] = self.r[i, j] + self.beta[None] * self.s[i, j]




poisson_solver = CGSolver(n_grid,n_grid, grid_u, grid_v, Voxelized)

def solver_cg():
    # 以下是共轭梯度求解压力泊松方程的部分
    scale_A = dt / (rh0 * dx * dx)
    scale_b = 1 / dx
    poisson_solver.system_init(scale_A, scale_b)
    poisson_solver.solve(200)
    p.copy_from(poisson_solver.p)
    # for i,j in ti.ndrange(n_grid,n_grid):
    #     print(i,j,p[i,j])
    # ---------------------------------

@ti.kernel
def moveParticles():
    for p in x:
        x[p] = confine_position_to_boundary(x[p] + v[p] * dt)

def substep_PIC():
    init_vox()
    ParticlesToGrid()
    GridOp()
    enforce_boundary()

    solver_cg()
    GridOp2()
    enforce_boundary()
    GridToParticles()
    moveParticles()


def render(gui):
    pos_np = x.to_numpy()
    gui.circles(pos_np,  radius=3, color=0x068587)
    gui.show()


def main():
    init_state()
    gui = ti.GUI('Hybrid2D', res=600, background_color=0x112F41)
    while gui.running and not gui.get_event(gui.ESCAPE):
        clear_grid()
        substep_PIC()
        render(gui)



if __name__ == '__main__':
    main()


