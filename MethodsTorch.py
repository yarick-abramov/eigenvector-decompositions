import torch
import sympy
from sympy import Symbol, Mul, Pow, Add

def print_gpu_memory():
    """Prints the current allocated, reserved, and free memory on the GPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert to MB
        free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)  # Free memory in reserved area
        free_memory_mb = free_memory / (1024 ** 2)  # Convert to MB

        print(f"Allocated memory: {allocated_memory:.2f} MB")
        print(f"Reserved memory: {reserved_memory:.2f} MB")
        print(f"Free memory: {free_memory_mb:.2f} MB")
    else:
        print("CUDA is not available. No GPU detected.")

t=sympy.Symbol('t')
sqrt_3=torch.sqrt(torch.tensor([3.])).cuda()

# Auxillary computations for Ferrari Solution of Quartic Equations

def complex_cbrt(z):
    """Calculate the cubic root of a complex number."""
    # Calculate the magnitude and angle of the complex number
    r = z.abs()  # Magnitude
    theta = torch.angle(z)  # Angle in radians
    #print("z")
    #print(z)
    #print("r")
    #print(r)
    #print("theta")
    #print(theta)

    # Calculate the cubic root of the magnitude and the angle divided by 3
    r = r ** (1/3)  # Magnitude of the cubic root
    theta = theta / 3  # Angle of the cubic root
    #print("root_magnitude")
    #print(r)
    #print("root_angle")
    #print(theta)

    return r * torch.cos(theta) + 1j * r * torch.sin(theta)  # Return as a complex number

def quartic_solver(a,b,c,d,e):
    e = e/a
    d = d/a
    c = c/a
    b = b/a
    i, lr_1 = torch.frexp(e)
    i, lr_2 = torch.frexp(d)
    i, lr_3 = torch.frexp(c)
    i, lr_4 = torch.frexp(b)
    i = torch.max(
        torch.cat((2 + 3 * torch.max(torch.cat((2 * lr_3, 2 + lr_4 + lr_2, 4 + lr_1), 
                                           dim = -1), dim = -1, keepdim = True)[0], 
                  2 * torch.max(torch.cat((1 + 3 * lr_3, 3 + lr_4 + lr_3 + lr_2, 5 + 2 * lr_4 + lr_1, 5 + 2 * lr_2, 8 + lr_3 + lr_1), 
                                        dim = -1), dim = -1, keepdim = True)[0]), 
                  dim = -1), dim = -1, keepdim = True)[0]//12
        
    a = c ** 2 - 3 * b * d + 12 * e
    e = 2 * c ** 3 - 9 * b * c * d + 27 * b ** 2 * e + 27 * d ** 2 - 72 * c * e
    e = torch.ldexp(e, -6*i)
    a = torch.ldexp(a, -4*i)
    #print("Delta_0")
    #print(a)
    #print("Delta_1")
    #print(e)
    #print("under sqrt")
    #print(e ** 2 - 4 * a ** 3)
    #print("sqrt")
    #print(torch.sqrt(torch.tensor(e ** 2 - 4 * a ** 3, dtype=torch.complex64)))

    Q = complex_cbrt((e + torch.sqrt(torch.tensor(e ** 2 - 4 * a ** 3, dtype=torch.complex64))) / 2) 
    #print("Q")
    #print(Q)
    
    # Calculate p and q
    e = torch.ldexp(c - 0.375 * b ** 2, -2*i)
    d = torch.ldexp((0.5*b) ** 3 - 0.5 * b * c + d, -3*i)
    #print("q")
    #print(d)
    S = torch.sqrt(-2 / 3 * e + (Q + a / Q) / 3) / 2
    a = 0.5*torch.sqrt(-4 * S ** 2 - 2 * e + d / S)
    c = 0.5*torch.sqrt(-4 * S ** 2 - 2 * e - d / S)
    b = torch.ldexp(-b, -i-2)

    lr_1 = torch.ldexp(torch.real(b - S + a), i)
    lr_2 = torch.ldexp(torch.real(b - S - a), i)
    lr_3 = torch.ldexp(torch.real(b + S + c), i)
    lr_4 = torch.ldexp(torch.real(b + S - c), i)
    
    return torch.cat([lr_1, lr_2, lr_3, lr_4],dim = -1)

def cubic_solver(a,b,c,d):
    d=d/a
    c=c/a
    b=b/a
    a = c - b ** 2 / 3
    d = 2/27*b**3 - b*c/3 + d
    c = (a/3)**3 + (d/2)**2
    c = torch.sqrt(c)
    alpha = -d/2 + c
    beta = -d/2 - c
    alpha = complex_cbrt(alpha)
    beta = complex_cbrt(beta)
    
    lr1 = torch.real(alpha+beta)
    lr2 = torch.real((-(alpha+beta) + 1j * sqrt_3 * (alpha-beta))/2)
    lr3 = torch.real((-(alpha+beta) - 1j * sqrt_3 * (alpha-beta))/2)
    return torch.cat([lr1, lr2, lr3, torch.zeros_like(lr1)],dim=-1)


def quadratic_solver(c,d,e):
    D = torch.sqrt(d ** 2 - 4 * e * c)
    x1 = torch.real((-d + D)/(2*c))
    x2 = torch.real((-d - D)/(2*c))
    return torch.cat([x1, x2, torch.zeros_like(x1), torch.zeros_like(x1)], dim=-1)

def polynomial_solver(a,b,c,d,e):
    return torch.where(a!=0, 
                       quartic_solver(a,b,c,d,e), torch.where(
                        b!=0,
                        cubic_solver(b,c,d,e), torch.where(
                        c!=0,
                        quadratic_solver(c,d,e), torch.where(
                        d!=0, 
                        (e/d).repeat((len(a.shape)-1)*(1,)+(4,)), 
                        e.repeat((len(a.shape)-1)*(1,)+(4,))))))


def optimal_lr(A, x):
    """Calculate optimal learning rate based on matrix A and vector x."""
    # Compute intermediate vectors
    u = A(x)
    v = A(u)
    w = A(v)
    
    # Compute inner products
    a_0 = torch.sum(x*x, dim = -1, keepdim = True)
    a_1 = torch.sum(u*x, dim = -1, keepdim = True)
    a_2 = torch.sum(v*x, dim = -1, keepdim = True)
    a_3 = torch.sum(w*x, dim = -1, keepdim = True)
    a_4 = torch.sum(w*u, dim = -1, keepdim = True)
    a_5 = torch.sum(w*v, dim = -1, keepdim = True)
    a_6 = torch.sum(w*w, dim = -1, keepdim = True)

    # Calculate r_0, r_1, r_2
    r_0 = 4 * a_2 / (a_1 * a_1) - 2 * a_1 / (a_1 * a_0) - 2 * a_3 / (a_1 * a_2) - 2 * a_3 / (a_1 * a_2) + a_4 / (a_2 * a_2) + a_2 / (a_0 * a_2) - 2 * a_1 / (a_1 * a_0) + a_0 / (a_0 * a_0) + a_2 / (a_0 * a_2)
    r_1 = 4 * a_3 / (a_1 * a_1) - 2 * a_2 / (a_1 * a_0) - 2 * a_4 / (a_1 * a_2) - 2 * a_4 / (a_1 * a_2) + a_5 / (a_2 * a_2) + a_3 / (a_0 * a_2) - 2 * a_2 / (a_1 * a_0) + a_1 / (a_0 * a_0) + a_3 / (a_0 * a_2)
    r_2 = 4 * a_4 / (a_1 * a_1) - 2 * a_3 / (a_1 * a_0) - 2 * a_5 / (a_1 * a_2) - 2 * a_5 / (a_1 * a_2) + a_6 / (a_2 * a_2) + a_4 / (a_0 * a_2) - 2 * a_3 / (a_1 * a_0) + a_2 / (a_0 * a_0) + a_4 / (a_0 * a_2)

    # Calculate q and p
    q_1 = 2 * a_2 / a_1 - a_1 / a_0 - a_3 / a_2
    q_2 = 2 * a_3 / a_1 - a_2 / a_0 - a_4 / a_2
    #p_0 = a_0
    #p_1 = a_1
    #p_2 = a_2

    a = r_0 * r_1 * q_2 - 2 * r_0 * q_1 * r_2
    b = a_0 * r_1 * r_2 - 2 * a_1 * r_0 * r_2 + a_2 * r_0 * r_1 - 2 * q_1 * q_2 * r_0
    c = 3 * a_0 * r_1 * q_2 - 3 * r_0 * a_1 * q_2
    d = 2 * a_0 * q_1 * q_2 + 2 * a_0 * r_1 * a_2 - a_0 * a_1 * r_2 - r_0 * a_1 * a_2
    e = 2 * a_0 * q_1 * a_2 - a_0 * a_1 * q_2
    lr = polynomial_solver(a,b,c,d,e)
    g = a_1 + 2 * lr * q_1 + lr ** 2 * r_1
    f = a_0 + lr ** 2 * r_0
    h = a_2 + 2 * lr * q_2 + lr ** 2 * r_2
    eigenness = g**2/(f*h)
    n = torch.argmax(eigenness, dim=-1, keepdim=True)
    lr = torch.gather(lr, -1, n)
    return torch.gather(eigenness, -1, n), lr * (-x / a_0 + 2 * u / a_1 - v / a_2), n, lr
    return torch.gather(eigenness, -1, n), (-lr / a_0) * x + (2 * lr / a_1) * u + (-lr / a_2) * v, n, lr


def update_vector(x, v):
    x = x + v
    return x/torch.linalg.norm(x, dim = -1, keepdim = True)

#Optimal LR Gradient Ascend

def grad_ascend_lr(A,x,threshold,steps_already,steps_max):
    e,v,i,l = optimal_lr(A, x)
    x = update_vector(x, v)
    steps = steps_already + 1
    cond = (1 - e > threshold) & (steps < steps_max)
    while cond.any():
        steps = torch.where(cond, steps + 1, steps)
        e,v,i,l = optimal_lr(A, x)
        x = torch.where(cond, update_vector(x, v), x)
        cond = (1 - e > threshold) & (steps < steps_max)
    v=[]
    f=A(x)
    #print_gpu_memory()
    return x, torch.einsum('ij,ij->i',f,x)/torch.linalg.norm(x,dim=-1).unsqueeze(-1)**2, (
            f-(torch.einsum('ij,ij->i',f,x)/torch.linalg.norm(x,dim=-1)).unsqueeze(-1)**2*x
            )/torch.linalg.norm(f, dim=-1).unsqueeze(-1), steps

import torch.nn as nn

def add(x,y):
    return x+y

class square(nn.Module):
    def __init__(self, g):
        super(square, self).__init__()
        self.g=g
    
    def forward(self, x):
        return self.g(self.g(x))

class minus_f(nn.Module):
    def __init__(self, g, f):
        super(minus_f, self).__init__()
        self.g=g
        self.f=f

    def forward(self, x):
        return self.g(x) - self.f*x

class t_minus(nn.Module):
    def __init__(self, g, t):
        super(t_minus, self).__init__()
        self.g=g
        self.t=t

    def forward(self, x):
        return self.t*x - self.g(x)


#Polynomial Transform

class modified(nn.Module):
    def __init__(self,expr,A,batch_size):
        super(modified, self).__init__()
        self.parts=[]
        for arg in expr.args:
            self.parts.append(modified(arg,A,batch_size))
        if expr.is_Number:
            self.param = nn.Parameter(torch.random.uniform((batch_size,1))*float(expr))
        self.f=torch.rand(batch_size,1).cuda()
        self.t=((torch.rand(batch_size,1).cuda())*torch.max(torch.cat([self.f**2, (1-self.f)**2], dim=1), dim=1, keepdim = True)[0]).cuda()
        self.z=torch.max(torch.cat([self.t**2, (self.t-self.f**2)**2, (self.t-(1-self.f)**2)**2], dim=1), dim=1, keepdim = True)[0]
        if expr == sympy.Symbol('t'):
            self.A=A
        self.expr=expr
        self.minus_f = minus_f(self.A, self.f)
        self.square_1 = square(self.minus_f)
        self.t_minus = t_minus(self.square_1, self.t)
        self.square_2 = square(self.t_minus)
    
    def forward_1(self,x):
        return x - self.square_2(x)/self.z

        
    def forward(self,x):
        y = self.forward_1(x)
        y = self.forward_1(y)
        y = self.forward_1(y)
        return self.forward_1(y) 
        if self.expr == sympy.Symbol('t'):
            return self.A(x)

        if self.expr.is_Number:
            return self.param*x

        if self.expr.is_Add:
            # Initialize res as float zeros to avoid dtype conflicts
            res = torch.zeros_like(x)
            for arg in self.parts:
                res += arg(x)
            return res

        if self.expr.is_Mul:
            # product: to mimic polynomial multiplication
            for arg in self.parts:
                x = arg(x)
            return x

        if self.expr.is_Pow:
            for _ in range(self.parts[1]):
                x = self.parts[0](x)
            return x
 

        
#Truncated version of operator from the method of Decreasing "Effective" Dimensionality

class truncated(nn.Module):
    def __init__(self, f, x):
        super(truncated, self).__init__()
        
        y = f(x)
        xx = torch.sum(x**2, dim = -1, keepdim = True)
        xAx = torch.sum(x*y, dim = -1, keepdim = True)
        xAAx = torch.sum(y**2, dim = -1, keepdim = True)
        mean = xAx/xx
        sigma = torch.sqrt(xAAx*xx - xAx**2)/xx
        self.mean = mean + (-1)**torch.bernoulli(probs) * sigma
        self.sigma = torch.sqrt(-2*sigma*torch.log(delta))
        self.lambda_min = self.mean - self.sigma
        self.lambda_max = self.mean + self.sigma
        self.f = f

    def forward(self, x):
        y = self.f(x) - self.mean * x
        y = self.f(y) - self.mean * y
        return x - y / torch.min(torch.cat((
            (self.lambda_min - self.mean)**2, 
            (self.lambda_max - self.mean)**2), 
                dim = -1), dim = -1, keepdim = True)[0]
