#from sympy.abc import x,y,z, s

from gblend.geometry import *
from sympy import lambdify,diff, sqrt, simplify, exp, Symbol
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
import trimesh as tm
from skimage import measure
from numba import jit

s = Symbol('s')

#sigmoid = 1/(1+exp(-2*s))

def compute_jet(ff):
    f = lambdify([x,y,z],ff)
    fx = lambdify([x,y,z], diff(ff,x))
    fy = lambdify([x,y,z], diff(ff,y))
    fz = lambdify([x,y,z], diff(ff,z))
    return f, fx, fy, fz

def compute_jet2(formula):
    f = lambdify([x,y,z],formula)
    fx = lambdify([x,y,z], diff(formula,x))
    fy = lambdify([x,y,z], diff(formula,y))
    fz = lambdify([x,y,z], diff(formula,z))
    fxx = lambdify([x,y,z], diff(formula,x,x))
    fxy = lambdify([x,y,z], diff(formula,x,y))
    fxz = lambdify([x,y,z], diff(formula,x,z))
    fyy = lambdify([x,y,z], diff(formula,y,y))
    fyz = lambdify([x,y,z], diff(formula,y,z))
    fzz = lambdify([x,y,z], diff(formula,z,z))
    return f, fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz

def compute_jet2_v2(formula):
    f = lambdify([x,y,z],formula)
    grad_f = lambdify([x,y,z], [diff(formula,x),diff(formula,y),diff(formula,y)])
    fxx = diff(formula,x,x)
    fxy = diff(formula,x,x)
    fxz = diff(formula,x,x)
    fyy = diff(formula,x,x)
    fyz = diff(formula,x,x)
    fzz = diff(formula,x,x)
    hess_f = lambdify([x,y,z], [[fxx,fxy,fxz],[fxy,fyy,fyz],[fxz,fyz,fzz]])
    return f, grad_f, hess_f

def compute_curvatures(formula, points):
    f, fx, fy, fz, fxx, fxy,fxz, fyy, fyz, fzz = compute_jet2(formula)
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    fxxs = fxx(xs,ys,zs)
    fxys = fxy(xs,ys,zs)
    fxzs = fxz(xs,ys,zs)
    fyys = fyy(xs,ys,zs)
    fyzs = fyz(xs,ys,zs)
    fzzs = fzz(xs,ys,zs)
    hessians = np.array([[fxxs, fxys, fxzs],
                            [fxys, fyys, fyzs],
                            [fxzs, fyzs, fzzs]]).T
    
    norms, ns = get_normals(fx,fy,fz, points)
    v1s, v2s = get_orthogonal_vectors(ns)
    ks = np.dot(np.dot(hessians, np.transpose(v1s)),np.traspose(v2s)) / norms
    return ks
    


def iso_surface(f, box, res = 150):
    x_min, x_max, y_min, y_max, z_min, z_max = box
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    z = np.linspace(z_min, z_max, res)
    X, Y, Z = np.meshgrid(x, y, z)
    vertices, faces, _, _ = measure.marching_cubes(f(Y,X,Z), level=0, spacing=(1.0, 1.0, 1.0))
    vertices[:,0] = vertices[:,0] * (box[1] - box[0]) / res + box[0]
    vertices[:,1] = vertices[:,1] * (box[3] - box[2]) / res + box[2]
    vertices[:,2] = vertices[:,2] * (box[5] - box[4]) / res + box[4]
    return vertices, faces

def get_spherical_triangle_condition(n0,n1,n2):
    n01 = np.cross(n0,n1)
    n12 = np.cross(n1,n2)
    n20 = np.cross(n2,n0)
    epsilon = 0.0
    def cond(x,y,z):
        c01 = x*n01[0] + y*n01[1] + z*n01[2] > epsilon
        c12 = x*n12[0] + y*n12[1] + z*n12[2] > epsilon
        c20 = x*n20[0] + y*n20[1] + z*n20[2] > epsilon
        return c01 & c12 & c20
    return cond

def get_point_normal_condition(n, epsilon=.1):
    def cond(x,y,z):
        return (x-n[0])**2 + (y-n[1])**2 + (z-n[2])**2 < epsilon**2
    return cond


def get_normals(fx,fy,fz, points):
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    fxs = fx(xs,ys,zs)
    fys = fy(xs,ys,zs)
    fzs = fz(xs,ys,zs)
    ns = np.sqrt(fxs**2 + fys**2 + fzs**2)
    return ns, np.array([fxs/ns, fys/ns, fzs/ns]).T

def get_inverse_triangles_points(srf_vs, srf_ns, sph_vs, sph_fs):
    result = []      
    for face in sph_fs:
        n0,n1,n2 = sph_vs[face]
        cond = get_spherical_triangle_condition(n0,n1,n2)
        inv_ids, = np.where(cond(srf_ns[:,0], srf_ns[:,1], srf_ns[:,2]))
        result.append(srf_vs[inv_ids])
    return result


def get_inverse_points(srf_vs, srf_ns, sph_vs):
    result = []
    for i,normal in enumerate(sph_vs):
        cond = get_point_normal_condition(normal, epsilon=0.1)
        inv_region, = np.where(cond(srf_ns[:,0], srf_ns[:,1], srf_ns[:,2]))
        if len(inv_region) == 0:
            print("No points in region %s"%i)
            continue
        vs = srf_vs[inv_region]
        clustering = DBSCAN(eps=0.125, min_samples=1).fit(vs)
        labels = np.unique(clustering.labels_)
        
        result_per_normal = []
        for l in labels[labels != -1]:
            ids, = np.where(clustering.labels_ == l)
            lvs = vs[ids]
            dsts = np.linalg.norm(srf_ns[inv_region[ids]]-normal,axis=1)
            p_min = lvs[np.argmin(dsts)]
            result_per_normal.append(p_min)
        result.append(np.array(result_per_normal))
    return result


def get_minimal_vertices_distance(vs, fs):
    sp_mesh = tm.Trimesh(vs, fs)
    vnghs = sp_mesh.vertex_neighbors
    d_min = np.inf
    for i in range(len(vs)):
        nghs = np.array(vnghs[i])
        p = vs[i]
        d = np.min(np.linalg.norm(p-vs[nghs]))
        if d < d_min:
            d_min = d
    return d_min

def get_plotly_normal_lines(vs,ns):
    normal_lines_x =[]
    normal_lines_y =[]
    normal_lines_z =[]
    for p,n in zip(vs, ns):
        scale = 0.1
        normal_lines_x.extend([p[0], p[0]+scale*n[0], None])
        normal_lines_y.extend([p[1], p[1]+scale*n[1], None])
        normal_lines_z.extend([p[2], p[2]+scale*n[2], None])
    return normal_lines_x, normal_lines_y, normal_lines_z

#@jit(nopython=True)
def get_orthogonal_vectors(ns):
    v1s = []
    v2s = []
    for n in ns:
        if np.allclose(n, [0,0,1]) or np.allclose(n, [0,0,-1]):
            v1 = np.array([1,0,0])
        else:
            v1 = np.cross(n, [0,0,1])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(n, v1)
        v2 = v2 / np.linalg.norm(v2)
        v1s.append(v1)
        v2s.append(v2)
    return np.array(v1s), np.array(v2s)


