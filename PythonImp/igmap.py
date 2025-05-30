#from sympy.abc import x,y,z, s

from gblend.geometry import *
from sympy import lambdify,diff, sqrt, simplify, exp, Symbol
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
import trimesh as tm
from skimage import measure
from numba import jit
import plotly.graph_objects as go



def compute_jet2(formula):
    f = lambdify([x,y,z],formula)
    grad_f_lst = lambdify([x,y,z], [diff(formula,x),diff(formula,y),diff(formula,z)])

    def grad_f(x,y,z):
        return np.array(grad_f_lst(x,y,z))
    
    fxx = diff(formula,x,x)
    fxy = diff(formula,x,y)
    fxz = diff(formula,x,z)
    fyy = diff(formula,y,y)
    fyz = diff(formula,y,z)
    fzz = diff(formula,z,z)
    hess_f_lst = lambdify([x,y,z], [[fxx,fxy,fxz],[fxy,fyy,fyz],[fxz,fyz,fzz]])

    def hess_f(x,y,z):
        return np.array(hess_f_lst(x,y,z))
    
    return f, grad_f, hess_f

def compute_normals(grad_f, points):
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    gs = grad_f(xs,ys,zs)
    gs /= np.linalg.norm(gs, axis=0)
    return gs.T

#@jit(nopython=True)
def get_orthogonal_vectors(ns):
    v1s = np.zeros((ns.shape[0],3))
    v2s = np.zeros((ns.shape[0],3))
    for i, n in enumerate(ns):
        #if np.allclose(n, [0,0,1]) or np.allclose(n, [0,0,-1]):
        if n[2] > 0.8 or n[2] < -0.8:
            v1 = np.array([1.0,0.0,0.0])
        else:
            v1 = np.cross(n, [0.0,0.0,1.0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(n, v1)
        v2 = v2 / np.linalg.norm(v2)
        v1s[i] = v1
        v2s[i] = v2
    return v1s, v2s


def get_plotly_normal_lines(vs,ns, scale=0.1):
    normal_lines_x =[]
    normal_lines_y =[]
    normal_lines_z =[]
    for p,n in zip(vs, ns):
        normal_lines_x.extend([p[0], p[0]+scale*n[0], None])
        normal_lines_y.extend([p[1], p[1]+scale*n[1], None])
        normal_lines_z.extend([p[2], p[2]+scale*n[2], None])
    return normal_lines_x, normal_lines_y, normal_lines_z


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

    

def compute_curvatures_old(srf_vs, srf_ns, grad_f, hess_f):
    ks = np.zeros(srf_ns.shape[0])
    v1s,v2s = get_orthogonal_vectors(srf_ns)  
    for i in range(len(srf_vs)):   
        x,y,z = srf_vs[i] 
        h = hess_f(x,y,z)
        a = np.dot(np.dot(h, v1s[i]), v1s[i])
        b = np.dot(np.dot(h, v1s[i]), v2s[i])
        c = np.dot(np.dot(h, v2s[i]), v2s[i])
        g = grad_f(x,y,z)
        ks[i] = (a*c - b**2)/np.dot(g,g)
    return ks

def compute_curvatures(srf_vs, srf_ns, grad_f, hess_f):
    xs = srf_vs[:,0]
    ys = srf_vs[:,1]
    zs = srf_vs[:,2]
    hs = hess_f(xs,ys,zs)
    gs = grad_f(xs,ys,zs)
    v1s, v2s = get_orthogonal_vectors(srf_ns)
    h1 = np.einsum('ijk,kj->ki', hs, v1s)
    h2 = np.einsum('ijk,kj->ki', hs, v2s)
    h11 = np.einsum('ki,ki->k', h1, v1s)
    h22 = np.einsum('ki,ki->k', h2, v2s)
    h12 = np.einsum('ki,ki->k', h1, v2s)
    return (h11*h22 - h12**2)/np.einsum('ik,ik->k', gs, gs)


def get_spherical_triangle_condition(n0,n1,n2):
    n01 = np.cross(n0,n1)
    n12 = np.cross(n1,n2)
    n20 = np.cross(n2,n0)
    epsilon = 0.0
    def cond(x,y,z):
        c01 = x*n01[0] + y*n01[1] + z*n01[2] <= epsilon
        c12 = x*n12[0] + y*n12[1] + z*n12[2] <= epsilon
        c20 = x*n20[0] + y*n20[1] + z*n20[2] <= epsilon
        return c01 & c12 & c20
    return cond


def get_inverse_triangles_points(srf_vs, srf_ns, sph_vs, sph_fs):
    result = []     
    result2 = []
    for i, face in enumerate(sph_fs):
        n0,n1,n2 = sph_vs[face]
        cond = get_spherical_triangle_condition(n0,n1,n2)
        inv_ids, = np.where(cond(srf_ns[:,0], srf_ns[:,1], srf_ns[:,2]))
        result.append(srf_vs[inv_ids])
        result2.append(inv_ids)
    return result, result2


def get_point_normal_condition(n, epsilon=.1):
    def cond(x,y,z):
        return (x-n[0])**2 + (y-n[1])**2 + (z-n[2])**2 < epsilon**2
    return cond

def get_inverse_points(srf_vs, srf_ns, sph_vs):
    result = []
    result2 = []
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
        result_per_normal2 = []
        for l in labels[labels != -1]:
            ids, = np.where(clustering.labels_ == l)
            lvs = vs[ids]
            dsts = np.linalg.norm(srf_ns[inv_region[ids]]-normal,axis=1)
            largmin = np.argmin(dsts)
            p_min = lvs[largmin]
            result_per_normal2.append(inv_region[ids[largmin]])
            result_per_normal.append(p_min)
        result.append(np.array(result_per_normal))
        result2.append(np.array(result_per_normal2))
    return result, result2

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


# Graph utils

def get_mesh_wireframe(vs,fs):
    edges = []
    for face in fs:
        num_vertices = len(face)
        for i in range(num_vertices):
            edge = [face[i], face[(i + 1) % num_vertices]]
            edges.append(edge)
    xs = []
    ys = []
    zs = []
    for edge in edges:
        i0 = edge[0]
        i1 = edge[1]
        p0 = vs[i0]
        p1 = vs[i1]
        xs.extend([p0[0], p1[0], None])
        ys.extend([p0[1], p1[1], None])
        zs.extend([p0[2], p1[2], None])

    wireframe_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='lines',
        line=dict(
            color='black',
            width=2
        ),
        name='Wireframe'
    )
    return wireframe_trace