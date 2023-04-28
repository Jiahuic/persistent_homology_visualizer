import sys, os, itertools
import numpy as np
import networkx as nx
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from pathlib import Path
np.random.seed(4)


def plotBarcode(ax, simplexes, radii, eigs):
    from matplotlib import collections as mc
    ax_ = ax.twinx()
    ax_.plot(radii, eigs, 'r')
    lines = []
    start = 0
    for simplex in simplexes:
        lines.append([(simplex[0], start), (simplex[1], start)])
        start += 0.01
    lc = mc.LineCollection(lines, linewidths=2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_yticks([])
    return

def generate_random_vertices_inside_cylinder(radius1, radius2, thickness, num_vertices):
    """
    Generate random vertices inside a cylindrical region.

    Parameters:
    - radius1: Radius of the inner circle in the x-y plane
    - radius2: Radius of the outer circle in the x-y plane
    - thickness: Thickness of the cylinder in the z-direction
    - num_vertices: Number of vertices to generate

    Returns:
    - vertices: Array of vertices with shape (num_vertices, 3)
    """
    vertices = np.zeros((num_vertices, 3))

    for i in range(num_vertices):
        # Generate random polar coordinates in the x-y plane
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(radius1, radius2)

        # Convert polar coordinates to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Generate random z-coordinate
        z = np.random.uniform(-thickness / 2, thickness / 2)

        vertices[i, :] = [x, y, z]

    return vertices


def generate_random_vertices_between_spheres(inner_radius, outer_radius, num_vertices):
    """
    Generate random vertices between two spheres.

    Parameters:
    - inner_radius: Radius of the inner sphere
    - outer_radius: Radius of the outer sphere
    - num_vertices: Number of vertices to generate

    Returns:
    - vertices: Array of vertices with shape (num_vertices, 3)
    """
    vertices = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        # generate random point on the unit sphere
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        x = np.cos(theta)*np.sin(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(phi)
        # generate random radius
        r = np.random.uniform(inner_radius, outer_radius)
        # scale the point
        vertices[i, :] = r*np.array([x, y, z])
    return vertices


def generate_regular_polygon(num_vertices, radius):
    angles = np.linspace(0, 2 * np.pi, num_vertices + 1)[:-1]
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(num_vertices)
    polygon = np.vstack((x, y, z)).T
    return polygon


class SimplicialComplex:
    def __init__(self, plottype='1', numVertex=8):
        """
        Initialize the simplicial complex.

        Parameters:
        - numVertex: Number of vertices
        - boxx: x-axis size of the box
        - boxy: y-axis size of the box
        - boxz: z-axis size of the box
        """
        self.numVertex = numVertex

        ## generate vertices use numpy random. another way is (import) random.gauss
        #boxx=10, boxy=10, boxz=10
        #self.vertices = np.zeros((self.numVertex, 3))
        #self.boxx = boxx; self.boxy = boxy; self.boxz = boxz
        #for i in range(self.numVertex):
        #    self.vertices[i, :] = np.random.rand(3)
        #self.vertices[:,0] = self.vertices[:,0]*self.boxx
        #self.vertices[:,1] = self.vertices[:,1]*self.boxy
        #self.vertices[:,2] = self.vertices[:,2]*self.boxz

        #self.vertices = generate_random_vertices_between_spheres(1, 1.2, self.numVertex)
        self.vertices = generate_random_vertices_inside_cylinder(0.2, 0.3, 0.02, self.numVertex)
        self.vertices = generate_regular_polygon(self.numVertex, 2)


        ## ============== four points ============== #
        #self.numVertex = 4
        ##self.vertices = np.array([[np.sqrt(3), 0, 0], [0, 1., 0],
        ##                          [2*np.sqrt(3), 1, 0], [2*np.sqrt(3), 3, 0],
        ##                          [0, 3, 0], [np.sqrt(3), 4, 0]], dtype = np.float32)
        #self.vertices = np.array([[0, 1, 0], [1, 1, 0], 
        #                          [1, 0, 0], [0, 0, 0]], dtype = np.float32)


        # for networkx
        self.pos = {}
        for i in range(self.numVertex):
            self.pos.update({i: (self.vertices[i,:])})

    def filtration(self, interval=1, death=5, flag_plotBarcode=False):
        """
        Perform filtration on the simplicial complex.

        Parameters:
        - interval: Interval for the filtration
        - death: Death value for the filtration
        - flag_plotBarcode: Flag to plot the barcode
        """
        import gudhi
        matrixA = np.zeros((self.numVertex, self.numVertex))
        for i in range(self.numVertex):
            for j in range(i+1, self.numVertex):
                dis = np.linalg.norm(self.vertices[i]-self.vertices[j])
                matrixA[i, j] = dis
                matrixA[j, i] = dis
        rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=death)
        PH = rips_complex.create_simplex_tree(max_dimension=2).persistence()
        simplexes0 = []; simplexes1 = []; radii = [0.]
        for simplex in PH:
            dim, b, d = simplex[0], simplex[1][0], simplex[1][1]
            if simplex[1][1] > 100: d = death
            if dim == 0: 
                simplexes0.append([b, d])
                radii.append(np.round(d, 3))
                radii.append(np.round(d, 3)+0.001)
            if dim == 1: 
                simplexes1.append([b, d])
                radii.append(np.round(b, 3))
                radii.append(np.round(b, 3)+0.001)
                radii.append(np.round(d, 3))
                radii.append(np.round(d, 3)+0.001)
        radii = np.sort(radii)
        radii = np.linspace(0, 5, 20)
        L0eigs = []; L1eigs = []
        for idx, diameter in enumerate(radii):
            G = nx.random_geometric_graph(self.numVertex, diameter, pos=self.pos)
            self.plotGraph(G, idx, radius=diameter/2)
            L0eig, L1eig = self.persistenceLaplacian(G, diameter)
            L0eigs.append(L0eig)
            L1eigs.append(L1eig) # this is not right.

        if flag_plotBarcode:
            # setup figures
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8,5))
            plotBarcode(ax0, simplexes0, radii, L0eigs)
            plt.xlim([0, death])
            plotBarcode(ax1, simplexes1, radii, L1eigs)
            plt.xlim([0, death])
            plt.show()
        return

    def generator(self, radius):
        """
        Generate Rips complex.

        Parameters:
        - radius: Radius for the Rips complex
        """
        G = nx.random_geometric_graph(self.numVertex, radius*2, pos=self.pos)
        self.plotGraph(G, radius=radius)
        return

    def persistenceLaplacian(self, G, radius):
        """
        Compute the persistence Laplacian.

        Parameters:
        - G: Graph
        - radius: Radius for the persistence Laplacian

        Returns:
        - L0eig: L0 eigenvalue
        - L1eig: L1 eigenvalue
        """
        edges = [x for x in nx.enumerate_all_cliques(G) if len(x) == 2]
        triangles = [x for x in nx.enumerate_all_cliques(G) if len(x) == 3]

        matrixA = np.zeros((self.numVertex, self.numVertex), dtype=np.int16)
        for idx, pos in enumerate(self.vertices):
            for jdx in range(idx+1, self.numVertex):
                dis = np.linalg.norm(pos-self.vertices[jdx])
                if dis < radius: matrixA[idx, jdx] = 1
        matrixL = matrixA+matrixA.T
        matrixL = np.diag(sum(matrixL)) - matrixL
        eigs, eigvs = np.linalg.eig(matrixL)
        eigs = np.sort(eigs)
        index = np.sum(eigs<1e-6)
        if index != self.numVertex: 
            L0eig = eigs[index]
        else:
            L0eig = 0.

        ## NOTE: the following is wrong: there is no orientation on each edge (correct)
        D1T = np.zeros((len(edges), self.numVertex), dtype=np.int16)
        for idx, edge in enumerate(edges):
            D1T[idx, edge[0]] = 1
            D1T[idx, edge[1]] = -1
        D1 = D1T.T
        L0 = D1@D1T
        eigs, eigvs = np.linalg.eig(L0)

        ## NOTE: edge part is still not right
        D2T = np.zeros((len(triangles), len(edges)), dtype=np.int16)
        for idx, tri in enumerate(triangles):
            D2T[idx, edges.index([tri[1], tri[2]])] = 1
            D2T[idx, edges.index([tri[0], tri[2]])] = -1
            D2T[idx, edges.index([tri[0], tri[1]])] = 1
        D2 = D2T.T
        L1 = D2@D2T + D1T@D1
        eigs, eigvs = np.linalg.eig(L1)
        eigs = np.sort(eigs)
        index = np.sum(eigs<1e-6)
        if index != len(triangles):
            L1eig = eigs[index]
        else:
            L1eig = 0.

        return L0eig, L1eig


    def plotGraph(self, G, save_idx=0, radius=0.1):
        """
        Plot the graph.

        Parameters:
        - G: Graph
        - save_idx: Index for saving the figure
        """
        k2Simplex = [x for x in nx.enumerate_all_cliques(G) if len(x) == 3]
        k3Simplex = [x for x in nx.enumerate_all_cliques(G) if len(x) == 4]

        # define subplots as a square window
        fig, ax = plt.subplots(figsize=(5,5))
        # same to pos = nx.get_node_attributes(G, 'pos')
        pos = nx.get_node_attributes(G, 'pos')

        patches = []
        for s in k2Simplex:
            polygon = Polygon([[pos[s[0]][0], pos[s[0]][1]],
                               [pos[s[1]][0], pos[s[1]][1]],
                               [pos[s[2]][0], pos[s[2]][1]]], closed=True)
            patches.append(polygon)
        p = PatchCollection(patches, alpha=0.4)
        colors = 100*np.random.rand(len(patches))
        p.set_array(colors)
        ax.add_collection(p)

        # plot connecting lines, only
        for idx in G.edges():
            x = np.array((pos[idx[0]][0], pos[idx[1]][0]))
            y = np.array((pos[idx[0]][1], pos[idx[1]][1]))
            #z = np.array((pos[idx[0]][2], pos[idx[1]][2]))

            # plot the connecting lines # zorder to control the layers
            ax.plot(x, y, c='k', zorder=5)

        # Draw circles around each vertex with transparency
        for vertex in G.nodes():
            circle = plt.Circle((pos[vertex][0], pos[vertex][1]), radius, alpha=0.2, color='lightblue')
            ax.add_patch(circle)

        ax.scatter(self.vertices[:,0], self.vertices[:,1], c='r', alpha=0.8, edgecolors='k', zorder=10)
        # remove ticks and grid
        plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        #plt.show()
        plt.savefig(f'figures/simplex_{save_idx}.pdf')
        return

    def plotVertices(self):
        """
        Plot the 3D vertices in a 2D plane.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(self.vertices[:,0], self.vertices[:,1], c='r', edgecolors='k')

        # remove ticks and grid
        plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        plt.show()
        #plt.savefig('vertices.pdf', transparent=True)
        return

    def writeXYZ(self, file_name="seed4.txt"):
        """
        Write vertices to a file.

        Parameters:
        - file_name: Name of the file to save the vertices
        """
        np.savetxt(file_name, self.vertices)


def main():

    # Create a directory for the output figures
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    scomplex = SimplicialComplex()
    # scomplex.plotVertices()
    # scomplex.generator(float(sys.argv[1]))
    scomplex.filtration(flag_plotBarcode=True)
    # scomplex.writeXYZ()


if __name__ == "__main__":
    main()
