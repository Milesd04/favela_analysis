import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import momepy
from libpysal import graph
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from shapely.geometry import LineString
import matplotlib.pyplot as plt


# check what we are even doing with the topology metircs skull

class BuildingMetricsCalculator:
    def __init__(self, clean_buildings, tessellation, output_dir):
        """
        Initialize the calculator with building and tessellation data

        Parameters:
            clean_buildings (GeoDataFrame): The cleaned building data
            tessellation (GeoDataFrame): The tessellation data
            output_dir (str): Directory to save output plots
        """
        self.clean_buildings = clean_buildings
        self.tessellation = tessellation
        self.output_dir = output_dir

    def calculate_basic_metrics(self):
        """Calculate basic geometric metrics for buildings"""
        self.clean_buildings["area"] = self.clean_buildings.area
        self.clean_buildings["perimeter"] = self.clean_buildings.length
        self.clean_buildings["facade_ratio"] = momepy.facade_ratio(self.clean_buildings)
        self.clean_buildings["lal"] = momepy.longest_axis_length(self.clean_buildings)
        self.clean_buildings["shape_idx"] = momepy.shape_index(self.clean_buildings)
        self.clean_buildings["compact_weighted_axis"] = momepy.compactness_weighted_axis(self.clean_buildings)
        self.clean_buildings["convexity"] = momepy.convexity(self.clean_buildings)

    def calculate_spatial_metrics(self):
        """Calculate spatial distribution metrics"""
        queen_tess = graph.Graph.build_contiguity(self.tessellation, rook=False)
        dist200 = graph.Graph.build_distance_band(self.clean_buildings.centroid, 200)

        self.clean_buildings["neighbour_dist"] = momepy.neighbor_distance(self.clean_buildings, queen_tess)
        self.clean_buildings["mean_interbuilding_distance"] = momepy.mean_interbuilding_distance(
            self.clean_buildings, adjacency_graph=queen_tess, neighborhood_graph=dist200
        )
        self.clean_buildings["adjacency"] = momepy.building_adjacency(
            contiguity_graph=queen_tess, neighborhood_graph=dist200
        )

        self.tessellation['num_neighbors'] = momepy.neighbors(self.tessellation, queen_tess) #should be in calculate tessellation metricss

    def calculate_shape_metrics(self):
        """Calculate shape-related metrics"""
        self.clean_buildings['shared_walls'] = momepy.shared_walls(self.clean_buildings)
        self.clean_buildings['perimeter_wall'] = momepy.perimeter_wall(self.clean_buildings)
        self.clean_buildings['corners'] = momepy.corners(self.clean_buildings, include_interiors=True)
        self.clean_buildings['rect_idx'] = momepy.equivalent_rectangular_index(self.clean_buildings)
        self.clean_buildings['rectangularity'] = momepy.rectangularity(self.clean_buildings)
        self.clean_buildings['squareness'] = momepy.squareness(self.clean_buildings)
        self.clean_buildings['square_compact'] = momepy.square_compactness(self.clean_buildings)
        self.clean_buildings['elongation'] = momepy.elongation(self.clean_buildings)
        self.clean_buildings['fractal'] = momepy.fractal_dimension(self.clean_buildings)

    def calculate_tessellation_metrics(self):
        """Calculate tessellation-related metrics"""
        self.tessellation['cwa'] = momepy.compactness_weighted_axis(self.tessellation)
        self.tessellation['tes_area'] = self.tessellation.geometry.area
        # self.tessellation['num_neighbors'] = momepy.neighbors(self.tessellation, queen_tess) it'd be nice to have it here, but don't want to repeat queentess calc

        # Calculate CAR (Covered Area Ratio)
        building_areas = self.clean_buildings.groupby('uID')['area'].sum() #change this so it doesn't rely on UID tbh
        self.tessellation['CAR'] = building_areas / self.tessellation['tes_area']

        # Calculate cell alignment
        blg_orient = momepy.orientation(self.clean_buildings)
        tess_orient = momepy.orientation(self.tessellation)
        self.clean_buildings['cell_alignment'] = momepy.cell_alignment(blg_orient, tess_orient)

    def calculate_topology_metrics(self):
        """Calculate topology-related metrics"""
        # Calculate centroids
        self.clean_buildings['centroid'] = self.clean_buildings.geometry.apply(
            lambda geom: geom.convex_hull.centroid if geom.geom_type == 'MultiPolygon' else geom.centroid
        )

        # Create centroids GeoDataFrame
        self.centroids_gdf = gpd.GeoDataFrame({
            'uID': self.clean_buildings['uID'],
            'geometry': self.clean_buildings['centroid']
        }, crs=self.clean_buildings.crs)


        coordinates = np.column_stack((
            self.centroids_gdf.geometry.x,
            self.centroids_gdf.geometry.y
        ))

         # Calculate Delaunay triangulation and weighted distances
        tri = Delaunay(coordinates)
        G = self._create_delaunay_graph(coordinates, tri)
        self.weighted_edges = self._calculate_weighted_edges(G, coordinates)
        self.avg_distances = self._calculate_average_distances(G)
        self.clean_buildings['avg_weighted_dist'] = self.clean_buildings.index.map(self.avg_distances)

    def _create_delaunay_graph(self, coordinates, tri):
        """Create networkx graph from Delaunay triangulation"""
        G = nx.Graph()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    p1 = coordinates[simplex[i]]
                    p2 = coordinates[simplex[j]]
                    distance = euclidean(p1, p2)
                    G.add_edge(simplex[i], simplex[j], weight=distance)
        return G

    def _calculate_weighted_edges(self, G, coordinates):
        """Calculate weighted edges for the Delaunay triangulation"""
        edges = []
        for u, v, data in G.edges(data=True):
            line = LineString([coordinates[u], coordinates[v]])
            edges.append({'geometry': line, 'weight': data['weight']})
        return gpd.GeoDataFrame(edges)

    def _calculate_average_distances(self, G):
        """Calculate average weighted distances for each node"""
        avg_distances = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                weights = [G[node][neighbor]['weight'] for neighbor in neighbors]
                avg_distances[node] = sum(weights) / len(weights)
            else:
                avg_distances[node] = 0.0
        return avg_distances

    def run_all_calculations(self):
        """Run all metric calculations"""
        self.calculate_basic_metrics()
        self.calculate_spatial_metrics()
        self.calculate_shape_metrics()
        self.calculate_tessellation_metrics()
        self.calculate_topology_metrics()

    def save_results(self):
        """Save the updated GeoDataFrames"""
        self.clean_buildings.to_file(f"{self.output_dir}/clean_buildings_with_metrics.shp")
        self.tessellation.to_file(f"{self.output_dir}/tessellation_with_metrics.shp")