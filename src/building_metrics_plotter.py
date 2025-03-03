import matplotlib.pyplot as plt

class BuildingMetricsPlotter:
    def __init__(self, calculator, output_dir):
        """
        Initialize the plotter

        Parameters:
            calculator (BuildingMetricsCalculator): Calculator instance with computed metrics
            output_dir (str): Directory to save plots
        """
        self.calc = calculator
        self.output_dir = output_dir

    def _setup_basic_plot(self, figsize=(10, 10)):
        """Setup a basic plot with common parameters"""
        f, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        return f, ax

    def plot_basic_metrics(self):
        """Plot basic geometric metrics"""
        metrics = {
            'area': ('Building Areas', 'magma'),
            'perimeter': ('Building Perimeters', 'magma'),
            'facade_ratio': ('Facade Ratio', 'flare'),
            'lal': ('Longest Axis Length', 'Wistia'),
            'shape_idx': ('Shape Index', 'Blues'),
            'compact_weighted_axis': ('Compactness Weighted Axis', 'Greens'),
            'convexity': ('Convexity', 'Reds')
        }

        for metric, (title, cmap) in metrics.items():
            f, ax = self._setup_basic_plot()
            self.calc.clean_buildings.plot(
                ax=ax,
                column=metric,
                legend=True,
                scheme='quantiles',
                k=15,
                cmap=cmap
            )
            plt.title(title)
            plt.savefig(f"{self.output_dir}/{metric}_plot.png")
            plt.close()

    def plot_spatial_metrics(self):
        """Plot spatial distribution metrics"""
        metrics = {
            'neighbour_dist': ('Mean Distance Between Buildings', 'Spectral'),
            'mean_interbuilding_distance': ('Mean Interbuilding Distance', 'Spectral'),
            'adjacency': ('Building Adjacency', 'Spectral')
        }

        for metric, (title, cmap) in metrics.items():
            f, ax = self._setup_basic_plot()
            self.calc.clean_buildings.plot(
                ax=ax,
                column=metric,
                scheme='naturalbreaks',
                k=15,
                legend=True,
                cmap=cmap
            )
            plt.title(title)
            plt.savefig(f"{self.output_dir}/{metric}_plot.png")
            plt.close()

    def plot_shape_metrics(self):
        """Plot shape-related metrics"""
        metrics = {
            'shared_walls': ('Shared Walls', 'Reds'),
            'perimeter_wall': ('Perimeter Wall', 'Oranges'),
            'corners': ('Number of Corners', 'Blues'),
            'rect_idx': ('Equivalent Rectangular Index', 'flare'),
            'rectangularity': ('Rectangularity', 'magma'),
            'squareness': ('Squareness', 'magma'),
            'square_compact': ('Square Compactness', 'cool'),
            'circular_com': ('Circular Compactness', 'viridis'),
            'elongation': ('Elongation', 'Blues_r'),
            'fractal': ('Fractal Dimension', 'Wistia')
        }

        for metric, (title, cmap) in metrics.items():
            f, ax = self._setup_basic_plot()
            self.calc.clean_buildings.plot(
                ax=ax,
                column=metric,
                legend=True,
                scheme='quantiles',
                k=10,
                cmap=cmap
            )
            plt.title(title)
            plt.savefig(f"{self.output_dir}/{metric}_plot.png")
            plt.close()

    def plot_tessellation_metrics(self):
        """Plot tessellation-related metrics"""
        # Plot CAR
        f, ax = self._setup_basic_plot()
        self.calc.tessellation.plot(
            ax=ax,
            column='CAR',
            legend=True,
            scheme='quantiles',
            k=10,
            cmap='viridis'
        )
        self.calc.clean_buildings.plot(ax=ax, color='white', alpha=0.5)
        plt.title('Covered Area Ratio')
        plt.savefig(f"{self.output_dir}/covered_area_ratio.png")
        plt.close()

        # Plot Cell Alignment
        f, ax = self._setup_basic_plot()
        self.calc.clean_buildings.plot(
            ax=ax,
            column='cell_alignment',
            legend=True,
            scheme='quantiles',
            k=10,
            cmap='Blues'
        )
        plt.title('Cell Alignment')
        plt.savefig(f"{self.output_dir}/cell_alignment.png")
        plt.close()

    def plot_topology_metrics(self):
        """Plot topology-related metrics"""
        # Plot centroids with tessellation
        f, ax = self._setup_basic_plot()
        self.calc.tessellation.plot(
            ax=ax,
            edgecolor='white',
            linewidth=0.5,
            facecolor='grey',
            alpha=.7
        )
        self.calc.clean_buildings.plot(ax=ax, color='red', alpha=.7)
        self.calc.centroids_gdf.plot(
            ax=ax,
            color='black',
            markersize=0.5,
            marker='o',
            label='Centroids'
        )
        plt.title('Building Footprints with Centroids')
        plt.savefig(f"{self.output_dir}/centroids.png")
        plt.close()

        # Plot weighted Delaunay triangulation
        f, ax = self._setup_basic_plot()
        self.calc.weighted_edges.plot(
            ax=ax,
            column='weight',
            legend=True,
            scheme='quantiles',
            k=10,
            cmap='viridis',
            linewidth=0.7
        )
        plt.title('Weighted Delaunay Triangulation')
        plt.savefig(f"{self.output_dir}/delaunay_triangulation.png")
        plt.close()

    def plot_all(self):
        """Generate all plots"""
        self.plot_basic_metrics()
        self.plot_spatial_metrics()
        self.plot_shape_metrics()
        self.plot_tessellation_metrics()
        self.plot_topology_metrics()