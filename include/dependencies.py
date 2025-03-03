# dependencies.py

# Third-party packages (make sure these are installed)
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import momepy as mm
import momepy
import pyproj
import osmnx as ox
import libpysal
from shapely.geometry import box
from time import time
from clustergram import Clustergram
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.basemap import Basemap
from bokeh.io import output_notebook
from bokeh.plotting import show
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from libpysal import graph
from shapely import LineString
from shapely.geometry import Point

# Custom Packages
from src import *



#INSTALLATION:
# !pip install momepy
# !pip install osmnx
# !pip install PyDrive
# !pip install clustergram
# !pip install folium matplotlib mapclassify
# !pip install shapely
# !pip install matplotlib-scalebar
# !pip install basemap
# !pip install bokeh
# !pip install seaborn
# !pip install libpysal