"""
2.5D DC Resistivity Least-Squares Inversion
===========================================

Here we invert a line of DC resistivity data to recover an electrical
conductivity model. We formulate the inverse problem as a least-squares
optimization problem. For this tutorial, we focus on the following:

    - Defining the survey
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem (data misfit, regularization, directives)
    - Applying sensitivity weighting
    - Plotting the recovered model and data misfit


"""

#########################################################################
# Import modules
# --------------
#

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tarfile

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import surface2ind_topo, model_builder
from SimPEG import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static.utils.static_utils import plot_pseudosection

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

mpl.rcParams.update({"font.size": 16})
# sphinx_gallery_thumbnail_number = 2


#############################################
# Define File Names
# -----------------
#
# Here we provide the file paths to assets we need to run the inversion. The
# path to the true model conductivity and chargeability models are also
# provided for comparison with the inversion results. These files are stored as a
# tar-file on our google cloud bucket:
# "https://storage.googleapis.com/simpeg/doc-assets/dcip2d.tar.gz"
#

# storage bucket where we have the data
data_source = "https://storage.googleapis.com/simpeg/doc-assets/dcip2d.tar.gz"

# download the data
downloaded_data = utils.download(data_source, overwrite=True)

# unzip the tarfile
tar = tarfile.open(downloaded_data, "r")
tar.extractall()
tar.close()

# path to the directory containing our data
dir_path = downloaded_data.split(".")[0] + os.path.sep

# files to work with
topo_filename = dir_path + "xyz_topo.txt"
data_filename = dir_path + "dc_data.obs"


#############################################
# Load Data, Define Survey and Plot
# ---------------------------------
#
# Here we load the observed data, define the DC and IP survey geometry and
# plot the data values using pseudo-sections.
#

# Load data
topo_xyz = np.loadtxt(str(topo_filename))
dobs = np.loadtxt(str(data_filename))

# Extract source and receiver electrode locations and the observed data
A_electrodes = dobs[:, 0:2]
B_electrodes = dobs[:, 2:4]
M_electrodes = dobs[:, 4:6]
N_electrodes = dobs[:, 6:8]
dobs = dobs[:, -1]

# Define survey
unique_tx, k = np.unique(np.c_[A_electrodes, B_electrodes], axis=0, return_index=True)
n_sources = len(k)
k = np.r_[k, len(A_electrodes) + 1]

source_list = []
for ii in range(0, n_sources):

    # MN electrode locations for receivers. Each is an (N, 3) numpy array
    M_locations = M_electrodes[k[ii] : k[ii + 1], :]
    N_locations = N_electrodes[k[ii] : k[ii + 1], :]
    receiver_list = [dc.receivers.Dipole(M_locations, N_locations, data_type="volt")]

    # AB electrode locations for source. Each is a (1, 3) numpy array
    A_location = A_electrodes[k[ii], :]
    B_location = B_electrodes[k[ii], :]
    source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))

# Define survey
survey = dc.survey.Survey_ky(source_list)

# Define the a data object. Uncertainties are added later
dc_data = data.Data(survey, dobs=dobs)

# Plot apparent conductivity using pseudo-section
mpl.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_axes([0.05, 0.05, 0.8, 0.9])
plot_pseudosection(
    dc_data,
    ax=ax1,
    survey_type="dipole-dipole",
    data_type="appConductivity",
    space_type="half-space",
    scale="log",
    y_values="pseudo-depth",
    pcolor_opts={"cmap": "viridis"},
)
ax1.set_title("Apparent Conductivity [S/m]")

plt.show()

#############################################
# Assign Uncertainties
# --------------------
#
# Inversion with SimPEG requires that we define standard deviation on our data.
# This represents our estimate of the noise in our data. For DC data, a relative
# error is applied to each datum.
#

# Compute standard deviations
std = 0.05 * np.abs(dobs)

# Add standard deviations to data object
dc_data.standard_deviation = std

########################################################
# Create Tree Mesh
# ------------------
#
# Here, we create the Tree mesh that will be used to predict both DC
# resistivity and IP data.
#

dh = 10.0  # base cell width
dom_width_x = 2400.0  # domain width x
dom_width_z = 1200.0  # domain width z
nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))  # num. base cells x
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))  # num. base cells z

# Define the base mesh
hx = [(dh, nbcx)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hz], x0="CN")

# Mesh refinement based on topography
mesh = refine_tree_xyz(
    mesh, topo_xyz[:, [0, 2]], octree_levels=[1], method="surface", finalize=False
)

# Mesh refinement near transmitters and receivers
electrode_locations = np.r_[
    survey.locations_a, survey.locations_b, survey.locations_m, survey.locations_n
]

unique_locations = np.unique(electrode_locations, axis=0)

mesh = refine_tree_xyz(
    mesh, unique_locations, octree_levels=[2, 4], method="radial", finalize=False
)

# Refine core mesh region
xp, zp = np.meshgrid([-800.0, 800.0], [-800.0, 0.0])
xyz = np.c_[mkvc(xp), mkvc(zp)]
mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 2], method="box", finalize=False)

mesh.finalize()


###############################################################
# Project Surveys to Discretized Topography
# -----------------------------------------
#
# It is important that electrodes are not model as being in the air. Even if the
# electrodes are properly located along surface topography, they may lie above
# the discretized topography. This step is carried out to ensure all electrodes
# like on the discretized surface.
#

# Create 2D topography. Since our 3D topography only changes in the x direction,
# it is easy to define the 2D topography projected along the survey line. For
# arbitrary topography and for an arbitrary survey orientation, the user must
# define the 2D topography along the survey line.
topo_2d = np.unique(topo_xyz[:, [0, 2]], axis=0)

# Find cells that lie below surface topography
ind_active = surface2ind_topo(mesh, topo_2d)

# Shift electrodes to the surface of discretized topography
survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

########################################################
# Starting/Reference Model and Mapping on Tree Mesh
# ---------------------------------------------------
#
# Here, we would create starting and/or reference models for the DC inversion as
# well as the mapping from the model space to the active cells. Starting and
# reference models can be a constant background value or contain a-priori
# structures. Here, the starting model is the natural log of 0.01 S/m.
#

# Define conductivity model in S/m (or resistivity model in Ohm m)
air_conductivity = np.log(1e-8)
background_conductivity = np.log(1e-2)

active_map = maps.InjectActiveCells(mesh, ind_active, np.exp(air_conductivity))
nC = int(ind_active.sum())

conductivity_map = active_map * maps.ExpMap()

# Define model
starting_conductivity_model = background_conductivity * np.ones(nC)

##############################################
# Define the Physics of the DC Simulation
# ---------------------------------------
#
# Here, we define the physics of the DC resistivity problem.
#

# Define the problem. Define the cells below topography and the mapping
simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, sigmaMap=conductivity_map, Solver=Solver
)

#######################################################################
# Define DC Inverse Problem
# -------------------------
#
# The inverse problem is defined by 3 things:
#
#     1) Data Misfit: a measure of how well our recovered model explains the field data
#     2) Regularization: constraints placed on the recovered model and a priori information
#     3) Optimization: the numerical approach used to solve the inverse problem
#
#

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)

# Define the regularization (model objective function)
reg = regularization.Simple(
    mesh,
    indActive=ind_active,
    mref=starting_conductivity_model,
    alpha_s=0.01,
    alpha_x=1,
    alpha_y=1,
)

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=5, lower=-10.0, upper=2.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

#######################################################################
# Define DC Inversion Directives
# ------------------------------
#
# Here we define any directives that are carried out during the inversion. This
# includes the cooling schedule for the trade-off parameter (beta), stopping
# criteria for the inversion and saving inversion results at each iteration.
#

# Apply and update sensitivity weighting as the model updates
update_sensitivity_weighting = directives.UpdateSensitivityWeights()

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=2e1)

# Set the rate of reduction in trade-off parameter (beta) each time the
# the inverse problem is solved. And set the number of Gauss-Newton iterations
# for each trade-off paramter value.
beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=2)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

directives_list = [
    update_sensitivity_weighting,
    starting_beta,
    beta_schedule,
    save_iteration,
    target_misfit,
]

#####################################################################
# Running the DC Inversion
# ------------------------
#
# To define the inversion object, we need to define the inversion problem and
# the set of directives. We can then run the inversion.
#

# Here we combine the inverse problem and the set of directives
dc_inversion = inversion.BaseInversion(inv_prob, directiveList=directives_list)

# Run inversion
recovered_conductivity_model = dc_inversion.run(starting_conductivity_model)

############################################################
# Plotting True and Recovered Conductivity Model
# ----------------------------------------------
#

# Recreate true conductivity model
true_background_conductivity = 1e-2
true_conductor_conductivity = 1e-1
true_resistor_conductivity = 1e-3

true_conductivity_model = true_background_conductivity * np.ones(len(mesh))

ind_conductor = model_builder.getIndicesSphere(np.r_[-120.0, -180.0], 60.0, mesh.gridCC)
true_conductivity_model[ind_conductor] = true_conductor_conductivity

ind_resistor = model_builder.getIndicesSphere(np.r_[120.0, -180.0], 60.0, mesh.gridCC)
true_conductivity_model[ind_resistor] = true_resistor_conductivity

true_conductivity_model[~ind_active] = np.NaN

# Plot True Model
norm = LogNorm(vmin=1e-3, vmax=1e-1)

fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_axes([0.1, 0.12, 0.72, 0.8])
im = mesh.plot_image(
    true_conductivity_model,
    ax=ax1,
    grid=False,
    range_x=[-700, 700],
    range_y=[-700, 0],
    pcolor_opts={"norm": norm},
)
ax1.set_title("True Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.83, 0.12, 0.05, 0.8])
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical")
cbar.set_label("$S/m$", rotation=270, labelpad=15, size=12)

plt.show()

# # Plot Recovered Model
fig = plt.figure(figsize=(9, 4))

recovered_conductivity = conductivity_map * recovered_conductivity_model
recovered_conductivity[~ind_active] = np.NaN

ax1 = fig.add_axes([0.1, 0.12, 0.72, 0.8])
mesh.plotImage(
    recovered_conductivity,
    normal="Y",
    ax=ax1,
    grid=False,
    range_x=[-700, 700],
    range_y=[-700, 0],
    pcolorOpts={"norm": norm},
)
ax1.set_title("Recovered Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.83, 0.12, 0.05, 0.8])
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical")
cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.show()

###################################################################
# Plotting Predicted DC Data and Misfit
# -------------------------------------
#

# Predicted data from recovered model
dpred = inv_prob.dpred
dc_data_predicted = data.Data(survey, dobs=dpred)

data_array = [dc_data, dc_data_predicted, dc_data]
dobs_array = [None, None, (dobs - dpred) / std]

fig = plt.figure(figsize=(17, 5.5))
plot_title = ["Observed", "Predicted", "Normalized Misfit"]
plot_type = ["appConductivity", "appConductivity", "misfitMap"]
plot_units = ["S/m", "S/m", ""]
scale = ["log", "log", "linear"]

ax1 = 3 * [None]
norm = 3 * [None]
cbar = 3 * [None]
cplot = 3 * [None]

for ii in range(0, 3):

    ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.05, 0.25, 0.9])
    cplot[ii] = plot_pseudosection(
        data_array[ii],
        dobs=dobs_array[ii],
        ax=ax1[ii],
        survey_type="dipole-dipole",
        data_type=plot_type[ii],
        scale=scale[ii],
        space_type="half-space",
        y_values="pseudo-depth",
        pcolor_opts={"cmap": "viridis"},
    )
    ax1[ii].set_title(plot_title[ii])

plt.show()
