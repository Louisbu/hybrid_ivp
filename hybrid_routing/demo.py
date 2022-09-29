"""
This is a simple web application to try out the code.
To start the web, run:

streamlit run hybrid_routing/demo.py --server.port 8501

To access it you must first do a bridge in your PC by running:

ssh -f louis@bowie.incubazul.es -L 8501:localhost:8501 -N

Where 8501 is the server port.
The you can access the web in your PC by going to:

http://localhost:8501
"""

import inspect
import sys
from math import atan2, cos, pi, sin, sqrt
from typing import List

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from hybrid_routing.jax_utils.dnj import DNJRandomGuess
from hybrid_routing.jax_utils.dnj import DNJ
from hybrid_routing.jax_utils.optimize import optimize_route
from hybrid_routing.jax_utils.route import RouteJax
from hybrid_routing.vectorfields import *
from hybrid_routing.vectorfields.base import Vectorfield

X_MIN, X_MAX = 0.0, 6.0
Y_MIN, Y_MAX = -1.0, 6.0
X_START, Y_START = 0.0, 0.0
X_END, Y_END = 6.0, 2.0
VEL_MIN, VEL, VEL_MAX = 0.1, 1.0, 2.0
TIME_MIN, TIME, TIME_MAX = 0.1, 0.5, 2.0
NUM_ITER_DNJ = 50
NUM_ITER_DNJ_END = 500

st.set_page_config(
    layout="centered", page_icon="img/dalhousie.png", page_title="Hybrid Routing"
)

#########
# Title #
#########

row0col1, row0col2 = st.columns([1, 2], gap="medium")

with row0col1:
    st.image(Image.open("img/dalhousie.png"))

with row0col2:
    st.title("Hybrid Routing")
    st.markdown("A demostration of the ZIVP and DNJ algorithms.")

st.markdown("---")

################
# Vector field #
################

dict_vectorfields = dict(
    inspect.getmembers(sys.modules["hybrid_routing.vectorfields"], inspect.isclass)
)

vectorfield_name = st.selectbox("Vector field:", sorted(dict_vectorfields.keys()))
rowvcol1, rowvcol2, rowvcol3 = st.columns([1, 2, 1], gap="medium")
with rowvcol1:
    do_discretize = st.checkbox("Discretized", value=False)
with rowvcol2:
    optimize_for = st.selectbox("Optimize for:", ["time", "fuel"])
with rowvcol3:
    use_rk = st.checkbox("Use Runge-Kutta", value=True)

# Initialize vectorfield
vectorfield: Vectorfield = dict_vectorfields[vectorfield_name]()
if do_discretize:
    vectorfield = vectorfield.discretize(
        x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, step=0.1
    )


###############
# Coordinates #
###############

WIDTH = X_MAX - X_MIN
HEIGHT = Y_MAX - Y_MIN

row1col1, row1col2, row1col3, row1col4 = st.columns(4)

with row1col1:
    st.markdown("Start point")
    x_start = st.number_input(
        "X", min_value=X_MIN, max_value=X_MAX, value=X_START, key="x_start"
    )
    vel = st.slider(
        "Boat velocity", min_value=VEL_MIN, max_value=VEL_MAX, value=VEL, key="velocity"
    )


with row1col2:
    st.markdown("(green)")
    y_start = st.number_input(
        "Y", min_value=Y_MIN, max_value=Y_MAX, value=Y_START, key="y_start"
    )
    time_iter = st.slider(
        "Time between decisions",
        min_value=TIME_MIN,
        max_value=TIME_MAX,
        value=TIME,
        key="time",
    )


with row1col3:
    st.markdown("End point")
    x_end = st.number_input(
        "X", min_value=X_MIN, max_value=X_MAX, value=X_END, key="x_end"
    )
    angle = st.slider(
        "Angle amplitude (degrees)",
        min_value=0,
        max_value=180,
        value=120,
        step=1,
        key="angle",
    )


with row1col4:
    st.markdown("(red)")
    y_end = st.number_input(
        "Y", min_value=Y_MIN, max_value=Y_MAX, value=Y_END, key="y_end"
    )
    num_angles = st.slider(
        "Number of angles", min_value=3, max_value=40, value=6, step=1, key="num_angle"
    )

# DNJ
time_step = time_iter / 20
dnj = DNJ(vectorfield=vectorfield, time_step=time_step)

###########
# Buttons #
###########

row2col1, row2col2 = st.columns(2)

with row2col1:
    do_run = st.button("Run")

with row2col2:
    do_run_dnj = st.button("Run only DNJ")


########
# Plot #
########

LIST_PLOT_TEMP: List = []


def remove_plot_lines_temporal():
    for line in LIST_PLOT_TEMP:
        line.pop(0).remove()
    LIST_PLOT_TEMP.clear()


def plot_start_and_goal(x1, y1, x2, y2, angle_amplitude: float):
    dx = x2 - x1
    dy = y2 - y1
    dist = sqrt(dx**2 + dy**2) / 2
    angle_rad = atan2(dy, dx)
    angle_amp_rad = angle_amplitude * pi / 180
    angle_max = angle_rad + angle_amp_rad / 2
    x_up = x1 + dist * cos(angle_max)
    y_up = y1 + dist * sin(angle_max)
    angle_min = angle_rad - angle_amp_rad / 2
    x_down = x1 + dist * cos(angle_min)
    y_down = y1 + dist * sin(angle_min)
    line_up = plt.plot([x1, x_up], [y1, y_up], "g--", alpha=0.4)
    line_down = plt.plot([x1, x_down], [y1, y_down], "g--", alpha=0.4)

    line_center = plt.plot([x1, x2], [y1, y2], "r--", alpha=0.8)
    plt.scatter(x1, y1, c="g")
    plt.scatter(x2, y2, c="r")
    LIST_PLOT_TEMP.extend([line_up, line_down, line_center])


def plot_vectorfield():
    vectorfield.plot(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX)
    plt.xlim([X_MIN, X_MAX])
    plt.ylim([Y_MIN, Y_MAX])
    plt.gca().set_aspect("equal")


fig = plt.figure()
plot = st.pyplot(fig=fig)
plot_vectorfield()
plot_start_and_goal(x_start, y_start, x_end, y_end, angle_amplitude=angle)
plot.pyplot(fig=fig)
remove_plot_lines_temporal()

#######
# Run #
#######

if do_run:
    # Initialize both raw optimized route and optimized route with DNJ
    route_raw = RouteJax(x=x_start, y=y_start, t=0)
    route_dnj = RouteJax(x=x_start, y=y_start, t=0)
    # Build iteration over optimization
    iter_optim = optimize_route(
        vectorfield,
        x_start,
        y_start,
        x_end,
        y_end,
        time_iter=time_iter,
        time_step=time_step,
        angle_amplitude=angle * pi / 180,
        num_angles=num_angles,
        vel=vel,
        use_rk=use_rk,
    )
    # Loop through optimization
    for list_routes in iter_optim:
        # Loop through the route segments
        for idx, route in enumerate(list_routes):
            if idx == 0:
                color = "red"
                # The best route segment is appended to the optimal route
                route_raw.append_points(route.x, route.y, route.t)
                route_dnj.append_points(route.x, route.y, route.t)
            else:
                color = "grey"
            # Plot the route segment
            plt.plot(route.x, route.y, color=color, linestyle="-", alpha=0.4)

        # Apply DNJ to the optimal route
        dnj.optimize_route(route_dnj, num_iter=NUM_ITER_DNJ)
        # Plot both raw and DNJ optimized routes
        line_raw = plt.plot(
            route_raw.x, route_raw.y, color="orange", linestyle="--", alpha=0.6
        )
        line_dnj = plt.plot(
            route_dnj.x, route_dnj.y, color="green", linestyle="--", alpha=0.7
        )
        # Include this lines in the temporal dict, they will be updated in each iteration
        LIST_PLOT_TEMP.extend([line_raw, line_dnj])
        plot.pyplot(fig=fig)
        remove_plot_lines_temporal()

    # Once optimization finishes, append last point
    t_end = 0
    route_raw.append_point_end(x=x_end, y=y_end, vel=vel)
    route_dnj.append_point_end(x=x_end, y=y_end, vel=vel)
    dnj.optimize_route(route, num_iter=NUM_ITER_DNJ_END)

    # Plot both raw and DNJ optimized routes
    plt.plot(route_raw.x, route_raw.y, color="orange", linestyle="--", alpha=0.6)
    plt.plot(route_dnj.x, route_dnj.y, color="green", linestyle="--", alpha=0.7)
    plot.pyplot(fig=fig)
    plt.close(fig)

################
# Run only DNJ #
################

if do_run_dnj:
    # The number of iterations between plots
    # is adjusted depending on the number of routes to explore
    num_iter_plot = int(3000 / num_angles)
    num_iter_gen = int(num_angles / 1.2)
    # Initialize generator
    dnj_random_guess = DNJRandomGuess(
        vectorfield=vectorfield,
        q0=(x_start, y_start),
        q1=(x_end, y_end),
        time_step=time_step,
        optimize_for=optimize_for,
        angle_amplitude=angle * pi / 180,
        num_points=80,
        num_routes=num_angles,
        num_iter=num_iter_plot,
    )
    for iter in range(num_iter_gen):
        list_routes: List[RouteJax] = next(dnj_random_guess)
        for route in list_routes:
            line = plt.plot(route.x, route.y, color="green", linestyle="--", alpha=0.7)
            LIST_PLOT_TEMP.append(line)
        plot.pyplot(fig=fig)
        remove_plot_lines_temporal()
    plt.close(fig)

###########
# Credits #
###########

st.markdown("---")
st.markdown(
    "Original idea by Robert Milson. Code by Louis Bu. Interface by Daniel Precioso."
)
