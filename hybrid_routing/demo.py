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

from math import atan2, cos, pi, sin, sqrt
from time import sleep

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from hybrid_routing.benchmark import background_vector_field
from hybrid_routing.optimize import optimize_route
from hybrid_routing.plot import plot_vector_field

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

###############
# Coordinates #
###############

X_MIN, X_MAX = -40.0, 130.0
Y_MIN, Y_MAX = -40.0, 130.0
WIDTH = X_MAX - X_MIN
HEIGHT = Y_MAX - Y_MIN

row1col1, row1col2, row1col3, row1col4 = st.columns(4)

with row1col1:
    st.markdown("Start point")
    x_start = st.number_input(
        "X", min_value=X_MIN, max_value=X_MAX, value=X_MIN + WIDTH / 4, key="x_start"
    )
    vel = st.slider(
        "Boat velocity",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        key="velocity",
    )


with row1col2:
    st.markdown("(green)")
    y_start = st.number_input(
        "Y", min_value=Y_MIN, max_value=Y_MAX, value=Y_MIN + HEIGHT / 4, key="y_start"
    )
    step_time = st.slider(
        "Time step", min_value=1, max_value=50, value=20, step=1, key="time"
    )


with row1col3:
    st.markdown("End point")
    x_end = st.number_input(
        "X", min_value=X_MIN, max_value=X_MAX, value=X_MIN + 3 * WIDTH / 4, key="x_end"
    )
    angle = st.slider(
        "Angle amplitude (degrees)",
        min_value=0,
        max_value=100,
        value=25,
        step=1,
        key="angle",
    )


with row1col4:
    st.markdown("(red)")
    y_end = st.number_input(
        "Y", min_value=Y_MIN, max_value=Y_MAX, value=Y_MIN + 3 * HEIGHT / 4, key="y_end"
    )
    num_angles = st.slider(
        "Number of angles", min_value=3, max_value=60, value=10, step=1, key="num_angle"
    )

###########
# Buttons #
###########

row2col1, row2col2 = st.columns(2)

with row2col1:
    do_run = st.button("Run")


########
# Plot #
########


def plot_preview(x1, y1, x2, y2, angle_amplitude):
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

    plot_vector_field(
        background_vector_field, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX
    )
    plt.plot([x1, x2], [y1, y2], "r--", alpha=0.8)
    plt.plot([x1, x_up], [y1, y_up], "g--", alpha=0.4)
    plt.plot([x1, x_down], [y1, y_down], "g--", alpha=0.4)
    plt.scatter(x1, y1, c="g")
    plt.scatter(x2, y2, c="r")
    plt.xlim([X_MIN, X_MAX])
    plt.ylim([Y_MIN, Y_MAX])


fig = plt.figure()
plot = st.pyplot(fig=fig)


if any([x_start, y_start, x_end, y_end, angle]):
    fig = plt.figure()
    plot_preview(x_start, y_start, x_end, y_end, angle)
    plot.pyplot(fig=fig)
    sleep(0.2)

#######
# Run #
#######

if do_run:
    list_x, list_y = [
        x_start,
    ], [y_start]
    for (x, y, theta) in optimize_route(
        x_start,
        y_start,
        x_end,
        y_end,
        step_time=step_time,
        angle_amplitude=angle,
        num_angles=num_angles,
        vel=vel,
    ):
        if x < X_MIN or x > X_MAX or y < Y_MIN or y > Y_MAX:
            break
        list_x.append(x)
        list_y.append(y)
        fig = plt.figure()
        plt.plot(list_x, list_y, color="grey", linestyle="--", marker=".", alpha=0.6)
        plot_preview(x, y, x_end, y_end, angle)
        plot.pyplot(fig=fig)

###########
# Credits #
###########

st.markdown("---")
st.markdown(
    "Original idea by Robert Milson. Code by Louis Bu. Interface by Daniel Precioso."
)
