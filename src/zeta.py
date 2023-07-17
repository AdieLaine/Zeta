import datetime
import pytz
import ntplib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import lagrange
from scipy.special import comb
from sympy import *
from fractions import Fraction
from numpy.polynomial.polynomial import Polynomial
from scipy.io.wavfile import write
from scipy.signal import sawtooth, square
from io import BytesIO
import streamlit as st
import random
from mpmath import zeta

st.set_page_config(
    page_title='Zeta Function Explorer',
    page_icon="🧮",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/Zeta.git',
        'Report a bug': 'https://github.com/AdieLaine/Zeta.git/issues',
        'About': """

        This Python script uses Streamlit to visualize the behavior of the Riemann Zeta function, a complex-valued function that has important implications in number theory. The script includes the following features:

        Riemann Zeta Function Trajectory: This feature visualizes the trajectory of the Zeta function on the complex plane. It uses the plot_riemann_trajectory function to generate a plotly scatter plot, which represents the real and imaginary parts of the Zeta function values. The plot is colored according to the density of the points.

        Bernoulli Numbers and Pascal's Triangle: This feature shows the relationship between Bernoulli numbers and the coefficients of the binomial expansion, as illustrated by Pascal's Triangle. It uses the plot_pascal_bernoulli function to generate a matplotlib plot that shows the first 10 rows of Pascal's Triangle and the corresponding Bernoulli numbers.

        Zeta Function and Bernoulli Numbers: This feature shows the relationship between the Zeta function and the Bernoulli numbers. It uses the plot_zeta_bernoulli function to generate a plotly line plot of the Zeta function, with markers indicating the values at even positive integers and the corresponding Bernoulli numbers.

        The script is designed to be interactive, allowing the user to specify the number of points to be used in the Zeta function trajectory plot.

        Developer's Note
        
        The logic, theory and mathematics behind this application are based on the use of multidiscplinary tools and solutions. The intent is not to claim any truth or not to claim any new discovery. The intent is to expand on creativity, logic and provide a tool that can be used to explore the Riemann Hypothesis and its connections to other mathematical concepts.
        
        The application aims to provide an intuitive understanding of the Riemann Zeta function and its connections to other mathematical concepts.
        
        Project Repository: [GitHub](https://github.com/AdieLaine/Zeta.git/)

        The Triadic Harmony Hypothesis and Logic use was introduced by: [Madie Laine](https://twitter.com/justmadielaine/)
        """
    }
)

# Title and description
st.markdown('<h1 style="text-align: center; color: Teal; margin-top: -70px;">Zeta Function Explorer</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center;"><strong>Exploring the Riemann Hypothesis</strong></h3>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# Subheader and description
st.subheader('Riemann Zeta Function Trajectory')
st.markdown("This application utilizes the known zeros of the Riemann Zeta Function to chart its trajectory within the complex plane. By initiating this trajectory at a starting point and concluding at an endpoint, we can visually represent the function's path and its corresponding zeros. This visualization provides valuable insights into the distribution of these zeros and their correlation with the critical line, a fundamental concept in understanding the Riemann Hypothesis.")

def display_info():
    with st.expander("Exploring the Riemann Hypothesis?"):
        st.markdown("""
        The [Riemann Hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis), one of the seven [Millennium Prize Problems](https://en.wikipedia.org/wiki/Millennium_Prize_Problems), proposes that all non-trivial zeros of the [Riemann Zeta Function](https://en.wikipedia.org/wiki/Riemann_zeta_function) lie on the critical line in the complex plane, i.e., their real part is 0.5.

        **This application visualizes key properties of the Riemann Zeta function and related concepts that are central to understanding the Riemann Hypothesis:**

        - **Riemann Zeta Function Trajectory:** By visualizing the trajectory of the Zeta function on the complex plane, we can see where the function's values become zero. This can provide intuitive insight into the distribution of the zeros and their relationship to the critical line.

        - **Bernoulli Numbers and Pascal's Triangle:** The Bernoulli numbers are closely related to the values of the Zeta function at negative integers. This plot shows the connection between Bernoulli numbers and Pascal's Triangle, illustrating an important aspect of the function's behavior.

        - **Zeta Function and Bernoulli Numbers:** This plot shows the relationship between the Zeta function and the Bernoulli numbers. By seeing how these values correspond, we can better understand the properties of the Zeta function.

        Please note that the interactive nature of this application allows you to explore these concepts at your own pace and level of detail. This is not intended to be a comprehensive explanation of the Riemann Hypothesis, but rather a tool to help you explore the concepts that are central to understanding it.
                    
        This is intended to serve as part of a larger project that aims to provide an intuitive understanding of the Riemann Zeta function and its connections to other mathematical concepts. We look forward to your feedback and suggestions for improvement!
        """)

def theory_info():
    with st.expander("The Triadic Harmony: A Deeper Perspective"):
        st.markdown(r"""
The Riemann Zeta Function, Bernoulli Numbers, and Pascal's Triangle form a remarkable triad that suggests deeper connections within number theory and mathematics as a whole. Let's take a closer look at the relationships that weave these three mathematical concepts together.

**Riemann Zeta Function**

The Riemann Zeta Function, $\zeta(s)$, is defined for complex numbers $s$ with real part greater than 0. Its definition is given by the series:

$$
\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
$$

For complex numbers $s$ with real part less than 1, it can be analytically continued via the functional equation relating $\zeta(s)$ and $\zeta(1-s)$.

**Bernoulli Numbers and the Riemann Zeta Function**

The Riemann Zeta function at negative integers is intimately related to Bernoulli numbers. More specifically, for $n > 0$, we have $\zeta(1-n) = -\frac{B_n}{n}$, where $B_n$ are the Bernoulli numbers. This relationship arises from the process of analytically continuing the Zeta function to the whole complex plane, excluding the point $s = 1$.

The Bernoulli numbers emerge naturally in the calculus of finite sums, and the Riemann Zeta function extends this to infinite series. Thus, this relationship effectively bridges finite and infinite realms in the context of sums of power series, providing a fascinating insight into how the concept of infinity can be tamed within the world of mathematics.

**Bernoulli Numbers and Pascal's Triangle**

Bernoulli numbers can be generated by a generating function, which is inherently connected to the binomial theorem. This connection provides a pathway linking Bernoulli numbers and Pascal's Triangle, the geometric representation of the coefficients in the binomial theorem.

Moreover, the $n$th Bernoulli number bears a relationship with the sum of entries in the $n$th row of Pascal's Triangle divided by $2^n$. This relationship subtly reveals the intricate interplay between combinatorics (as represented by Pascal's Triangle) and the algebraic properties of number theory (as embodied by Bernoulli numbers).

**Pascal's Triangle and the Riemann Zeta Function**

While a direct connection between Pascal's Triangle and the Riemann Zeta Function might not be immediately apparent, a deeper examination through the lens of combinatorial number theory could reveal potential links. One possible avenue is through the Euler Product representation of the Zeta function, which ties the Zeta function with prime numbers.

The prime numbers, in turn, have known, albeit complex, relationships with binomial coefficients that form Pascal's Triangle. For instance, the Lucas' theorem provides conditions for a binomial coefficient to be divisible by a prime number. This suggests that the coefficients of Pascal's Triangle, prime numbers, and the Zeta function might be interconnected in a non-trivial way.

In conclusion, the "Triadic Harmony" hypothesis proposes that the interconnections among the Riemann Zeta Function, Bernoulli Numbers, and Pascal's Triangle may serve as a fertile ground for deeper mathematical discoveries. Understanding these connections could shed new light on the mysterious Riemann Hypothesis and pave the way for novel insights in number theory. This exploration could be powered by the synergistic interplay of traditional mathematical investigation and innovative AI-based analysis tools.

**Developer's Note**
                    
The logic, theory and mathematics behind this application are based on the use of multidiscplinary tools and solutions. The intent is not to claim any truth or not to claim any new discovery. The intent is to expand on creativity, logic and provide a tool that can be used to explore the Riemann Hypothesis and its connections to other mathematical concepts.
                    
The Triadic Harmony Hypothesis and Logic use was introduced by: [Madie Laine](https://twitter.com/justmadielaine/)
        """)

def future_work():
    with st.expander("Future Work"):
        st.markdown("""
### Future Work

- **AI Agent Tasking:** Integrate AI agents to specific tasks configured and trained on focused disciplines with hybrid overlapping, such as exploring Pascal's Triangle and Geometic Systems.

- **Leveraging Language Models:** Utilize multiple language models for optimal data extraction and analysis.

- **Collaborative Learning and Community Engagement:** Foster collaboration among users to share insights, solve problems, and participate in group projects.

- **Adaptive AI Learning and Feedback:** Implement adaptive learning techniques to tailor the pre-training experience of the AI system and it's scaled learning.
                    
- **AI Agent and Human Collaboration:** Explore the potential of AI Agents and Humans working in a symbiotic way to solve complex problems.

These future directions aim to enhance the learning experience, foster exploration of mathematical concepts, and push the boundaries of logical challenges.
        """)

@st.cache_data
def calculate_day(date: datetime.datetime) -> str:
    """Calculate the day of the week for a given date.

    Args:
        date (datetime.datetime): The date to calculate the day of the week for.

    Returns:
        str: The day of the week.
    """
    d = date.day
    m = (date.month - 2) % 12
    if m < 3:
        m += 10
    else:
        m -= 2
    Y = date.year if date.month > 2 else date.year - 1
    y = Y % 100
    c = Y // 100
    f = d + ((13 * m - 1) // 5) + y + (y // 4) + (-2 * c) + (c // 4)
    day_number = f % 7
    days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    return days[day_number]

@st.cache_resource
def get_current_time() -> datetime.datetime:
    """Get the current time from an NTP server.

    Returns:
        datetime.datetime: The current time.
    """
    ntp_client = ntplib.NTPClient()
    response = ntp_client.request('pool.ntp.org')
    utc_date = datetime.datetime.fromtimestamp(response.tx_time)
    est_date = utc_date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Eastern'))

    hour = est_date.hour
    if 5 <= hour < 12:
        greeting = 'Good Morning!'
    elif 12 <= hour < 18:
        greeting = 'Good Afternoon!'
    elif 18 <= hour < 21:
        greeting = 'Good Evening!'
    else:
        greeting = 'Good Evening!'

    return greeting

greeting = get_current_time()
print(get_current_time())

@st.cache_resource
def zeta_function(s: complex) -> complex:
    """Compute the Riemann Zeta function at a given complex number.

    Args:
        s (complex): The complex number to compute the Zeta function for.

    Returns:
        complex: The value of the Zeta function.
    """
    return zeta(s, tol=1e-100)  # Adjust precision with tol parameter

def plot_riemann_trajectory(start: complex, end: complex, num_points: int) -> str:
    """
    Plot the trajectory of the Riemann Zeta function from a start to an end point in the complex plane. 
    The plot depicts where the function's values become zero, providing insight into the distribution of zeros 
    and their relationship with the critical line, an essential aspect of the Riemann Hypothesis.

    Args:
        start (complex): The start point in the complex plane.
        end (complex): The end point in the complex plane.
        num_points (int): The number of points to compute for the plot.

    Returns:
        str: A string representation of the known zeros in LaTeX format.
    """
    # Generate equally spaced points between start and end
    s_values = np.linspace(start, end, num=num_points)

    # Compute the Zeta function for each point
    zeta_values = [zeta_function(s) for s in s_values]

    # Convert each complex Zeta value to a tuple of (real part, imaginary part)
    zeta_values = [(float(z.real), float(z.imag)) for z in zeta_values]

    # Known zeros in the critical strip (only include those relevant to the start and end points)
    known_zeros = [
        0.5 + 14.134725j, 0.5 - 14.134725j, 
        0.5 + 21.022040j, 0.5 - 21.022040j, 
        0.5 + 25.010858j, 0.5 - 25.010858j,
        0.5 + 30.424876j, 0.5 - 30.424876j, 
        0.5 + 32.935062j, 0.5 - 32.935062j, 
        0.5 + 37.586178j, 0.5 - 37.586178j, 
        0.5 + 40.918719j, 0.5 - 40.918719j, 
        0.5 + 43.327073j, 0.5 - 43.327073j, 
        0.5 + 48.005151j, 0.5 - 48.005151j, 
        0.5 + 49.773832j, 0.5 - 49.773832j
    ]

    zeros_string = ", ".join([f"{z.real} + {z.imag}i" for z in known_zeros])

    # Display known zeros in LaTeX format
    st.info(f"Leveraging mathematical logic, we traverse between known zeros of the Riemann Zeta Function to locate the zeros of the Lagrange polynomial:\n\n{zeros_string}")

    # Initialize list of new zeros found
    zeros = []

    # Find new zeros (crossings of the x-axis) using Lagrange interpolation
    for i in range(2, len(zeta_values)):
        if zeta_values[i-2][1] * zeta_values[i][1] < 0:  # if crosses the x-axis
            lagrange_poly = lagrange([s_values[i-2], s_values[i]], [zeta_values[i-2][1], zeta_values[i][1]])
            poly = Polynomial(lagrange_poly)
            roots = poly.roots()
            zero_roots = [root for root in roots if s_values[i-2] <= root <= s_values[i]]
            for zero in zero_roots:
                if not abs(zero - known_zeros).min() < 1e-6:  # if not a known zero
                    zeros.append(zero)

    # Display information about new zeros found
    st.markdown(f"Using a total of <span style='color: SpringGreen'>{num_points}</span> points, our search found <span style='color: Crimson'>{len(zeros)}</span> new zeros</span> along the specified critical line points.", unsafe_allow_html=True)
    
    # Create the figure
    fig = make_subplots(rows=1, cols=1)

    # Plot the trajectory of the Zeta function
    fig.add_trace(
        go.Scattergl(
            x=[x for x, _ in zeta_values],
            y=[y for _, y in zeta_values],
            mode='markers',
            marker=dict(
                color=np.random.randn(num_points),
                colorscale='rainbow',
                line_width=1
            )
        )
    )

    # Plot the critical line
    fig.add_trace(
        go.Scatter(
            x=[0.5 for _ in zeta_values],
            y=[y for _, y in zeta_values],
            mode='lines',
            line=dict(color='red'),
            name='Critical Line'
        )
    )

    # Update layout of the figure
    fig.update_layout(
        width=700,
        title='Riemann Zeta Function Trajectory',
        xaxis=dict(title='Real Part'),
        yaxis=dict(title='Imaginary Part')
    )
    # Display information about the plot
    st.info("This plot shows the trajectory of the Riemann Zeta Function. The path is traced by varying the input complex number along the line from the 'start' to the 'end' point.")
    # Display the plot
    st.plotly_chart(fig)
    
    return zeros_string

start_point = 0.5 + 14.135j
end_point = 0.5 - 25.011j
zeros = [0.5 + 14.135j, 0.5 - 25.011j]
num_points = st.number_input('Input for the search on the Riemann Zeta Function plot (Enter between 100 and 10,000,000):', min_value=100, value=369, max_value=10000000, step=100000, help='Please enter a value between 100 and 10,000,000.')

with st.spinner("Processing..."):
    if num_points > 3000:
        st.info("This computation involves a large number of points and may take a few moments.")
    for i in range(len(zeros) - 1):
        plot_riemann_trajectory(zeros[i], zeros[i+1], num_points)
st.info("Each plot represents a section of the Riemann Zeta function trajectory between two consecutive zeros. The color of the points indicates the density of the trajectory at that point.")

@st.cache_data
def bernoulli_number(n: int) -> Fraction:
    """
    Calculate the nth Bernoulli number.

    Parameters:
    n (int): The Bernoulli number to calculate.

    Returns:
    Fraction: The nth Bernoulli number.
    """
    B = [0] * (n+1)
    for m in range(n+1):
        B[m] = Fraction(1, m+1)
        for j in range(m, 0, -1):
            B[j-1] = j*(B[j-1] - B[j])
    return B[0]  # Return nth Bernoulli number

def plot_pascal_bernoulli() -> None:
    """
    Generates interactive visualizations of the first rows of Pascal's triangle and their 
    corresponding Bernoulli numbers. The function provides two plots: a line chart showing 
    the rows of Pascal's Triangle, and a scatter plot showcasing the correlation between 
    Bernoulli numbers and Pascal's Triangle. The function also includes a new plot of 
    binomial coefficients as a function of the row number.

    The number of rows to visualize can be adjusted using a slider. 

    No parameters are required to call this function.
    """
    with st.expander("Pascal's Triangle and Bernoulli Numbers Visualizations"):
        rows = st.slider('Number of rows in Pascal\'s Triangle', min_value=10, max_value=50)

        # Initialize the list to store the y-coordinates
        y_coords = []

        for n in range(rows):  # Generate the first "rows" rows of Pascal's triangle
            row = [comb(n, k, exact=True) for k in range(n+1)]
            y_coords.append(row)

        # Convert the list to a pandas DataFrame
        df = pd.DataFrame(y_coords)
        st.line_chart(df)
        st.info(f"This line chart displays the first {rows} rows of Pascal's Triangle. Each line's y-coordinate corresponds to an entry in Pascal's Triangle, while the x-coordinate corresponds to the row number.")

        fig = go.Figure()
        st.markdown("The Connection between Bernoulli Numbers and Pascal's Triangle")
        for n in range(rows):
            fig.add_trace(go.Scatter(
                x=[n]*len(y_coords[n]),
                y=y_coords[n],
                mode='markers',
                name=f"B_{n} = {bernoulli(n)}"
            ))

        st.plotly_chart(fig)
        st.info(f"""This scatter plot displays the correlation between the first {rows} Bernoulli numbers and Pascal's Triangle. 
        The x-coordinate corresponds to the row number in Pascal's triangle, while the y-coordinate corresponds to the binomial coefficient. 
        Each marker on the plot represents a Bernoulli number (B_n), showcasing the remarkable connection between these mathematical concepts.""")

        # New feature: Plot of binomial coefficients as a function of the row number
        binomial_coefficients = [comb(rows, k, exact=True) for k in range(rows+1)]
        fig2 = go.Figure(data=go.Scatter(x=list(range(rows+1)), y=binomial_coefficients, mode='lines+markers'))
        fig2.update_layout(title='Binomial Coefficients as a function of the row number',
                           xaxis_title='Row number',
                           yaxis_title='Binomial Coefficient')
        st.plotly_chart(fig2)

#-- Audio Processing --#
scale = [] 
for k in range(35, 65): 
    note = 440 * 2 ** ((k - 49) / 12)
    if k % 12 != 0 and k % 12 != 2 and k % 12 != 5 and k % 12 != 7 and k % 12 != 10:
        scale.append(note)  # add musical note (skip half tones)
n_notes = len(scale)  # number of musical notes

# Function to generate a sine wave
def sine_wave(frequency, duration):
    t = np.linspace(0, duration, int(44100 * duration))
    return np.sin(2 * np.pi * frequency * t)

# Function to generate a triangle wave
def triangle_wave(frequency, duration):
    t = np.linspace(0, duration, int(44100 * duration))
    return 2 * np.abs(sawtooth(2 * np.pi * frequency * t)) - 1

# Function to generate a square wave
def square_wave(frequency, duration):
    t = np.linspace(0, duration, int(44100 * duration))
    return square(2 * np.pi * frequency * t)

# Function to generate a sawtooth wave
def sawtooth_wave(frequency, duration):
    t = np.linspace(0, duration, int(44100 * duration))
    return sawtooth(2 * np.pi * frequency * t)

# Function to generate audio
def generate_audio(zeros, duration_factor, zero_offset, wave_function, play_mode):
    sound = np.array([])
    zeros_np = np.array(zeros)  # convert list to numpy array
    zeros_np += zero_offset
    min_zero = np.min([np.abs(z.imag) for z in zeros_np] + [np.abs(z.real) for z in zeros_np])
    max_zero = np.max([np.abs(z.imag) for z in zeros_np] + [np.abs(z.real) for z in zeros_np])

    # Adjust zeros according to the selected play mode
    if play_mode == 'Reverse':
        zeros_np = zeros_np[::-1]
    elif play_mode == 'Random shuffle':
        np.random.shuffle(zeros_np)

    for zero in zeros_np:
        if play_mode == "Random frequency":
            frequency = random.choice(scale)
        else:
            frequency = scale[int(0.999 * n_notes * (np.abs(zero.imag) - min_zero) / (max_zero - min_zero))]

        if play_mode == "Random duration":
            duration = random.uniform(0.1, 1.0) * duration_factor
        else:
            duration = np.abs(0.1 + 0.4 * (np.abs(zero.real) - min_zero) / (max_zero - min_zero)) * duration_factor

        tone = wave_function(frequency, duration)
        sound = np.concatenate((sound, tone))

    sound *= 32767 / np.max(np.abs(sound))
    sound = sound.astype(np.int16)
    return sound

with st.expander("Listen to the Riemann Zeta Function"):
    st.info("We generate an audio file by playing a sequence of tones, where each tone is determined by the zeros of the Riemann Zeta function. Sound is an amazing ensemble of zero's and one's, off and on. The frequency of each tone is determined by the distance of the zero from the origin, while the duration of each tone is determined by the distance of the zero from the imaginary axis. The zeros of the Riemann Zeta function are displayed in the complex plane below. You can adjust the duration of the audio, the offset from the defined zeros, and the play mode.")

    wave_dict = {'Sine Wave': sine_wave, 'Triangle': triangle_wave, 'Sawtooth': sawtooth_wave, 'Square': square_wave}
    wave_type = st.selectbox('Select wave type:', list(wave_dict.keys()), help='Select the type of wave to use for the audio.')

    duration_factor = st.slider('Duration factor:', min_value=0.1, max_value=3.0, value=1.0, help='Adjust the duration of the audio. This will change the duration of each tone.')

    zero_offset = st.slider('Zero offset:', min_value=-50.0, max_value=50.0, value=0.0, help='Adjust the offset of the zeros from the origin. This will change the frequency of the tones.')

    st.info("You can also select a creative play mode, which will adjust the frequency and duration of each tone in a creative way.")
    play_mode = st.selectbox('Select play mode:', ['Normal', 'Reverse', 'Random shuffle', 'Random frequency', 'Random duration'], help='Select the play mode for the audio.')

    zeros = [0.5 + 14.134725j, 0.5 - 14.134725j,
            0.5 + 21.02204j, 0.5 - 21.02204j,
            0.5 + 25.010858j, 0.5 - 25.010858j,
            0.5 + 30.424876j, 0.5 - 30.424876j,
            0.5 + 32.935062j, 0.5 - 32.935062j,
            0.5 + 37.586178j, 0.5 - 37.586178j,
            0.5 + 40.918719j, 0.5 - 40.918719j,
            0.5 + 43.327073j, 0.5 - 43.327073j,
            0.5 + 48.005151j, 0.5 - 48.005151j,
            0.5 + 49.773832j, 0.5 - 49.773832j]

    audio = generate_audio(zeros, duration_factor, zero_offset, wave_dict[wave_type], play_mode)

    audio_io = BytesIO()
    write(audio_io, 44100, audio)

    st.audio(audio_io.getvalue(), format='audio/wav')

def plot_zeta_bernoulli() -> None:
    """
    Generates an interactive plot of the Riemann Zeta function, marking the values at even positive integers 
    and their corresponding Bernoulli numbers. The function provides a line plot showing the Riemann Zeta function, 
    with markers representing Bernoulli numbers at even integers. 

    The function also includes two new features: a plot of the real and imaginary parts of the Zeta function in 
    the complex plane for a range of inputs, and an optional visualization of the known zeros of the Riemann Zeta function.

    The range of inputs for the Zeta function can be adjusted using a slider. 

    No parameters are required to call this function.
    """
    with st.expander("Zeta Function in Complex Plane and Bernoulli Numbers Plot"):
        max_x = st.slider('Maximum x value', min_value=10, max_value=50, step=1)

        # New feature: Plot of the real and imaginary parts of the Zeta function in the complex plane
        complex_x = np.linspace(0.5, max_x, 50)
        complex_y = np.linspace(-max_x, max_x, 100)
        X, Y = np.meshgrid(complex_x, complex_y)
        Z = X + 1j * Y

        zeta_real = np.empty(Z.shape)
        zeta_imag = np.empty(Z.shape)

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                zeta_value = zeta(Z[i, j])
                zeta_real[i, j] = np.real(zeta_value)
                zeta_imag[i, j] = np.imag(zeta_value)

        fig2 = go.Figure(data=[
            go.Surface(x=X, y=Y, z=zeta_real, name='Real part', colorscale='Rainbow'),
            go.Surface(x=X, y=Y, z=zeta_imag, name='Imaginary part', showscale=False, colorscale='Rainbow')
        ])
        fig2.update_layout(title='Real and Imaginary Parts of the Zeta Function in the Complex Plane', autosize=True)
        st.plotly_chart(fig2)
        st.info("""This plot shows the real and imaginary parts of the Riemann Zeta function in the complex plane. 
                 The rainbow-colored surfaces represent the real and imaginary parts of the Zeta function for a range of inputs. 
                 The x and y coordinates represent the real and imaginary parts of the input, and the z coordinate represents the real or imaginary part of the output.""")

        show_zeros = st.radio("Show known zeros of the Zeta function?", ("No", "Yes"))
        if show_zeros == "Yes":
            # Add a visual representation of the known zeros of the Zeta function
            zeros = [0.5 + 14.134725j, 0.5 - 14.134725j,
                     0.5 + 21.02204j, 0.5 - 21.02204j,
                     0.5 + 25.010858j, 0.5 - 25.010858j,
                     0.5 + 30.424876j, 0.5 - 30.424876j,
                     0.5 + 32.935062j, 0.5 - 32.935062j,
                     0.5 + 37.586178j, 0.5 - 37.586178j,
                     0.5 + 40.918719j, 0.5 - 40.918719j,
                     0.5 + 43.327073j, 0.5 - 43.327073j,
                     0.5 + 48.005151j, 0.5 - 48.005151j,
                     0.5 + 49.773832j, 0.5 - 49.773832j]

            for zero in zeros:
                fig2.add_trace(go.Scatter3d(x=[np.real(zero)], y=[np.imag(zero)], z=[0], mode='markers', marker=dict(size=5, color='red')))
            st.plotly_chart(fig2)

        x = np.arange(0.5, max_x, 0.1)  # Start at 0.5 instead of 1
        y = [float(zeta(xi)) for xi in x]

        # Create a trace for the zeta function
        zeta_trace = go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Zeta function'
        )

        # Create traces for the points at even integers and corresponding Bernoulli numbers
        point_traces = [
            go.Scatter(
                x=[n],
                y=[float(zeta(n))],
                mode='markers',
                name=f'B_{n} = {bernoulli(n)}',
                marker=dict(size=10)
            )
            for n in range(2, max_x if max_x % 2 == 0 else max_x - 1, 2)
        ]

        # Combine the traces
        data = [zeta_trace] + point_traces

        # Define the layout
        layout = go.Layout(
            title="Visualizing the Relationship Between the Riemann Zeta Function and Bernoulli Numbers",
            xaxis=dict(title="Input to zeta function"),
            yaxis=dict(title="Output of zeta function")
        )

        # Create and show the figure
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)
        st.info("This plot shows the relationship between the Riemann Zeta function and the Bernoulli numbers. The line represents the Zeta function, and the markers represent the Bernoulli numbers. As the input to the Zeta function increases, the output approaches the corresponding Bernoulli number.")

def generate_pascal_triangle(n):
    triangle = np.zeros((n, n))
    triangle[:, 0] = 1

    for i in range(1, n):
        for j in range(1, i+1):
            triangle[i, j] = triangle[i - 1, j - 1] + triangle[i - 1, j]
            
    return triangle

with st.expander("Interactive Pascal's Triangle Visualization"):
    rows = st.number_input('Enter the number of rows for Pascal\'s Triangle:', min_value=1, max_value=500, value=10)
    triangle = generate_pascal_triangle(rows)

    # Create x and y coordinates
    x = np.repeat(np.arange(rows), rows)
    y = np.tile(np.arange(rows), rows)

    # Create 3D scatter plot
    scatter = go.Scatter3d(x=x, y=y, z=triangle.flatten(),
                           mode='lines+markers',
                           marker=dict(
                                size=3,
                                color=triangle.flatten(),  
                                colorscale='rainbow',
                                opacity=1.0))

    # Add plots to figure
    fig = go.Figure(data=[scatter])

    # Update layout for better visualization
    fig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Value'),
                      width=600,
                      margin=dict(r=20, b=10, l=10, t=10))
    
    st.plotly_chart(fig)

def main():
    plot_pascal_bernoulli()
    plot_zeta_bernoulli()
    display_info()
    theory_info()
    future_work()

if __name__ == "__main__":
    main()
# gloqowej