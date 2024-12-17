import streamlit as st
import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chisquare
from datetime import datetime, timedelta
from io import BytesIO

# Ensure that openpyxl is installed for Excel operations
# You can install it using: pip install openpyxl

# Set Streamlit page configuration
st.set_page_config(page_title="Supermarket POS Customer Simulation", layout="wide")

# Title
st.title("Supermarket POS Customer Simulation")

# Initialize Session State for Simulation Results
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'customers' not in st.session_state:
    st.session_state.customers = None
if 'wait_times' not in st.session_state:
    st.session_state.wait_times = None
if 'service_times_sim' not in st.session_state:
    st.session_state.service_times_sim = None
if 'total_times' not in st.session_state:
    st.session_state.total_times = None

# Load Data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./pos_data.csv")
    except FileNotFoundError:
        st.error("The file './pos_data.csv' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while reading the data: {e}")
        st.stop()

    # Identify the correct time columns based on your data
    time_columns = ['arrival', 'service start', 'service end']

    for col in time_columns:
        if col in data.columns:
            # Attempt to parse with seconds first
            data[col] = pd.to_datetime(data[col], format='%I:%M:%S %p', errors='coerce')

            # If parsing failed, try without seconds
            if data[col].isna().any():
                data[col] = pd.to_datetime(data[col], format='%I:%M %p', errors='coerce')

            # Final check for any NaT values
            if data[col].isna().any():
                st.warning(f"Some entries in '{col}' could not be parsed and were set to NaT.")
        else:
            st.error(f"Column '{col}' not found in the data.")
            st.stop()

    # Calculate service_time and wait_time in data
    required_columns = ['service start', 'service end', 'arrival']
    if all(col in data.columns for col in required_columns):
        data['service_time'] = (data['service end'] - data['service start']).dt.total_seconds() / 60
        data['wait_time'] = (data['service start'] - data['arrival']).dt.total_seconds() / 60

        # Replace negative wait times with 0
        data['wait_time'] = data['wait_time'].apply(lambda x: x if x >= 0 else 0)

        # Replace NaN wait_time and service_time with 0
        data['wait_time'] = data['wait_time'].fillna(0)
        data['service_time'] = data['service_time'].fillna(0)
    else:
        st.error("Required columns for calculating service_time and wait_time are missing.")
        st.stop()

    return data

data = load_data()

# Debugging: Show data after processing
st.subheader("Processed POS Data (First 5 Rows)")
st.write(data.head())

# Sidebar for Simulation Parameters
st.sidebar.header("Simulation Parameters")

# 1. Choose Queuing Model
model_options = ["M/M/1", "M/M/c", "M/G/1"]
model_choice = st.sidebar.selectbox("Select Queuing Model", model_options, key="queuing_model")

# 2. Number of Servers
if model_choice in ["M/M/c", "M/G/1"]:
    num_servers = st.sidebar.number_input(
        "Number of Servers", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1,
        key="num_servers"
    )
else:
    num_servers = 1

# 3. Arrival Distribution
arrival_dist_options = ["Exponential", "Uniform", "Normal"]
arrival_dist_choice = st.sidebar.selectbox("Arrival Time Distribution", arrival_dist_options, key="arrival_distribution")

# Arrival Distribution Parameters
st.sidebar.subheader("Arrival Distribution Parameters")
if arrival_dist_choice == "Exponential":
    arrival_lambda = st.sidebar.number_input(
        "Lambda (λ)", 
        min_value=0.1, 
        max_value=10.0, 
        value=0.5, 
        step=0.1,
        key="arrival_lambda_exponential"
    )
elif arrival_dist_choice == "Uniform":
    arrival_low = st.sidebar.number_input(
        "Low (min)", 
        min_value=0.0, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        key="arrival_low_uniform"
    )
    arrival_high = st.sidebar.number_input(
        "High (max)", 
        min_value=0.0, 
        max_value=20.0, 
        value=5.0, 
        step=0.1,
        key="arrival_high_uniform"
    )
    # Validation: Ensure High > Low
    if arrival_high <= arrival_low:
        st.sidebar.error("High must be greater than Low for Uniform Distribution.")
elif arrival_dist_choice == "Normal":
    arrival_mu = st.sidebar.number_input(
        "Mean (μ)", 
        min_value=0.0, 
        max_value=20.0, 
        value=5.0, 
        step=0.1,
        key="arrival_mu_normal"
    )
    arrival_sigma = st.sidebar.number_input(
        "Standard Deviation (σ)", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        key="arrival_sigma_normal"
    )

# 4. Service Distribution
service_dist_options = ["Exponential", "Uniform", "Normal"]
service_dist_choice = st.sidebar.selectbox("Service Time Distribution", service_dist_options, key="service_distribution")

# Service Distribution Parameters
st.sidebar.subheader("Service Distribution Parameters")
if service_dist_choice == "Exponential":
    service_lambda = st.sidebar.number_input(
        "Lambda (λ)", 
        min_value=0.1, 
        max_value=10.0, 
        value=0.5, 
        step=0.1,
        key="service_lambda_exponential"
    )
elif service_dist_choice == "Uniform":
    service_low = st.sidebar.number_input(
        "Low (min)", 
        min_value=0.0, 
        max_value=20.0, 
        value=2.0, 
        step=0.1,
        key="service_low_uniform"
    )
    service_high = st.sidebar.number_input(
        "High (max)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0, 
        step=0.1,
        key="service_high_uniform"
    )
    # Validation: Ensure High > Low
    if service_high <= service_low:
        st.sidebar.error("High must be greater than Low for Uniform Distribution.")
elif service_dist_choice == "Normal":
    service_mu = st.sidebar.number_input(
        "Mean (μ)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0, 
        step=0.1,
        key="service_mu_normal"
    )
    service_sigma = st.sidebar.number_input(
        "Standard Deviation (σ)", 
        min_value=0.1, 
        max_value=15.0, 
        value=1.0, 
        step=0.1,
        key="service_sigma_normal"
    )

# 5. Simulation Time
sim_time = st.sidebar.number_input(
    "Total Simulation Time (minutes)", 
    min_value=10, 
    max_value=1000, 
    value=10,  # Adjusted to 10 as per user parameters
    step=10,
    key="simulation_time"
)

# 6. Run Simulation Button
run_sim = st.sidebar.button("Run Simulation", key="run_simulation_button")

# Function to generate inter-arrival times based on distribution
def get_interarrival_time():
    if arrival_dist_choice == "Exponential":
        return np.random.exponential(1 / arrival_lambda)
    elif arrival_dist_choice == "Uniform":
        return np.random.uniform(arrival_low, arrival_high)
    elif arrival_dist_choice == "Normal":
        return max(0, np.random.normal(arrival_mu, arrival_sigma))

# Function to generate service times based on distribution
def get_service_time():
    if service_dist_choice == "Exponential":
        return np.random.exponential(1 / service_lambda)
    elif service_dist_choice == "Uniform":
        return np.random.uniform(service_low, service_high)
    elif service_dist_choice == "Normal":
        return max(0, np.random.normal(service_mu, service_sigma))

# Simulation Environment
class Customer:
    def __init__(self, env, name, server_store):
        self.env = env
        self.name = name
        self.server_store = server_store
        self.server_id = None
        self.wait_time = 0
        self.service_time = 0
        self.start_time = 0
        self.end_time = 0
        self.arrival_time = 0  # New attribute to store arrival time

    def process(self):
        self.arrival_time = self.env.now  # Record arrival time
        # Request a server by getting a server ID from the store
        server_id = yield self.server_store.get()
        self.server_id = server_id
        self.start_time = self.env.now
        self.wait_time = self.start_time - self.arrival_time
        service_time = get_service_time()
        self.service_time = service_time
        # Simulate service time
        yield self.env.timeout(service_time)
        self.end_time = self.env.now
        # Release the server back to the store
        yield self.server_store.put(self.server_id)

# Function to initialize the server store
def init_server_store(env, server_store, num_servers):
    for i in range(1, num_servers + 1):
        yield server_store.put(i)

# Helper function to create a customer process
def create_customer(env, i, arrival_time, server_store, customers):
    yield env.timeout(arrival_time - env.now)
    customer = Customer(env, f"C{i}", server_store)
    customers.append(customer)
    env.process(customer.process())

# Function to run the simulation
def run_simulation():
    env = simpy.Environment()
    server_store = simpy.Store(env, capacity=num_servers)
    # Initialize the server store with server IDs
    env.process(init_server_store(env, server_store, num_servers))

    customers = []
    # Start arrivals at time=0
    arrival_times = [0]

    # Generate customer arrivals
    while arrival_times[-1] < sim_time:
        inter_arrival = get_interarrival_time()
        next_arrival = arrival_times[-1] + inter_arrival
        if next_arrival > sim_time:
            break
        arrival_times.append(next_arrival)

    # Create Customer processes
    for i, arrival in enumerate(arrival_times[1:], start=1):
        env.process(create_customer(env, i, arrival, server_store, customers))

    # Run the simulation
    env.run()

    # Collect Metrics
    wait_times = [c.wait_time for c in customers]
    service_times_sim = [c.service_time for c in customers]
    total_times = [c.wait_time + c.service_time for c in customers]

    # Server Utilization
    total_busy_time = sum(service_times_sim)
    utilization = (total_busy_time) / (num_servers * total_simulation_time) * 100

    # Metrics Calculation
    system_efficiency = utilization
    system_idle_time = 100 - utilization
    L = len(customers) / total_simulation_time  # Average number in system
    Lq = sum(wait_times) / total_simulation_time  # Average queue length
    W = np.mean(total_times) if total_times else 0  # Average time in system
    Wq = np.mean(wait_times) if wait_times else 0  # Average wait time
    arrival_rate = len(customers) / sim_time  # λ
    service_rate = len(customers) / sum(service_times_sim) if sum(service_times_sim) > 0 else 0  # μ
    overall_utilization = utilization / 100
    total_customers_served = len(customers)
    total_servers = num_servers

    metrics = {
        "System Efficiency (%)": round(system_efficiency, 2),
        "System Idle Time (%)": round(system_idle_time, 2),
        "System L (Avg Number in System)": round(L, 2),
        "System Lq (Avg Queue Length)": round(Lq, 2),
        "System W (Avg Time in System)": round(W, 2),
        "System Wq (Avg Wait Time)": round(Wq, 2),
        "System λ (Arrival Rate)": round(arrival_rate, 2),
        "System μ (Service Rate)": round(service_rate, 2),
        "System ρ (Overall Utilization)": round(overall_utilization, 2),
        "Total Customers Served": total_customers_served,
        "Total Servers": total_servers
    }

    return metrics, wait_times, service_times_sim, total_times, customers

# Function to save customer data to Excel
def save_customer_data(customers):
    # Prepare data
    customer_records = []
    for c in customers:
        record = {
            "Name": c.name,
            "Server ID": c.server_id,
            "Arrival Time (min)": round(c.arrival_time, 2),
            "Service Start Time (min)": round(c.start_time, 2),
            "Service End Time (min)": round(c.end_time, 2),
            "Wait Time (min)": round(c.wait_time, 2),
            "Service Time (min)": round(c.service_time, 2)
        }
        customer_records.append(record)

    df_customers = pd.DataFrame(customer_records)

    # Generate filename with current datetime
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"customer_data_{current_datetime}.xlsx"

    # Save to Excel
    try:
        df_customers.to_excel(filename, index=False)
        st.success(f"Customer data successfully saved to '{filename}'!")
    except Exception as e:
        st.error(f"An error occurred while saving the Excel file: {e}")

# Run Simulation and Display Results
if run_sim:
    with st.spinner("Running simulation..."):
        metrics, wait_times, service_times_sim, total_times, customers = run_simulation()
        # Store simulation results in session_state
        st.session_state.metrics = metrics
        st.session_state.wait_times = wait_times
        st.session_state.service_times_sim = service_times_sim
        st.session_state.total_times = total_times
        st.session_state.customers = customers
    st.success("Simulation Completed!")

    # Save Customer Data to Excel
    save_customer_data(customers)

    # Prepare data for download
    try:
        # Combine all customer data into a DataFrame
        customer_records = []
        for c in st.session_state.customers:
            record = {
                "Name": c.name,
                "Server ID": c.server_id,
                "Arrival Time (min)": round(c.arrival_time, 2),
                "Service Start Time (min)": round(c.start_time, 2),
                "Service End Time (min)": round(c.end_time, 2),
                "Wait Time (min)": round(c.wait_time, 2),
                "Service Time (min)": round(c.service_time, 2)
            }
            customer_records.append(record)

        df_customers = pd.DataFrame(customer_records)

        # Convert DataFrame to Excel in memory
        output = BytesIO()
        df_customers.to_excel(output, index=False)
        excel_data = output.getvalue()

        # Generate download button with current datetime in filename
        download_filename = f"customer_data_{current_datetime}.xlsx"

        st.download_button(
            label="Download Customer Data as Excel",
            data=excel_data,
            file_name=download_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"An error occurred while preparing the download: {e}")

# Display Metrics if available
if st.session_state.metrics:
    st.subheader("Simulation Metrics")
    metrics_df = pd.DataFrame(st.session_state.metrics.items(), columns=["Metric", "Value"])
    st.table(metrics_df)

    # Utilization Graph
    st.subheader("Service Time Distribution (Simulation)")
    if st.session_state.service_times_sim:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(st.session_state.service_times_sim, bins=20, kde=True, ax=ax, color='skyblue')
        ax.set_xlabel("Service Time (minutes)")
        ax.set_ylabel("Frequency")
        ax.set_title("Service Time Distribution (Simulation)")
        st.pyplot(fig)
    else:
        st.write("No service time data available for visualization.")

    # Gantt Chart (Optional)
    gantt_checkbox = st.checkbox("Show Gantt Chart")
    if gantt_checkbox:
        st.subheader("Gantt Chart of Customer Service")
        try:
            gantt_data = []
            base_time = datetime(2024, 1, 1, 8, 0, 0)  # Arbitrary base time
            for c in st.session_state.customers:
                # Ensure that start_time and end_time are valid
                if c.end_time >= c.start_time:
                    start_time = base_time + timedelta(minutes=c.start_time)
                    finish_time = base_time + timedelta(minutes=c.end_time)
                    gantt_data.append({
                        "Task": c.name,
                        "Start": start_time,
                        "Finish": finish_time,
                        "Resource": f"Server {c.server_id}"
                    })

            if gantt_data:
                df_gantt = pd.DataFrame(gantt_data)
                fig_gantt = px.timeline(
                    df_gantt, 
                    x_start='Start', 
                    x_end='Finish', 
                    y='Resource', 
                    color='Resource', 
                    hover_name='Task',
                    title='Gantt Chart of Customer Service'
                )
                fig_gantt.update_yaxes(categoryorder='total ascending')
                fig_gantt.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Server",
                    title="Gantt Chart of Customer Service",
                    showlegend=False
                )
                # Use a separate container to prevent resetting the app
                with st.container():
                    st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.write("No data available for Gantt Chart.")
        except Exception as e:
            st.error(f"An error occurred while generating the Gantt Chart: {e}")

    # Chi-square Test
    st.subheader("Chi-square Test")
    try:
        # Define number of bins
        num_bins = 10

        # Create histograms
        sim_hist, bin_edges = np.histogram(st.session_state.service_times_sim, bins=num_bins)
        data_service_times = data['service_time'].dropna()
        data_hist, _ = np.histogram(data_service_times, bins=bin_edges)

        # Check if data_hist.sum() is zero
        if data_hist.sum() == 0:
            st.error("No service time data available for Chi-square Test.")
        else:
            # Scale data_hist to have the same total as sim_hist
            scale_factor = sim_hist.sum() / data_hist.sum()
            f_exp = data_hist * scale_factor

            # Round f_exp to nearest integer
            f_exp = np.round(f_exp).astype(int)

            # Adjust the difference to make sums equal
            difference = sim_hist.sum() - f_exp.sum()
            if difference > 0:
                # Add the difference to the bin with the highest expected frequency
                f_exp[np.argmax(f_exp)] += difference
            elif difference < 0:
                # Subtract the difference from the bin with the highest expected frequency
                f_exp[np.argmax(f_exp)] += difference  # difference is negative

            # Check for expected frequencies <5
            if np.any(f_exp < 5):
                st.warning("Some expected frequencies are less than 5. Chi-square test may not be valid.")

            # Ensure that the sums match exactly
            if f_exp.sum() != sim_hist.sum():
                st.warning("Observed and expected frequencies do not sum to the same total after scaling.")

            # Perform Chi-square test
            chi2_stat, p_val = chisquare(f_obs=sim_hist, f_exp=f_exp)

            chi2_results = {
                "Chi-square Statistic": round(chi2_stat, 2),
                "P-value": round(p_val, 4)
            }

            chi2_df = pd.DataFrame(chi2_results, index=[0])

            st.table(chi2_df)
    except Exception as e:
        st.error(f"An error occurred during the Chi-square Test: {e}")

    # Data Mean Comparison
    st.subheader("Data Mean Comparison")
    try:
        # Calculate mean wait time and service time from data
        data_mean_wait = data['wait_time'].mean()
        data_mean_service = data['service_time'].mean()

        # Check if data_mean_wait or data_mean_service is NaN
        if np.isnan(data_mean_wait) or np.isnan(data_mean_service):
            st.error("Cannot compute mean wait_time or service_time from the data.")
        else:
            # Simulation mean wait time and service time from metrics
            sim_mean_wait = st.session_state.metrics.get("System Wq (Avg Wait Time)", 0)
            sim_mean_service = st.session_state.metrics.get("System W (Avg Time in System)", 0)

            comparison_df = pd.DataFrame({
                "Metric": ["Average Wait Time (minutes)", "Average Service Time (minutes)"],
                "Simulation": [round(sim_mean_wait, 2), round(sim_mean_service, 2)],
                "Real Data": [round(data_mean_wait, 2), round(data_mean_service, 2)]
            })

            st.table(comparison_df)

            # Visualization of Mean Comparison
            fig_comparison = px.bar(
                comparison_df.melt(id_vars="Metric", var_name="Source", value_name="Value"),
                x="Metric",
                y="Value",
                color="Source",
                barmode='group',
                title="Comparison of Average Times"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred during Data Mean Comparison: {e}")

else:
    st.write("Adjust the simulation parameters in the sidebar and click 'Run Simulation'.")

# Display Original Data (Optional)
if st.checkbox("Show Original POS Data"):
    st.subheader("POS Customer Data")
    st.dataframe(data)
