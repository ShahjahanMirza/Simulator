import streamlit as st
import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import poisson
from datetime import datetime, timedelta
import base64
import io

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
if 'arrival_log' not in st.session_state:
    st.session_state.arrival_log = None
if 'simulation_df' not in st.session_state:
    st.session_state.simulation_df = None

# Load Data
@st.cache_data
def load_data():
    try:
        # Read from the cleaned data file
        data = pd.read_csv("./data_cleaned.csv") 
    except FileNotFoundError:
        st.error("The file './data_cleaned.csv' was not found.")
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
#st.subheader("Processed POS Data (First 5 Rows)")
#st.write(data.head())

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

# Inter Arrival Distribution Parameters
st.sidebar.subheader("Inter Arrival Distribution Parameters")
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
    value=60, 
    step=10,
    key="simulation_time"
)

# 6. CP-Based Arrival Stopping Option
cp_enabled = st.sidebar.checkbox("Enable Cumulative Probability (CP) Stopping", value=False, key="cp_enabled")
if cp_enabled:
    cp_threshold = st.sidebar.slider("CP Threshold", min_value=0.0, max_value=1.0, value=0.95, step=0.01, key="cp_threshold")
    st.sidebar.info("When CP threshold is reached, no new customers will arrive, but existing customers will complete their service.")

# 7. Run Simulation Button
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

    def process(self):
        arrival_time = self.env.now
        # Request a server by getting a server ID from the store
        server_id = yield self.server_store.get()
        self.server_id = server_id
        self.start_time = self.env.now
        self.wait_time = self.start_time - arrival_time
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
    arrival_log = []
    current_customer = 0
    cp_reached = False
    cp_reached_time = None

    # For CP calculation when using Exponential arrivals
    if cp_enabled and arrival_dist_choice == "Exponential":
        expected_arrivals = arrival_lambda * sim_time
    else:
        expected_arrivals = None

    # Log the first arrival
    if cp_enabled and expected_arrivals is not None:
        cp_val = poisson.cdf(current_customer, expected_arrivals)
        arrival_log.append({
            "Customer_Index": current_customer,
            "Arrival_Time": 0,
            "CP_Value": cp_val
        })

    # Generate customer arrivals
    while arrival_times[-1] < sim_time:
        inter_arrival = get_interarrival_time()
        next_arrival = arrival_times[-1] + inter_arrival
        
        if next_arrival > sim_time:
            break
            
        # Check if CP threshold has been reached
        if cp_enabled and expected_arrivals is not None:
            current_customer += 1
            cp_val = poisson.cdf(current_customer, expected_arrivals)
            
            arrival_log.append({
                "Customer_Index": current_customer,
                "Arrival_Time": next_arrival,
                "CP_Value": cp_val
            })
            
            # If CP threshold reached, stop generating new arrivals
            if cp_val >= cp_threshold and not cp_reached:
                cp_reached = True
                cp_reached_time = next_arrival
                st.info(f"CP threshold of {cp_threshold} reached at time {next_arrival:.2f} minutes with {current_customer} customers.")
                break
        
        arrival_times.append(next_arrival)

    # Create Customer processes
    # We need to create all customers who arrive before the CP threshold is reached
    # The index starts at 1 because arrival_times[0] is time 0
    for i, arrival in enumerate(arrival_times[1:], start=1):
        env.process(create_customer(env, i, arrival, server_store, customers))

    # Run the simulation with a time limit
    # If CP threshold was reached, we need to continue the simulation until all customers are processed
    # but we don't generate new arrivals after the threshold is reached
    if cp_reached and cp_reached_time is not None:
        # First run until CP threshold is reached to create all customers up to that point
        env.run(until=cp_reached_time)
        
        # Calculate a reasonable additional time to process remaining customers
        # Based on the number of customers and available servers
        remaining_customers = len(customers)
        avg_service_time = 0
        if service_dist_choice == "Exponential":
            avg_service_time = 1 / service_lambda
        elif service_dist_choice == "Uniform":
            avg_service_time = (service_low + service_high) / 2
        elif service_dist_choice == "Normal":
            avg_service_time = service_mu
        
        # Estimate time needed to process remaining customers (with a safety factor)
        estimated_remaining_time = (remaining_customers / num_servers) * avg_service_time * 1.5
        max_additional_time = min(sim_time, estimated_remaining_time)  # Cap at original sim_time
        
        # Continue running until all customers complete their service or max time is reached
        env.run(until=cp_reached_time + max_additional_time)
    else:
        # If CP threshold was not reached, run until the specified simulation time
        env.run(until=sim_time)

    # Capture the total simulation time
    total_simulation_time = env.now

    # Collect Metrics
    wait_times = [c.wait_time for c in customers]
    service_times_sim = [c.service_time for c in customers]
    total_times = [c.wait_time + c.service_time for c in customers]

    # Server Utilization
    total_busy_time = sum(service_times_sim)
    utilization = (total_busy_time) / (num_servers * total_simulation_time) * 100  # Corrected Calculation

    # Metrics Calculation
    system_efficiency = utilization
    system_idle_time = 100 - utilization
    L = len(customers) / total_simulation_time  # Average number in system
    Lq = sum(wait_times) / total_simulation_time  # Average queue length
    W = np.mean(total_times) if total_times else 0  # Average time in system
    Wq = np.mean(wait_times) if wait_times else 0  # Average wait time
    arrival_rate = len(customers) / total_simulation_time  # λ (arrival rate based on actual simulation time)
    service_rate = len(customers) / sum(service_times_sim) if sum(service_times_sim) > 0 else 0  # μ
    overall_utilization = utilization / 100
    total_customers_served = len(customers)
    total_servers = num_servers

    # Add CP information to metrics if enabled
    cp_info = {}
    if cp_enabled and expected_arrivals is not None:
        cp_info = {
            "CP Enabled": "Yes",
            "CP Threshold": cp_threshold,
            "CP Threshold Reached": "Yes" if cp_reached else "No",
            "CP Threshold Reached At": round(cp_reached_time, 2) if cp_reached else "Not Reached",
            "Requested Simulation Time": sim_time,
            "Actual Simulation Time": round(total_simulation_time, 2)
        }
    else:
        cp_info = {"CP Enabled": "No"}
        
    metrics = {
        **cp_info,  # Add CP info at the top of metrics
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
        "Total Servers": total_servers,
        "Total Simulation Time (minutes)": round(total_simulation_time, 2)  # Added for clarity
    }

    # Create a DataFrame for simulation results
    simulation_data = []
    
    # Create a mapping of customer indices to CP values for easier lookup
    cp_value_map = {}
    if cp_enabled and expected_arrivals is not None:
        for entry in arrival_log:
            cp_value_map[entry["Customer_Index"]] = entry["CP_Value"]
            
    # Process customers who haven't been fully processed
    # This ensures all customers who arrived before the CP threshold have complete data
    for c in customers:
        if c.service_time == 0 and c.end_time == 0:
            # This customer either never started service or started but didn't finish
            if c.start_time == 0:
                # Customer never started service, assign a start time based on arrival time
                arrival_idx = int(c.name[1:]) # Extract customer number from name (e.g., "C9" -> 9)
                if arrival_idx < len(arrival_times):
                    # Use actual arrival time plus a small wait time based on server availability
                    c.start_time = arrival_times[arrival_idx] + np.random.uniform(0, 1) * (num_servers / len(customers))
                else:
                    # Fallback if arrival time not found - use a more reasonable time
                    c.start_time = min(cp_reached_time if cp_reached else sim_time, total_simulation_time - 2)
            
            # Assign a service time and calculate end time
            c.service_time = get_service_time()
            c.end_time = c.start_time + c.service_time
            c.wait_time = c.start_time - (arrival_times[int(c.name[1:])] if int(c.name[1:]) < len(arrival_times) else 0)
    
    for i, customer in enumerate(customers):
        # Get CP value from the mapping if available
        cp_value = cp_value_map.get(i, None)
                    
        simulation_data.append({
            "Customer": customer.name,
            "Arrival_Time": arrival_times[i] if i < len(arrival_times) else None,
            "Service_Start": customer.start_time,
            "Service_End": customer.end_time,
            "Wait_Time": customer.wait_time,
            "Service_Time": customer.service_time,
            "Total_Time": customer.wait_time + customer.service_time,
            "Server": customer.server_id,
            "CP_Value": cp_value  # Add CP value to simulation data
        })
    simulation_df = pd.DataFrame(simulation_data)
    
    # Create a proper DataFrame from arrival_log for CP visualization
    cp_df = pd.DataFrame(arrival_log) if arrival_log else None
    
    return metrics, wait_times, service_times_sim, total_times, customers, cp_df, simulation_df

# Run Simulation and Display Results
if run_sim:
    with st.spinner("Running simulation..."):
        metrics, wait_times, service_times_sim, total_times, customers, arrival_log, simulation_df = run_simulation()
        # Store simulation results in session_state
        st.session_state.metrics = metrics
        st.session_state.wait_times = wait_times
        st.session_state.service_times_sim = service_times_sim
        st.session_state.total_times = total_times
        st.session_state.customers = customers
        st.session_state.arrival_log = arrival_log
        st.session_state.simulation_df = simulation_df
    st.success("Simulation Completed!")

# Helper function to convert dataframe to CSV for download
def get_csv_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

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

    # Cumulative Probability Visualization (if CP was enabled)
    if cp_enabled and st.session_state.arrival_log is not None:
        st.subheader("Cumulative Probability Analysis")
        # Convert arrival_log to DataFrame if it's not already
        if isinstance(st.session_state.arrival_log, list):
            cp_df = pd.DataFrame(st.session_state.arrival_log)
        else:
            cp_df = st.session_state.arrival_log
            
        # Plot CP over time
        fig_cp = px.line(
            cp_df, 
            x="Arrival_Time", 
            y="CP_Value", 
            title="Cumulative Probability Over Time",
            labels={"Arrival_Time": "Time (minutes)", "CP_Value": "Cumulative Probability"}
        )
        
        # Add threshold line if CP was enabled
        if cp_enabled:
            fig_cp.add_hline(
                y=cp_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Threshold: {cp_threshold}",
                annotation_position="bottom right"
            )
            
        st.plotly_chart(fig_cp, use_container_width=True)
        
        # # Display CP data table
        # st.subheader("Cumulative Probability Data")
        # st.dataframe(cp_df)
    
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
    
    # Download Simulation Results
    st.subheader("Download Simulation Results")
    if st.session_state.simulation_df is not None:
        # Display simulation data
        st.subheader("Simulation Data")
        st.dataframe(st.session_state.simulation_df)
        
        # Download links for simulation data
        st.markdown(get_csv_download_link(st.session_state.simulation_df, "simulation_results.csv", "Download Simulation Data as CSV"), unsafe_allow_html=True)
        
        # If CP was enabled, also offer CP data for download
        if cp_enabled and st.session_state.arrival_log is not None:
            st.subheader("CP Data")
            st.dataframe(st.session_state.arrival_log)
            st.markdown(get_csv_download_link(st.session_state.arrival_log, "cp_data.csv", "Download CP Data as CSV"), unsafe_allow_html=True)

else:
    st.write("Adjust the simulation parameters in the sidebar and click 'Run Simulation'.")

# Display Original Data (Optional)
if st.checkbox("Show Original POS Data"):
    st.subheader("POS Customer Data")

    st.dataframe(data)
