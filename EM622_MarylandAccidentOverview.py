import pandas as pd
import matplotlib.pyplot as plt

# Load the data with low_memory=False to avoid the DtypeWarning about mixed types
file_path = 'EM622_FinalsDataSet_ZengK.csv'
df = pd.read_csv(file_path, low_memory=False)

# Convert 'Crash Date/Time' to datetime and extract the year
df['Crash Date/Time'] = pd.to_datetime(df['Crash Date/Time'], errors='coerce')
df['Year'] = df['Crash Date/Time'].dt.year

# Filter dataset for the years 2015-2023, ensuring no NaT or invalid years
df_filtered = df[df['Year'].between(2015, 2023, inclusive=True)].dropna(subset=['Year'])

# Handle NaN values in 'Weather' column by filling them with 'Unknown'
df_filtered['Weather'] = df_filtered['Weather'].fillna('Unknown')

# Standardize the weather condition labels to a consistent format (e.g., title case)
df_filtered['Weather'] = df_filtered['Weather'].str.title()

# Filter the dataset to include only the relevant weather conditions
valid_weather_conditions = ['Clear', 'Cloudy', 'Raining', 'Snow']
df_filtered = df_filtered[df_filtered['Weather'].isin(valid_weather_conditions)]

# Normalize Injury Severity to lowercase and group them
df_filtered['Injury Severity'] = df_filtered['Injury Severity'].str.lower()

# Define Injury Severity groups
severity_mapping = {
    'fatal injury': 'Fatal Injury',
    'no apparent injury': 'No Injury',
    'possible injury': 'No Injury',
    'suspected minor injury': 'Injured',
    'suspected serious injury': 'Injured'
}

# Apply the grouping
df_filtered['Injury Severity'] = df_filtered['Injury Severity'].map(severity_mapping).fillna('Other')

# Collision type group mapping
collision_group_map = {
    'SAME DIR REAR END': 'Rear-End Collisions',
    'SAME DIR BOTH LEFT TURN': 'Rear-End Collisions',
    'ANGLE MEETS LEFT TURN': 'Rear-End Collisions',
    'HEAD ON': 'Head-On Collisions',
    'HEAD ON LEFT TURN': 'Head-On Collisions',
    'ANGLE MEETS RIGHT TURN': 'Head-On Collisions',
    'SAME DIRECTION SIDESWIPE': 'Side-Swipe Collisions',
    'OPPOSITE DIRECTION SIDESWIPE': 'Side-Swipe Collisions',
    'SAME DIRECTION LEFT TURN': 'Side-Swipe Collisions',
    'SAME DIRECTION RIGHT TURN': 'Side-Swipe Collisions',
    'SINGLE VEHICLE': 'Other / Miscellaneous',
    'OTHER': 'Other / Miscellaneous',
    'UNKNOWN': 'Other / Miscellaneous',
    'N/A': 'Other / Miscellaneous'
}

# Apply the mapping to the collision type
df_filtered['Collision Category'] = df_filtered['Collision Type'].map(collision_group_map)

# Check for the unique weather conditions after filtering
weather_conditions = df_filtered['Weather'].unique()

# Define colors for Driver At Fault and Injury Severity categories
driver_colors = {
    'Yes': '#1f77b4',  # Blue
    'No': '#ff7f0e',  # Orange
    'Unknown': '#2ca02c'  # Green
}

injury_colors = {
    'Fatal Injury': '#d62728',  # Red
    'No Injury': '#9467bd',  # Purple
    'Injured': '#8c564b',  # Brown
    'Other': '#e377c2'  # Pink
}

# Define different markers for collision categories
collision_category_markers = {
    'Rear-End Collisions': 'o',  # Circle
    'Head-On Collisions': '^',  # Triangle
    'Side-Swipe Collisions': 's',  # Square
    'Other / Miscellaneous': 'D'  # Diamond
}

# Define different colors for each collision category
collision_category_colors = {
    'Rear-End Collisions': '#1f77b4',  # Blue
    'Head-On Collisions': '#d62728',  # Red
    'Side-Swipe Collisions': '#2ca02c',  # Green
    'Other / Miscellaneous': '#9467bd'  # Purple
}

# Define the number of rows and columns for subplots
fig, axes = plt.subplots(1, len(weather_conditions),
                         figsize=(5 * len(weather_conditions), 5))  # Adjust width to fit plots

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Define background colors for each weather condition
weather_background_colors = {
    'Clear': '#87CEEB',  # Sky Blue for Clear
    'Cloudy': '#D3D3D3',  # Light Gray for Cloudy
    'Raining': '#B0C4DE',  # Light Steel Blue for Raining
    'Snow': '#F0F8FF'  # Alice Blue for Snow
}

# Loop through each weather condition and plot
for i, weather in enumerate(weather_conditions):
    # Filter data for the current weather condition
    weather_data = df_filtered[df_filtered['Weather'] == weather]

    # Group by year and count the number of accidents for each year (this is the basis for the graph)
    weather_counts = weather_data.groupby('Year').size().reset_index(name='Accident Count')

    # Create a subplot for the current weather condition
    ax1 = axes[i]

    # Set the background color for the current weather condition
    ax1.set_facecolor(weather_background_colors.get(weather, '#FFFFFF'))  # Default to white if no match

    # Plot Accident Count as a stacked area chart for Driver At Fault
    driver_at_fault_data = weather_data.groupby(['Year', 'Driver At Fault']).size().unstack(fill_value=0)
    driver_at_fault_data.plot(kind='area', stacked=True, alpha=0.6, ax=ax1,
                              color=[driver_colors.get(val, 'lightgray') for val in driver_at_fault_data.columns],
                              legend=False)

    # Plot Injury Severity as line charts with markers
    injury_severity_data = weather_data.groupby(['Year', 'Injury Severity']).size().unstack(fill_value=0)
    for j, col in enumerate(injury_severity_data.columns):
        ax1.plot(injury_severity_data.index, injury_severity_data[col], label=col,
                 marker='o', linestyle='-', linewidth=2, markersize=6, color=injury_colors.get(col, 'gray'))

    # **Plot a single marker for each Collision Category per year**
    # Group by year and collision category (total incidents for each collision category per year)
    collision_category_data = weather_data.groupby(['Year', 'Collision Category']).size().reset_index(
        name='Collision Count')

    # Plot a marker for each collision category, the Y position corresponds to the total incidents
    for collision_category in collision_category_data['Collision Category'].unique():
        collision_subset = collision_category_data[collision_category_data['Collision Category'] == collision_category]

        # Determine the marker and color for this collision category
        marker = collision_category_markers.get(collision_category, 'D')  # Default to 'D' if no match
        color = collision_category_colors.get(collision_category, '#FF6347')  # Default to Tomato color if no match

        # Plot the markers without jitter for continuity
        ax1.scatter(collision_subset['Year'], collision_subset['Collision Count'],
                    label=collision_category if collision_subset['Year'].iloc[0] == collision_subset['Year'].iloc[
                        0] else '',
                    marker=marker,  # Use the appropriate marker for this collision category
                    color=color,  # Use the appropriate color for this collision category
                    s=100, alpha=0.7)

    # Set title and labels for each subplot
    ax1.set_title(f"Weather: {weather}", fontsize=12)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Accident Count')

    # Make grid lines more subtle
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.4)  # Subtle grid

# Create a central legend for both Injury Severity and Driver At Fault categories
handles, labels = [], []

# Driver At Fault legend
handles.append(plt.Line2D([0], [0], color=driver_colors['Yes'], lw=6))
handles.append(plt.Line2D([0], [0], color=driver_colors['No'], lw=6))
handles.append(plt.Line2D([0], [0], color=driver_colors['Unknown'], lw=6))  # Include "Unknown"
labels.extend(['Driver At Fault: Yes', 'Driver At Fault: No', 'Driver At Fault: Unknown'])

# Injury Severity legend with circle markers in the middle of the line
for severity, color in injury_colors.items():
    line_with_marker = plt.Line2D([0], [0], color=color, lw=2, marker='o', markersize=8, linestyle='-',
                                  markerfacecolor=color)
    handles.append(line_with_marker)
    labels.append(severity)

# Collision Category legend with markers
for category, marker in collision_category_markers.items():
    color = collision_category_colors.get(category, '#FF6347')  # Get the appropriate color for the category
    handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=10, lw=0))
    labels.append(category)

# Add the legend outside the plot area
fig.legend(handles, labels, title="Legend", loc="center", bbox_to_anchor=(0.5, 0.05), ncol=5)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
