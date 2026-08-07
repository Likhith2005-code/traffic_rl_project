import traci
import os
import sys

# Check SUMO_HOME
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable.")

# Path to your SUMO configuration file
sumo_config = "simulation/config.sumocfg"

# Command to start SUMO GUI
sumoCmd = [
    "sumo-gui",
    "-c",
    sumo_config,
    "--start",
    "--delay","300"

]

# Start SUMO
traci.start(sumoCmd)

print("SUMO started successfully!")

# Run simulation
while traci.simulation.getMinExpectedNumber() > 0:

    # Move simulation one step
    traci.simulationStep()

    # Get all vehicle IDs
    vehicle_ids = traci.vehicle.getIDList()

    # Display information for each vehicle
    for vehicle in vehicle_ids:

        speed = traci.vehicle.getSpeed(vehicle)
        lane = traci.vehicle.getLaneID(vehicle)
        position = traci.vehicle.getPosition(vehicle)

        print(f"""
Vehicle : {vehicle}
Speed   : {speed:.2f} m/s
Lane    : {lane}
Position: {position}
---------------------------
""")

        # Example: Set maximum speed to 15 m/s
        if speed < 15:
            traci.vehicle.setSpeed(vehicle, 15)

# Close SUMO connection
traci.close()
print("Simulation finished.")