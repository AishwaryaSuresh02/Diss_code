import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_PED_GREEN = 0  # action 0 code 00
PHASE_PED_YELLOW = 1
PHASE_VEH_GREEN = 2  # action 1 code 01
PHASE_VEH_YELLOW = 3


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration_veh,green_duration_ped, yellow_duration,
                 num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration_veh = green_duration_veh
        self._green_duration_ped = green_duration_ped
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._cumulative_wait_store = []
        self._queue_length_episode = []

    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_tripfile(seed=str(episode))
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._ped_waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times() + self._collect_ped_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            if (action == 0):
                self._simulate(self._green_duration_veh)
            elif (action == 1):
                self._simulate(self._green_duration_ped)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        # saving only the meaningful reward to better see if the agent is behaving correctly
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time

    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in Sumo
        """
        if (
                self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)
              # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        # incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        incoming_roads = ["EC", "WC"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            vehicle_speed = traci.vehicle.getSpeed(car_id)
            is_stopped = vehicle_speed < 0.01
            if road_id in incoming_roads and is_stopped:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = len(self._waiting_times.values())
        return total_waiting_time

    def _collect_ped_waiting_times(self):
        """
        Retrieve the waiting time of every pedestrian in the walking areas
        """
        WALKINGAREAS = [':C_w0', ':C_w1']
        ped_list = traci.person.getIDList()
        for ped_id in ped_list:
            ped_wait_time = traci.person.getWaitingTime(ped_id)
            road_id = traci.person.getRoadID(ped_id)  # get the road id where the ped is located
            ped_speed = traci.person.getSpeed(ped_id)
            is_stopped = ped_speed < 0.01
            if road_id in WALKINGAREAS and is_stopped:  # consider only the waiting times of peds in WALKINGAREAS
                self._ped_waiting_times[ped_id] = ped_wait_time
            else:
                if ped_id in self._ped_waiting_times:  # a car that was tracked has cleared the intersection
                    del self._ped_waiting_times[ped_id]
        total_ped_waiting_time = len(self._ped_waiting_times.values())
        return total_ped_waiting_time

    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        e wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        prediction=self._Model.predict_one(state)
        print(prediction)
        return np.argmax(prediction)  # the best action given the current state

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("C", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("C", PHASE_VEH_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("C", PHASE_PED_GREEN)
        # elif action_number == 2:
        #    traci.trafficlight.setPhase("C", PHASE_EW_GREEN)
        # elif action_number == 3:
        #    traci.trafficlight.setPhase("C", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_EC = traci.edge.getLastStepHaltingNumber("EC")
        halt_WC = traci.edge.getLastStepHaltingNumber("WC")
        ped_1=len(traci.edge.getLastStepPersonIDs(":C_w1"))
        ped_2=len(traci.edge.getLastStepPersonIDs(":C_w0"))
        # halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        # halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_EC + halt_WC+ped_1+ped_2  # + halt_E + halt_W
        print("que_len",queue_length)
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        WALKINGAREAS = [':C_w0', ':C_w1']
        CROSSINGS = [':C_c0']
        state = np.zeros(self._num_states)
        halt_EC = traci.edge.getLastStepHaltingNumber("EC")
        halt_WC = traci.edge.getLastStepHaltingNumber("WC")
        # halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        # halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_EC + halt_WC
        if queue_length < 1:
            lane_cell = 0
        elif queue_length < 2:
            lane_cell = 1
        elif queue_length < 3:
            lane_cell = 2
        elif queue_length < 4:
            lane_cell = 3
        elif queue_length < 5:
            lane_cell = 4
        elif queue_length < 6:
            lane_cell = 5
        elif queue_length < 7:
            lane_cell = 6
        elif queue_length < 8:
            lane_cell = 7
        elif queue_length < 9:
            lane_cell = 8
        else:
            lane_cell = 9

        numWaiting = 0
        for edge in WALKINGAREAS:
            peds = traci.edge.getLastStepPersonIDs(edge)
            for ped in peds:
                if (traci.person.getWaitingTime(ped) > 0 and
                        traci.person.getNextEdge(ped) in CROSSINGS):
                    numWaiting = traci.trafficlight.getServedPersonCount("C", PHASE_PED_GREEN)

        if numWaiting < 1:
            lane_group = 0
        elif numWaiting < 2:
            lane_group = 1
        elif numWaiting < 3:
            lane_group = 2
        elif numWaiting < 4:
            lane_group = 3
        elif numWaiting < 5:
            lane_group = 4
        elif numWaiting < 6:
            lane_group = 5
        elif numWaiting < 7:
            lane_group = 6
        elif numWaiting < 8:
            lane_group = 7
        elif numWaiting < 9:
            lane_group = 8
        else:
            lane_group = 9

        valid_car = True
        if lane_group >= 1:
            car_position = int(str(lane_group) + str(
                lane_cell))  # composition of the two postion ID to create a number in interval 0-79

        elif lane_group == 0:
            car_position = lane_cell

        else:
            valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

        if valid_car:
            state[
                car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

    @property
    def reward_episode(self):
        return self._reward_episode

    @property
    def queue_length_episode(self):
        return  self._queue_length_episode

