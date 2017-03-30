import numpy as np
import copy
import logging

class QLearning:
    def __init__(self, state_variables, states_partition, failure_states_partitions, action_set, greek, begin_state=None,
                 begin_action=None):
        """
        
        :param state_variables: 
        :param states_partition: (dict-object) 
                    example: 
                        {"state_variable_x": {"1": lambda x: x > 10}} 
                        means: given x > 10, the state_variable_x
        :param failure_states_partitions: 
        :param action_set: 
        :param greek: 
        :param begin_state: 
        :param begin_action: 
        """

        # variables that will be initialized in functions
        self.begin_action = None
        self.begin_state = None
        self.action = None
        self.previous_action = None
        self.box = None
        self.__action_column_map = None
        self.state = None
        self.previous_state = None
        self.bases = None
        self.total_boxes = None

        # state variables
        self.state_variables = state_variables

        # rules to decide failure-states
        self.failure_states_partitions = failure_states_partitions

        # rules to decide states
        self.states_partition = states_partition

        self.ini_box_bases()
        self.ini_state(begin_state)

        # action
        self.actionSet = action_set
        self.ini_action(begin_action)

        self.q_matrix = np.zeros((self.total_boxes, len(action_set)), dtype=float)

        # learning rate
        self.alpha = greek["alpha"]

        # discount factor for future reinforcement
        self.gamma = greek["gamma"]

        # magnitude of noise added to choice
        self.delta = greek["delta"]

        self.box_vector = np.zeros((self.total_boxes, 1), dtype=float)

    def ini_action(self, begin_action):

        if not begin_action:
            self.begin_action = self.actionSet[0]

        self.action = self.begin_action
        self.previous_action = self.action

        self.__action_column_map = dict()
        for i in range(0, len(self.actionSet)):
            self.__action_column_map[self.actionSet[i]] = i

    def ini_state(self, begin_state):

        if not begin_state:
            self.begin_state = dict()
            for state_vector in self.state_variables:
                self.begin_state[state_vector] = 0.0
        else:
            self.begin_state = begin_state

        self.state = self.begin_state
        self.box = self.get_box(self.state)

        self.previous_state = self.state

    def ini_box_bases(self):
        self.bases = dict()
        self.total_boxes = 1
        for sv in self.state_variables:
            self.bases[sv] = self.total_boxes
            self.total_boxes *= len(self.states_partition[sv])

    def get_q_value(self, state, action):

        row_index = self.get_box(state)
        column_index = self.__action_column_map[action]

        return self.q_matrix[row_index, column_index]

    def update_q_value(self, state, action, reward, optimal_future_value=0):
        """
        
        :param state: (state_vector) the (state, action) decide a value
        :param action: ()
        :param reward: 
        :param optimal_future_value: the max q_value of the state_future = trans(state, action)
        :return: 
        """

        # new_q_value
        # = old_q_value + learning_rate * (reward + discount_factor * optimal_future_value - old_q_value)
        # = (1 - learning_rate) * old_q_value + learning_rate * (reward + discount_factor * optimal_future_value)
        # optimal_future_value : the max current q value given current state
        new_q_value = (1 - self.alpha) * self.get_q_value(self.previous_state,
                                                          self.previous_action) + self.alpha * (
            reward + self.gamma * optimal_future_value)

        row_index = self.get_box(state)
        column_index = self.__action_column_map[action]

        self.q_matrix[row_index, column_index] = new_q_value

    def choose_max_q_value(self, state):
        """
         choose action with max Q value give state
        :param state: 
        :return: 
        """
        max_q = -np.inf
        for action in self.actionSet:
            trail = self.get_q_value(state, action)
            if max_q < trail:
                max_q = trail

        return max_q

    def choose_action(self, state, with_noise=False):
        """
        choose the best action given state
        :param state: 
        :param with_noise: 
        :return: 
        """
        max_q = -np.inf
        optimal_action = self.actionSet[0]
        for action in self.actionSet:
            trail = self.get_q_value(state, action) + int(with_noise) * self.delta * np.random.random_sample()
            if max_q < trail:
                max_q = trail
                optimal_action = action

        return optimal_action

    def set_state(self, state):

        self.previous_state = copy.deepcopy(self.state)
        self.previous_action = copy.deepcopy(self.action)

        for sv in self.state_variables:
            self.state[sv] = state[sv]

        # new box-vector
        self.box_vector.fill(0)

        self.box = self.get_box(self.state)
        if not self.box < 0:
            self.box_vector[self.box, 0] = 1

    def calculate_box_value(self, partitions, state):
        """
        
        :param partitions: 
        :param state: 
        :return: 
        """
        _boxValue = 0

        for sv in self.state_variables:
            if sv not in partitions:
                continue
            judges = partitions[sv]
            base = self.bases[sv]

            for box in sorted(judges):
                judge = judges[box]
                if judge(state[sv]):
                    _boxValue += base * int(box)
                    break

        return _boxValue - 1

    def get_box(self, state):
        """
        get the box value (a integer) given a state-vector
        :param state: (vector like object) A vector implies the current observation
        :return: (int) A positive integer encoded the state-vector and -1 implying failure has happened
        """
        if self.calculate_box_value(self.failure_states_partitions, state) < -1:
            return -1
        else:
            state_box = self.calculate_box_value(self.states_partition, state)
            return state_box

    def get_action(self, reward=0):

        self.box = self.get_box(self.state)

        if self.get_box(self.previous_state) > 0:
            if self.box < 0:
                predicted_value = 0
            else:
                predicted_value = self.choose_max_q_value(self.state)

            self.update_q_value(self.previous_state, self.previous_action, reward, predicted_value)
        else:
            # A failure has happened.
            pass

        self.action = self.choose_action(self.state, with_noise=True)

        return self.action

    def failed_update(self, punish):
        """
        
        :param punish: (float) a negative reward when failure happened
        :return: 
        """
        self.update_q_value(self.previous_state, self.previous_action, punish)

    def is_failed(self, punish=-1):
        """
        check whether current state implies failure
        if failed then update reward (reinforcement) 
        :param punish: 
        :return: 
        """

        self.box = self.get_box(self.state)
        # print("current box is {bx}".format(bx=self.box))

        # current state failed
        if self.box < 0:
            self.failed_update(punish)
            return True

        return False
