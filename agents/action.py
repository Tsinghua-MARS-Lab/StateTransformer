from enum import Enum, unique, auto

COLORS = [[255, 204, 13], [255, 155, 38], [255, 25, 77], [191, 38, 105], [112, 42, 140], [136, 247, 226],
          [68, 212, 146], [0, 66, 90], [31, 138, 112], [191, 219, 56], [252, 115, 0], [3, 201, 136], [28, 130, 173],
          [0, 51, 124], [132, 210, 197], [228, 201, 138], [194, 118, 110], [176, 90, 122]]

@unique
class ActionLabel(Enum):
    """
    Assign an Action Label: an_action = Action(label=ActionLabel.Cruising)
    ActionLabel is hashable
    """
    Cruising = auto()  # starting from 1
    Follow = auto()
    Stop = auto()
    PassingIntersection = auto()
    TurningLeft = auto()
    TurningRight = auto()
    UTurn = auto()
    # NudgeLeft = auto()
    # NudgeRight = auto()
    MoveLeft = auto()
    MoveRight = auto()
    PassingRoundabout = auto()
    Merging = auto()
    ExitHighway = auto()

    @property
    def color(self):
        color_index = self.value % len(COLORS)
        return COLORS[color_index]

    def needs_a_goal(self):
        """
        Returns a boolean for compulsory goal point for this action
        """
        if self in [ActionLabel.Follow]:
            return False
        else:
            return True

    def get_index(self):
        return self.value

    def next_action(self, recycle=True):
        if self == ActionLabel.get_last():
            if recycle:
                return ActionLabel.get_first()
            else:
                return None
        else:
            return ActionLabel(self.value+1)

    def previous_action(self, recycle=True):
        if self == ActionLabel.get_first():
            if recycle:
                return ActionLabel.get_last()
            else:
                return None
        else:
            return ActionLabel(self.value-1)

    @staticmethod
    def get_first():
        for action in ActionLabel:
            return action

    @staticmethod
    def get_last():
        last_action = None
        for action in ActionLabel:
            last_action = action
        return last_action

    def __str__(self):
        return f'{self.name}'


class Action:
    def __init__(self, label, start_frame, end_frame=None, goal=None):
        """
        goal should be a list of [x, y, yaw]
        """
        assert type(label) == ActionLabel, f'pass ActionLabel as label to create an Action, not {type(label)}'
        self.label = label
        self.start_frame = start_frame
        self.end_frame = end_frame
        # if self.label.needs_a_goal():
        #     assert goal is not None, f'{label} needs a goal but got {goal}'
        #     assert len(goal) == 3, f'goal should fit the shape of 3, but got {goal}'
        self.goal = goal
        self.editing = False

    def __str__(self):
        return f'{self.label} w {self.goal} from {self.start_frame} to {self.end_frame}\n'
