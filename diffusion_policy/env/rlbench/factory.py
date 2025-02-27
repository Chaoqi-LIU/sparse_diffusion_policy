from typing import List

MT2_TASKS = [
    'close_box',
    'toilet_seat_down',
]

MT3_TASKS = [
    'close_box',
    'close_microwave',
    'toilet_seat_down',
]

MT4_TASKS = [
    "open_box",
    "toilet_seat_up",
    "open_drawer",
    "take_umbrella_out_of_umbrella_stand",
]

MT5_TASKS = [
    "open_box",
    "open_microwave",
    "toilet_seat_up",
    "open_drawer",
    "take_umbrella_out_of_umbrella_stand",
]

MT_TASKS = {
    'mt2': MT2_TASKS,
    'mt3': MT3_TASKS,
    'mt4': MT4_TASKS,
    'mt5': MT5_TASKS,
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]
    