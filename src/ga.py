"""Модуль для гілки `ga`: генетичний алгоритм та Random Search для підбору гіперпараметрів."""

def get_default_hyperparams(config: dict) -> dict:
    """Повертає базові параметри системи для вимірювання Q_baseline (Чекпоінт 1)."""
    return {
        "block_size": int(config.get("candidate_window_frames", 64)),
        "compression_threshold": 0.98,
        "priority_weight": 1.0,
    }


def optimize_with_ga(fitness_fn, config: dict, budget: int = 200) -> tuple[dict, float]:
    """Підбирає параметри за допомогою генетичного алгоритму з елітизмом (Чекпоінт 3)."""
    raise NotImplementedError("TODO: реалізувати у гілці ga")


def optimize_with_random_search(
    fitness_fn, config: dict, budget: int = 200
) -> tuple[dict, float]:
    """Базовий пошук випадковими параметрами при тому самому бюджеті обчислень (Чекпоінт 3)."""
    raise NotImplementedError("TODO: реалізувати у гілці ga")
