"""One explicit task override, otherwise the recorded episode/dataset task."""

def resolve_task(task, episode, dataset, label='input'):
    if task is not None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f'{label}: --task must be a nonempty string')
        return task
    for metadata in (episode, dataset):
        candidate = metadata.get('task')
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    raise ValueError(f'{label}: missing recorded task; supply --task')
