"""Admission units: local correction success, handback and whole-task outcome."""
import numpy as np


def update_failure_count(previous, raw):
    segments = raw.get('segments')
    if segments is not None:
        for segment in segments:
            if type(segment['passed']) is not bool:
                raise ValueError('Explicit local expert outcome required')
            if segment['passed']:
                previous = 0
            elif not str(segment.get('failure', '')).startswith(('ResourcePressure:', 'EpisodeBudgetReached:')):
                previous += 1
        return previous
    if raw['status'] == 'success':
        return 0
    if raw['status'] == 'expert_failed':
        return previous+1
    return previous


def actual_handbacks(episode):
    return sum(segment['end'] < len(episode) and episode.authority[segment['end']] == 0
               for segment in episode.valid_segments)


def pilot_eligible(episode, raw):
    # A later failed expert span remains masked and explicitly reported. It
    # does not invalidate already verified corrections and real model returns.
    handbacks = int(actual_handbacks(episode))
    if handbacks != raw.get('handback_count', 0):
        raise ValueError('Claimed handback count differs from real archived controls')
    return bool(len(episode.valid_segments) >= 2 and handbacks >= 2
                and episode.supervised_count > 0 and raw['all_observations_safe'] is True)
