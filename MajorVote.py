import time
from collections import Counter
from math import floor, isnan
from pathlib import Path

import pandas as pd


target_id = 'image_name'


def validate_ids(frame, label):
    if target_id not in frame.columns:
        raise ValueError('{} must contain {}'.format(label, target_id))
    ids = frame[target_id]
    if not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('{} has a missing or empty image ID'.format(label))
    if ids.duplicated().any():
        raise ValueError('{} has duplicate image IDs'.format(label))


def tag_set(value):
    # Empty CSV cells mean that this submission votes for no tags on that image.
    if value is None or (isinstance(value, float) and isnan(value)):
        return set()
    if not isinstance(value, str):
        raise ValueError('Tags must be text or an empty cell')
    return set(value.split())


def majority_vote(template, submissions, progress=False):
    """Return exact-token majority votes in sample-submission order."""
    submissions = list(submissions)
    if not submissions:
        raise ValueError('At least one submission is required')
    validate_ids(template, 'Sample submission')
    expected_ids = template[target_id].tolist()
    expected_set = set(expected_ids)
    aligned = []
    for number, submission in enumerate(submissions, start=1):
        label = 'Submission {}'.format(number)
        validate_ids(submission, label)
        if 'tags' not in submission.columns:
            raise ValueError('{} must contain tags'.format(label))
        actual_ids = set(submission[target_id])
        if actual_ids != expected_set:
            raise ValueError(
                '{} image IDs differ from sample: {} missing, {} extra'.format(
                    label, len(expected_set - actual_ids), len(actual_ids - expected_set)
                )
            )
        aligned.append(submission.set_index(target_id).loc[expected_ids, 'tags'].tolist())

    cutoff = floor(len(submissions) / 2) + 1
    all_major_tags = []
    for number in range(len(expected_ids)):
        counts = Counter()
        for tags in aligned:
            # A tag gets at most one vote from each submission.
            counts.update(tag_set(tags[number]))
        all_major_tags.append(' '.join(sorted(tag for tag, votes in counts.items() if votes >= cutoff)))
        if progress and (number + 1) % 1000 == 0:
            print('processed {:05d} samples'.format(number + 1), flush=True)
    if progress and (not expected_ids or len(expected_ids) % 1000 != 0):
        print('processed {:05d} samples'.format(len(expected_ids)), flush=True)
    return pd.DataFrame({target_id: expected_ids, 'tags': all_major_tags})


def read_submission_paths(voters_file):
    collection = pd.read_csv(voters_file, skipinitialspace=True, dtype=str, keep_default_na=False)
    if 'submission' not in collection.columns:
        raise ValueError('SubmVoters.csv must contain a submission column')
    if collection.empty:
        raise ValueError('SubmVoters.csv must list at least one submission')
    paths = []
    seen = set()
    for name in collection['submission']:
        name = name.strip()
        if not name:
            raise ValueError('SubmVoters.csv contains an empty filename')
        path = Path(name)
        identity = path.resolve()
        if identity in seen:
            raise ValueError('SubmVoters.csv references the same file more than once')
        seen.add(identity)
        paths.append(path)
    return paths


def run(voters_file='SubmVoters.csv', sample_file='../input/sample_submission_v2.csv', output_file=None):
    paths = read_submission_paths(voters_file)
    template = pd.read_csv(sample_file, dtype=str, keep_default_na=False)
    submissions = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    print('read in {} submissions'.format(len(submissions)), flush=True)
    cutoff = floor(len(submissions) / 2) + 1
    print('using majority vote (>={:d}) to make ensemble predictions'.format(cutoff))
    result = majority_vote(template, submissions, progress=True)
    if output_file is None:
        timestamp = time.strftime('%Y-%m-%d-%H-%M')
        output_file = 'sumbEns_major_{}.csv'.format(timestamp)
    # Exclusive creation prevents reruns in the same minute replacing a result.
    result.to_csv(output_file, index=False, mode='x')
    return Path(output_file)


if __name__ == '__main__':
    try:
        run()
    except (ValueError, OSError) as error:
        raise SystemExit('Majority vote failed: {}'.format(error)) from None
