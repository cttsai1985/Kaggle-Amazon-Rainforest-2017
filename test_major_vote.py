import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from MajorVote import majority_vote, read_submission_paths, run


def submission(tags, ids=None):
    return pd.DataFrame({'image_name': ['image_0'] if ids is None else ids, 'tags': tags})


class MajorityVoteTests(unittest.TestCase):
    def setUp(self):
        self.template = pd.DataFrame({'image_name': ['image_0']})

    def test_whole_tags_do_not_count_substrings(self):
        result = majority_vote(self.template, [submission(['partly_cloudy']),
                                               submission(['partly_cloudy']), submission(['cloudy'])])
        self.assertEqual(result.tags.tolist(), ['partly_cloudy'])

    def test_each_submission_votes_once_and_ties_do_not_pass(self):
        result = majority_vote(self.template, [submission(['water water water']), submission(['clear'])])
        self.assertEqual(result.tags.tolist(), [''])
        result = majority_vote(self.template, [submission(['water water']), submission(['water clear']),
                                               submission(['clear water'])])
        self.assertEqual(result.tags.tolist(), ['clear water'])

    def test_alignment_preserves_sample_order_and_sorted_tags(self):
        template = pd.DataFrame({'image_name': ['b', 'a']})
        votes = [submission(['water clear', 'haze'], ['a', 'b']),
                 submission(['haze', 'clear water'], ['b', 'a']),
                 submission(['haze', 'water clear'], ['b', 'a'])]
        result = majority_vote(template, votes)
        self.assertEqual(result.image_name.tolist(), ['b', 'a'])
        self.assertEqual(result.tags.tolist(), ['haze', 'clear water'])

    def test_bad_id_populations_and_missing_tags_are_rejected(self):
        bad_frames = [submission(['clear', 'water'], ['image_0', 'image_0']),
                      submission(['clear'], ['other']), submission(['clear', 'water'], ['image_0', 'extra']),
                      pd.DataFrame({'image_name': ['image_0']})]
        for frame in bad_frames:
            with self.subTest(columns=frame.columns.tolist(), rows=len(frame)):
                with self.assertRaises(ValueError):
                    majority_vote(self.template, [frame])
        with self.assertRaises(ValueError):
            majority_vote(pd.DataFrame({'image_name': ['', 'image_0']}), [submission(['clear'])])
        with self.assertRaises(ValueError):
            majority_vote(self.template, [])

    def test_empty_tags_are_abstentions_and_zero_rows_report_zero(self):
        result = majority_vote(self.template, [submission(['']), submission([None]), submission(['clear'])])
        self.assertEqual(result.tags.tolist(), [''])
        empty = pd.DataFrame({'image_name': pd.Series(dtype=str)})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = majority_vote(empty, [submission([], [])], progress=True)
        self.assertTrue(result.empty)
        self.assertEqual(output.getvalue(), 'processed 00000 samples\n')

    def test_trimmed_filenames_real_csv_alignment_and_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voter = root / 'vote.csv'
            sample = root / 'sample.csv'
            collection = root / 'SubmVoters.csv'
            output = root / 'result.csv'
            submission(['', 'clear'], ['b', 'a']).to_csv(voter, index=False)
            pd.DataFrame({'image_name': ['a', 'b'], 'tags': ['ignored', 'ignored']}).to_csv(sample, index=False)
            pd.DataFrame({'submission': ['  ' + str(voter) + '  ']}).to_csv(collection, index=False)
            with contextlib.redirect_stdout(io.StringIO()) as messages:
                run(collection, sample, output)
            self.assertIn('processed 00002 samples', messages.getvalue())
            result = pd.read_csv(output, keep_default_na=False)
            self.assertEqual(result.image_name.tolist(), ['a', 'b'])
            self.assertEqual(result.tags.tolist(), ['clear', ''])
            original = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                run(collection, sample, output)
            self.assertEqual(output.read_bytes(), original)

    def test_duplicate_files_and_empty_voter_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'SubmVoters.csv'
            pd.DataFrame({'submission': ['one.csv', './one.csv']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'same file'):
                read_submission_paths(path)
            pd.DataFrame({'submission': pd.Series(dtype=str)}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'at least one'):
                read_submission_paths(path)


if __name__ == '__main__':
    unittest.main()
