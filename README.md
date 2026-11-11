# Amazon Rainforest Submission Voting

A small majority-vote script for the multi-label submissions from the 2017 [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) competition.

1. Install pandas: `python -m pip install -r requirements.txt`.
2. Put the paths to your prediction CSVs in the `submission` column of `SubmVoters.csv`.
3. Keep the sample submission at `../input/sample_submission_v2.csv` and run `python MajorVote.py` from this directory.

Each prediction CSV needs `image_name` and `tags` columns and exactly the sample's image IDs. Row order may differ. The output follows sample order, with alphabetically sorted tags, and is written as `sumbEns_major_YYYY-MM-DD-HH-MM.csv`. An existing output is never overwritten.

A tag is retained only when it has at least `floor(number_of_submissions / 2) + 1` votes. Votes use whole tags: `cloudy` and `partly_cloudy` are different labels, and a repeated tag within one submission counts once. An empty tag cell casts no tag votes; an image may therefore have an empty output. Duplicate file references and duplicate, missing or extra image IDs are rejected.

The original prediction files are not included, so the checks establish voting behavior rather than a reproduced competition score or an F2 improvement.

Run the regression checks with `python -m unittest -v test_major_vote.py`.
