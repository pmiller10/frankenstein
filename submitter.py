import os

def submit(text):
    """Pass in the text for the submission.
    This will save it in a file in /submissions directory.
    The file name will be submission{x}.csv with x automatically incrementing.
    """
    # TODO verify that the file doesn't already exist
    submission_file_count = len(os.listdir('./submissions'))
    file_name = 'submission{}.csv'.format(submission_file_count)
    path = './submissions/{}'.format(file_name)
    with open(path, 'w') as f:
        f.write(text)
