import re

match_of = re.compile(":[0-9a-zA-Z]*-of")
match_no_of = re.compile(":[0-9a-zA-Z]*(?!-of)")

placeholder = '<`_placeholder_`>'


def readFile(filepath, keeplabel):
    with open(filepath, 'r') as content_file:
        content = content_file.read()
    amr_t = content
    assert placeholder not in amr_t, 'conflicting placeholder'
    amr_t = amr_t.replace(keeplabel + ' ', placeholder)
    amr_t = re.sub(match_no_of, ":label", amr_t)
    amr_t = re.sub(match_of, ":label-of", amr_t)
    amr_t = amr_t.replace(placeholder, keeplabel + ' ')
    print(amr_t)


import sys

readFile(sys.argv[1], sys.argv[2])
