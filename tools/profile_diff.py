# This program takes two directories that are the output of
# legion_prof.py (or its Rust reimplementation) and compares them against
# each other, highlighting the differences
import os
from glob import glob
import os.path
from tabulate import tabulate
from difflib import ndiff
from itertools import zip_longest
import csv
from argparse import ArgumentParser

class Profile:
    def __init__(self, path):
        self.path = path
        self.files = set(map(lambda path: os.path.relpath(path, self.path),
            glob(f"{path}/**", recursive=True)))

    def dump_diff(self, other):
        # First identify which files both profiles have
        fileset_diff = sorted(self.diff_fileset(other))
        # Replace True and False with emoji to make the diff easier to read
        diff_with_emoji = map(lambda triple: (triple[0], self.bool2emoji(triple[1]), self.bool2emoji(triple[2])),
                fileset_diff)
        # Tabulate and print the result
        print(tabulate(diff_with_emoji,
            headers=['Filename', f"Exists in {self.path}", f"Exists in {other.path}"]))

        # Now, find the diffs between the tsv files
        for filename, in_self, in_other in fileset_diff:
            # get the extension of the file
            (_, extension) = os.path.splitext(filename)
            # If the file is present in both self and other, and is a tsv...
            if in_self and in_other and extension == '.tsv':
                # ...and it has a non-None diff, then print it
                diff = list(self.diff_tsv(other, filename))
                if len(diff) > 0:
                    print(f"{filename}:")
                    print(tabulate(diff,
                        headers=[f"Row in {self.path}",
                            f"Row in {other.path}",
                            'Field',
                            f"Value in {self.path}",
                            f"Value in {other.path}"],
                        disable_numparse=True))

    def diff_tsv(self, other, filename):
        # We will store our variables in these lists
        with open(os.path.join(self.path, filename)) as self_file:
            self_header, *self_data = csv.reader(self_file, delimiter='\t')

        with open(os.path.join(other.path, filename)) as other_file:
            other_header, *other_data = csv.reader(other_file, delimiter='\t')

        # first, make sure the headers align
        if self_header != other_header:
            return ["Headers between files don't align".split(' ')]

        result = []
        for srow, orow in zip_longest(sorted(self_data), sorted(other_data),fillvalue=['Row missing' for _ in range(len(self_header))]):
            for col_idx, (sval, oval) in enumerate(zip(srow, orow)):
                srow_idx = self_data.index(srow) if srow in self_data else None
                orow_idx = other_data.index(orow) if orow in other_data else None
                field_name = self_header[col_idx]
                if not self.fields_equal(field_name, sval, oval):
                    result.append([srow_idx, orow_idx, self_header[col_idx], sval, oval])
        return result

    def diff_fileset(self, other):
        all_files = self.files.union(other.files)
        # Return an iterator of filenames, whether the filename is in self,
        # and whether the filename is in other
        return map(lambda filename: (filename, filename in self.files, filename in other.files),
                all_files)

    @staticmethod
    def fields_equal(field_name, left, right):
        if field_name == 'color':
            # We don't care about color comparisons
            return True
        elif field_name == 'time':
            return abs(float(right) - float(left)) < 0.01
        else:
            return left == right

    @staticmethod
    def diff_strings(left, right):
        result = ''
        for sign, _, ch, in ndiff(left, right):
            if sign == '-':
                result += '<' + ch + '>'
            elif sign == '+':
                result += '(' + ch + ')'
            else:
                result += ch
        return result

    @staticmethod
    def bool2emoji(b):
        return '✅' if b else '❌'

def warn(message):
    print(f"WARNING: {message}");

def main(args):
    if not os.path.exists(args.left):
        warn(f"{args.left} does not exist")
        return
    elif not os.path.isdir(args.left):
        warn(f"{args.left} is not a directory")
        return
    if not os.path.exists(args.right):
        warn(f"{args.right} does not exist")
        return
    elif not os.path.isdir(args.right):
        warn(f"{args.right} is not a directory")
        return

    lprofile, rprofile = Profile(args.left), Profile(args.right)
    lprofile.dump_diff(rprofile)

if __name__ == '__main__':
    # If we're running as main, parse in some arguments, and them
    # pass them to our main method
    parser = ArgumentParser(description='Show the diff between two profiler outputs')
    parser.add_argument('left', type=str, help='The first, or left profiler output')
    parser.add_argument('right', type=str, help='The second, or right profiler output')
    args = parser.parse_args()
    main(args)
