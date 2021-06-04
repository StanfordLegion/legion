# This program takes two directories that are the output of
# legion_prof.py (or its Rust reimplementation) and compares them against
# each other, highlighting the differences
import os
from glob import glob
from fnmatch import fnmatch
import os.path
from tabulate import tabulate
from difflib import ndiff
import daff
import csv
from argparse import ArgumentParser

class Profile:
    def __init__(self, path):
        self.path = path
        self.files = set(map(lambda path: os.path.relpath(path, self.path),
            glob(f"{path}/**", recursive=True)))

    def dump_diff(self, other, tsv_file_pattern):
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
            if in_self and in_other and extension == '.tsv' and fnmatch(filename, f"tsv/{tsv_file_pattern}"):
                # Calculate the diff
                table_diff = self.diff_tsv(other, filename)
                # And if the diff isn't None
                if table_diff is not None:
                    # Then print the filename and the diff
                    print()
                    print(filename)
                    print('-' * len(filename))
                    print(table_diff)

    def diff_tsv(self, other, filename):
        # We will store our variables in these lists
        with open(os.path.join(self.path, filename)) as self_file:
            self_data = daff.PythonTableView(list(csv.reader(self_file, delimiter='\t')))

        with open(os.path.join(other.path, filename)) as other_file:
            other_data = daff.PythonTableView(list(csv.reader(other_file, delimiter='\t')))

        alignment  = daff.Coopy.compareTables(self_data, other_data).align()
        result     = daff.PythonTableView([])

        diff = daff.TableDiff(alignment, daff.CompareFlags())
        diff.hilite(result)

        if diff.hasDifference():
            return daff.TerminalDiffRender().render(result)
        else:
            return None

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
    lprofile.dump_diff(rprofile, args.tsv_file_pattern)

if __name__ == '__main__':
    # If we're running as main, parse in some arguments, and them
    # pass them to our main method
    parser = ArgumentParser(description='Show the diff between two profiler outputs')
    parser.add_argument('left', type=str, help='The first, or left profiler output')
    parser.add_argument('right', type=str, help='The second, or right profiler output')
    parser.add_argument('--tsv-file-pattern', type=str, default='*', help='The pattern to match for tsv files to diff. Defaults to *.')
    args = parser.parse_args()
    main(args)
