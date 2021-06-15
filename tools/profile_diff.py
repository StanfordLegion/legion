# This program takes two directories that are the output of
# legion_prof.py (or its Rust reimplementation) and compares them against
# each other, highlighting the differences
import os
from glob import glob
from fnmatch import fnmatch
import os.path
from tabulate import tabulate
import pandas
from difflib import ndiff
import daff
import sys
from argparse import ArgumentParser

class Profile:
    def __init__(self, path, file_patterns):
        self.path = path
        # Glob all files in the chosen directory tree
        all_files = glob("%s/**" % path, recursive=True)
        # Convert each path to one relative to the base path
        relative_files = map(lambda path: os.path.relpath(path, self.path),all_files)
        # Filter out only paths that match one of the file_patterns
        filtered_files = filter(lambda path: any([fnmatch(path, pattern) for pattern in file_patterns]),
                relative_files)
        # And make this as our set of files
        self.files = set(filtered_files)

    # This method is very similar to dump_diff below but it returns a pair
    # (same, msg) that shows whether there is a diff between the two profiles
    # and gives a pref summary to explain
    def check(self, other, exclude_field_patterns):
        same = True
        msg = []
        fileset_diff = set(self.diff_fileset(other))
        for filename, inself, inother in fileset_diff:
            if not inself:
                msg.append("%s is missing %s" % (self.path, filename))
                same = False
            if not inother:
                msg.append("%s is missing %s" % (other.path, filename))
                same = False

        # Now, find the diffs between the tsv files
        for filename, in_self, in_other in fileset_diff:
            # get the extension of the file
            (_, extension) = os.path.splitext(filename)
            # If the file is present in both self and other, and is a tsv...
            if in_self and in_other and extension == '.tsv':
                # The fields to be excluded are only the fields where the filepattern is none or a
                # file pattern matched the given filename
                exclude_fields = set(map(lambda pair: pair[1],
                        filter(lambda pair: pair[0] is None or fnmatch(filename, pair[0]),
                            exclude_field_patterns)))
                # Calculate the diff
                table_diff = self.diff_tsv(other, filename, exclude_fields)
                # And if the diff isn't None
                if table_diff is not None:
                    msg.append("Diff detected in file %s" % filename)
                    same = False
        return same, msg

    def dump_diff(self, other, exclude_field_patterns):
        # First identify which files both profiles have
        fileset_diff = sorted(self.diff_fileset(other))
        # Replace True and False with emoji to make the diff easier to read
        diff_with_emoji = map(lambda triple: (triple[0], self.bool2emoji(triple[1]), self.bool2emoji(triple[2])),
                fileset_diff)
        # Tabulate and print the result
        print(tabulate(diff_with_emoji,
            headers=['Filename', "Exists in %s" % self.path, "Exists in %s" % other.path]))

        # Now, find the diffs between the tsv files
        for filename, in_self, in_other in fileset_diff:
            # get the extension of the file
            (_, extension) = os.path.splitext(filename)
            # If the file is present in both self and other, and is a tsv...
            if in_self and in_other and extension == '.tsv':
                # The fields to be excluded are only the fields where the filepattern is none or a
                # file pattern matched the given filename
                exclude_fields = set(map(lambda pair: pair[1],
                        filter(lambda pair: pair[0] is None or fnmatch(filename, pair[0]),
                            exclude_field_patterns)))
                # Calculate the diff
                table_diff = self.diff_tsv(other, filename, exclude_fields)
                # And if the diff isn't None
                if table_diff is not None:
                    # Then print the filename and the diff
                    print()
                    print(filename)
                    print('-' * len(filename))
                    print(table_diff)

    def diff_tsv(self, other, filename, exclude_fields):
        # We will store our variables in these lists
        with open(os.path.join(self.path, filename)) as self_file:
            self_dataframe = pandas.read_csv(self_file, delimiter='\t')
            self_dataframe_filtered = self_dataframe.drop(
                    filter(lambda field: field in self_dataframe.columns, exclude_fields),
                    axis = 1)
            self_list = [self_dataframe_filtered.columns.tolist()] + self_dataframe_filtered.values.tolist()
            self_data = daff.PythonTableView(self_list)

        with open(os.path.join(other.path, filename)) as other_file:
            other_dataframe = pandas.read_csv(other_file, delimiter='\t')
            other_dataframe_filtered = other_dataframe.drop(
                    filter(lambda field: field in other_dataframe.columns, exclude_fields),
                    axis = 1)
            other_list = [other_dataframe_filtered.columns.tolist()] + other_dataframe_filtered.values.tolist()
            other_data = daff.PythonTableView(other_list)

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
    print("WARNING: %s" % message);

def main(args):
    # Validate arguments
    if not os.path.exists(args.left):
        warn("%s does not exist" % args.left)
        return
    elif not os.path.isdir(args.left):
        warn("%s is not a directory" % args.left)
        return
    if not os.path.exists(args.right):
        warn("%s does not exist" % args.right)
        return
    elif not os.path.isdir(args.right):
        warn("%s is not a directory" % args.right)
        return

    # For each exclude_field arg, split at the ':' if there is one, or have the first
    # element of the pair be None
    exclude_field_patterns = list(map(lambda arg: tuple(arg.split(':', 1)) if arg.find(':') != -1 else (None, arg),
            args.exclude_field))

    file_patterns = args.file_pattern if len(args.file_pattern) > 0 else ['*']
    lprofile = Profile(args.left, file_patterns=file_patterns)
    rprofile = Profile(args.right, file_patterns=file_patterns)
    if args.check:
        same, msg = lprofile.check(rprofile, exclude_field_patterns=exclude_field_patterns)
        for line in msg:
            print(line)
        if same:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        lprofile.dump_diff(rprofile, exclude_field_patterns=exclude_field_patterns)

if __name__ == '__main__':
    # If we're running as main, parse in some arguments, and them
    # pass them to our main method
    parser = ArgumentParser(description='Show the diff between two profiler outputs')
    parser.add_argument('left', type=str, help='The first, or left profiler output')
    parser.add_argument('right', type=str, help='The second, or right profiler output')
    parser.add_argument('--file-pattern', type=str, default=[], action='append', help='''
    A pattern of files to match. Can be passed multiple times. If not passed, defaults to *.''')
    parser.add_argument('--check', action='store_true', help='''
    Run in 'check mode'. In this mode, instead of showing a diff, the tool will
    only print a brief summary of detected differences. It will exit(0) if no diff is detected
    and exit nonzero if a difference is detected.''')
    parser.add_argument('--exclude-field', type=str, default=[], action='append', help='''
    A pattern to exclude from comparison. Each pattern should have the format
    "<optional glob-style filepath>:[field name]". So two valid arguments would "count" and
    "tsv/Proc_0x*:prof_uid". This argument can be applied multiple times to exclude different
    patterns''')
    args = parser.parse_args()
    main(args)
