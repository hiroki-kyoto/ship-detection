# -*- coding:utf8 -*-
import os

class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix=None):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    if special_file.endswith(self.postfix):
                        files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    if special_file.startswith(self.prefix):
                        files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        _is_self = True
        for dirpath, dirnames, files in os.walk(self.directory):
            if not _is_self:
                subdir_list.append(dirpath)
            else:
                _is_self = False
        return subdir_list

''' usage:
if __name__ == "__main__":
    dir = r"/home/hiroki/ships/raw_data/"
    scan = ScanFile(dir, postfix=".*")
    subdirs = scan.scan_subdir()
    files = scan.scan_files()

    print u"扫描的子目录是:"
    for subdir in subdirs:
        print subdir

    print u"扫描的文件是:"
    for file in files:
        print file

    print '--------------------------'
    print files
    print '--------------------------'
'''
