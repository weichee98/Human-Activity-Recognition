import os


class FilePath:

    def __init__(self, path):
        self.__abspath = os.path.abspath(path)
        self.__directory, self.__file = os.path.split(self.__abspath)
        if len(self.__file.split('.')) == 0:
            self.__filename = self.__file
            self.__extension = None
        else:
            self.__filename = '.'.join(self.__file.split('.')[:-1])
            self.__extension = '.' + self.__file.split('.')[-1]

    def getAbsPath(self):
        return self.__abspath

    def getDirectory(self):
        return self.__directory

    def getFile(self):
        return self.__file

    def getFileName(self):
        return self.__filename

    def getExtension(self):
        return self.__extension