import os


def rename_files(directory, extension, name_length=5, start_num=0, logger=None):
    if start_num < 0:
        raise Exception("argument 'start_num' in function rename_files() cannot be less than 0")

    directory = os.path.abspath(directory) + "\\"
    file_names = os.listdir(directory)
    if not extension.startswith('.'):
        extension = '.' + extension
    extension = extension.lower()

    index = start_num
    exists = set()

    while len(file_names) > 0:
        file_name = file_names.pop().lower()
        if file_name.endswith(extension):
            file_name = file_name.replace(extension, '')
            try:
                if len(file_name) == name_length and int(file_name) < len(file_names) + start_num:
                    exists.add(int(file_name))
                    continue
            except ValueError:
                pass

            success = False
            while True:
                new_name = '0' * (name_length - len(str(index))) + str(index)
                try:
                    os.rename(directory + file_name + extension, directory + new_name + extension)
                    if logger is not None:
                        logger.info("Rename: " + file_name + extension + " >>>> " + new_name + extension)
                    exists.add(index)
                    while index in exists:
                        index += 1
                    success = True
                except FileExistsError:
                    exists.add(index)
                    while index in exists:
                        index += 1
                except Exception as e:
                    print(e)
                    success = True
                if success:
                    break


def add_prefix_to_filename(directory, prefix, extension=None):
    directory = os.path.abspath(directory) + "\\"
    file_names = os.listdir(directory)

    if extension is not None:
        if not extension.startswith('.'):
            extension = '.' + extension
        extension = extension.lower()

    for file_name in file_names:
        file_name = file_name.lower()
        if extension is not None:
            if not file_name.endswith(extension):
                continue
        try:
            new_name = prefix + "-" + file_name
            os.rename(directory + file_name, directory + new_name)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    dir_name = "E:\\Human Activity Recognition\\images\\Standing\\New"
    ext = ".jpg"
    rename_files(dir_name, ext, start_num=1877)
