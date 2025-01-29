import os
import pathlib
from pathlib import Path


class ProjectPaths:
    def __init__(self, basepath=None, parent=None, level=0, create_on_add=True):
        if basepath is not None:
            basepath = Path(basepath)
        self.basepath = basepath
        # self.parent = parent
        self.level = level
        self.create_on_add = create_on_add
        if self.create_on_add:
            self.make_all()

    def add_container(self, name, basepath):
        setattr(
            self,
            name,
            self.__class__(
                basepath=basepath,
                parent=self,
                level=self.level + 1,
                create_on_add=self.create_on_add,
            ),
        )
        if self.create_on_add:
            self.make_all()
        return getattr(self, name)

    def add_subcontainer(self, name, *child_path_elements):
        if child_path_elements:
            new_path = self.basepath.joinpath(*child_path_elements)
        else:
            new_path = self.basepath.joinpath(name)

        setattr(
            self,
            name,
            self.__class__(
                basepath=new_path,
                parent=self,
                level=self.level + 1,
                create_on_add=self.create_on_add,
            ),
        )
        if self.create_on_add:
            self.make_all()
        return getattr(self, name)

    def add_path(self, name, full_fpath):
        setattr(self, name, Path(full_fpath))
        return getattr(self, name)

    def add_subpath(self, name, *child_path_elements):
        if child_path_elements:
            new_path = self.basepath.joinpath(*child_path_elements)
        else:
            new_path = self.basepath.joinpath(name)
        setattr(self, name, new_path)
        return str(getattr(self, name))

    def joinpath(self, *path_extensions):
        return self.basepath.joinpath(*path_extensions)

    def glob(self, *args):
        return self.basepath.glob(*args)

    def _generate_tree_string(self):
        output = ""
        if self.level == 0:
            output = f"{self.__class__}\n"
            output = f"{self.basepath}\n"

        for attr_name, attr in vars(self).items():
            if attr_name == "basepath":
                continue

            if isinstance(attr, self.__class__):
                indent_string = "--" * attr.level
                output += f"{indent_string} {attr_name}\n"
                output += attr._generate_tree_string()
                # break
            elif isinstance(attr, pathlib.PosixPath):
                indent_string = "--" * (self.level + 1)
                output += f"{indent_string} {attr_name} *\n"

        return output

    def __str__(self):
        return str(self.basepath)

    def view(self):
        print(self._generate_tree_string())

    def ls(self):
        return os.listdir(self.basepath)

    def make_all(self):
        """
        Make each of these paths if they don't already exist
        """
        if self.basepath:
            # print(f'Making {self.basepath}...')
            self.basepath.mkdir(exist_ok=True)
        for attr_name, attr in vars(self).items():
            if isinstance(attr, self.__class__):
                # print(f'Calling make_all() on {attr}')
                attr.make_all()
        # print(f'Done making subpaths for {self.basepath}.')
