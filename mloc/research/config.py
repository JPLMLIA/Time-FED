"""
"""
import copy
import logging
import yaml

Logger = logging.getLogger('MilkyLib/config.py')

class Null:
    """
    Null class. Used by Section to access nonexistant configuration keys
    """
    def __call__(self):
        pass

    def __bool__(self):
        return False

    def __getattr__(self, key):
        return self

    def __getitem__(self, key):
        return self

    def __contains__(self):
        return False

    def __iter__(self):
        return iter(())

    def __repr__(self):
        return 'Null'

    def get(self, key, other=None):
        return other

class Section:
    """
    A section of the Config, essentially acts like a dict with dot notation
    """
    def __init__(self, name='', data={}):
        self.name = name
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Section(key, value))
            else:
                setattr(self, key, value)

    def __reduce__(self):
        data = self.__dict__.copy()
        data.pop('name')
        return (self.__class__, (self.name, data))

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        data = self.__dict__.copy()
        data.pop('name')
        return iter(data)

    def items(self):
        data = self.__dict__.copy()
        data.pop('name')
        return data.items()

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        return Null()

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__getattr__(key)

    def get(self, key, other=None):
        return self.__dict__.get(key, other)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __repr__(self):
        return f"<Section '{self.name}'>"

class Config:
    """
    Reads a yaml file and creates a configuration from it
    """
    def __init__(self, file, active=None, default='default'):
        """
        Loads a config.yaml file

        Parameters
        ----------
        file: str
            Path to the config.yaml file to read
        active: str or None
            Sets the active section
        default: str or None
            Defaults to this section when a key is not found in the active
            section
        """
        self.file = file
        with open(file, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        if active not in data:
            raise AttributeError(f'Active section {active!r} does not exist in the YAML: {list(data.keys())}')

        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Section(key, value))
            else:
                setattr(self, key, value)

        self._active  = active
        self._default = default if default in data else None

        # If there is a default section, deep copy it then override it using the active section
        if self._default is not None:
            self.active = copy.deepcopy(self.__dict__[self._default])
            print(self.active)
            self.active.name = f'Active {self._active}'
            self.update(self.__dict__[self._active], self.active)
        # Otherwise set the active section as active
        else:
            self.active = self.__dict__[self._active]

    def __reduce__(self):
        return (self.__class__, (self.file, ))

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __getattr__(self, key):
        # Key is a section key
        if key in self.__dict__:
            return self.__dict__[key]
        # Key is not a section key, return value from active section
        else:
            return self.active[key]

    def old__getattr__(self, key):
        if key not in self.__dict__:
            if self._active is not None:
                value = self.__dict__[self._active][key]
                if not isinstance(value, Null):
                    return value
            if self._default is not None:
                value = self.__dict__[self._default][key]
                return value
        else:
            return self.__dict__[key]
        return Null()

    def __getitem__(self, key):
        return self.__getattr__(key)

    def get(self, key, other=None):
        value = self.__getattr__(key)
        if not isinstance(value, Null):
            return value
        return other

    def __repr__(self):
        sections, attributes = [], []
        for key, value in self.__dict__.items():
            if isinstance(value, Section):
                sections.append(key)
            else:
                attributes.append(key)
        return f'<Config(attributes={attributes}, sections={sections})>'

    def update(self, a, b=None, child=False):
        """
        Updates the `b` section with the arguments of the `a` section

        Parameters
        ----------
        a: milkylib.config.Section
            Section object to be read from
        b: milkylib.config.Section
            Section object to be updated
        child: bool
            Informs the recursive function calls that the sections are children
            of the initial function call to prevent the function from deleting
            the sections
        """
        # Use the default section if not given
        if b is None:
            b = self.active

        # delete = set()
        for key, value in a.items():
            if key in b:
                if isinstance(value, Section):
                    self.update(value, b[key])#, child=True)
                else:
                    # print(f'Setting {b}[{key!r}] = {value!r}')
                    b[key] = value
                # delete.add(key)
            # elif child:
            #     b[key] = value
            b[key] = value

    def reset(self):
        """
        Resets a config back to its initialized state. The initial active state
        becomes active again.
        """
        self.__dict__.update(Config(self.file, active=self._active, default=self._default).__dict__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/config.yaml',
                                            help     = 'Path to a config.yaml file'
    )

    args = parser.parse_args()

    config = utils.Config(args.config)
