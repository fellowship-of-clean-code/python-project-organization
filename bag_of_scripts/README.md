## Script structure

```python some_definition.py
```

```python using_the_definition.py
```

This illustrates how an import can be performed between scripts,
and the function of the `if __name__ == '__main__':` clause.

## Shebang

This is a line we can add to the top of a python script, 
and which indicates that this is indeed a script, meant to be run "standalone".

The preferred shebang is:
```#! /usr/bin/env python3```

although something like
```#! /usr/bin/python3```
would work, it's not portable, i.e. it depends on the specific location where python is
installed on the user's system.

If this line is present, instead of running the script like

```python_3 shebang.py
```

we can do 

```./shebang.py
```

(but first, we must make it executable with `chmod +x shebang.py`).

## Python path

```python import_paths.py
```

try: 
```export PYTHONPATH="/some/other/path"
```

Then the full output I get is:
```
The path is:
/home/jacopo/Documents/focc/python-project-organization/bag_of_scripts
/some/other/path
/home/jacopo/.pyenv/versions/3.9.11/lib/python39.zip
/home/jacopo/.pyenv/versions/3.9.11/lib/python3.9
/home/jacopo/.pyenv/versions/3.9.11/lib/python3.9/lib-dynload
/home/jacopo/.local/lib/python3.9/site-packages
/home/jacopo/.pyenv/versions/3.9.11/lib/python3.9/site-packages
__editable__.ligo.skymap-1.0.6.dev1+gf146202.finder.__path_hook__

This finder is <class '_frozen_importlib.BuiltinImporter'>
When looking for "some_definition" it finds None

This finder is <class '_frozen_importlib.FrozenImporter'>
When looking for "some_definition" it finds None

This finder is <class '_frozen_importlib_external.PathFinder'>
When looking for "some_definition" it finds ModuleSpec(name='some_definition', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7fa37c57b3d0>, origin='/home/jacopo/Documents/focc/python-project-organization/bag_of_scripts/some_definition.py')

This finder is <class '__editable___ligo_skymap_1_0_6_dev1_gf146202_finder._EditableFinder'>
When looking for "some_definition" it finds None
```

