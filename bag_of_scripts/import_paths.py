import sys
print(f'The path is:')
print('\n'.join(sys.path))
for finder in sys.meta_path:
    print()
    print(f'This finder is {finder}')
    print(f'When looking for "some_definition" it finds {finder.find_spec("some_definition", sys.path)}')
