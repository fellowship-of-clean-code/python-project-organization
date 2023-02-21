def import_local_from():
    from . import multiply_by
    
    decuple = multiply_by.MultiplyBy(10)
    print(decuple.compute(2) == 20)
    

def import_local():
    from .multiply_by import MultiplyBy
    
    decuple = MultiplyBy(10)
    print(decuple.compute(2) == 20)

def import_local_as():
    from . import multiply_by as mb
    
    decuple = mb.MultiplyBy(10)
    print(decuple.compute(2) == 20)


def import_local_nasty():
    pass
    """
    This one is commented out since it's not allowed syntax inside 
    a function, and also because you shouldn't do it:
    
    from .multiply_by import *
    
    decuple = MultiplyBy(10)
    print(decuple.compute(2) == 20)
    """

if __name__ == '__main__':
    import_local_from()
    import_local()
    import_local_as()