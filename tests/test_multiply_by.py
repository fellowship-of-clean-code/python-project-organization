from python_project.multiply_by import MultiplyBy

def test_multiply_by():

    quintuple = MultiplyBy(5)

    assert quintuple.compute(1) == 5
    assert quintuple.compute_twice(1) == 25
